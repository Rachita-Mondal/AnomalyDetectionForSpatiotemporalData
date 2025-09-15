from time import perf_counter

import numpy as np
import torch
import torch.linalg as LA
from torch.nn.functional import softshrink
import matplotlib.pyplot as plt

from src.multilinear_ops.m2t import m2t
from src.multilinear_ops.t2m import t2m
from src.proximal_ops.soft_hosvd import soft_moden

class GRHoRPCA:
    r"""Solves the Graph Regularized Higher order Robust Principal Component Analysis problem with ADMM.

    The algorithm solves the following optimization problem following the
    singleton model proposed by Goldfarb and Qin (2014):
        min \sum_{i=1}^N( \alpha_i tr(Z_{(i)^T L_i Z_{(i)}} )  psi_{i}||Z_{(i)}||_*) + lda||S||_1
        s.t. Z + S = Y
    where X is the low-rank and smooth tensor and S is the sparse tensor and X_{(i)} is the
    matricization of X in the i'th mode. L_i is the graph laplacian of the i'th mode.
    """
    @torch.no_grad
    def __init__(self, Y, Ls, **kwargs):
        """Initializes the HoRPCA algorithm with the given parameters.

        Args:
            Y (np.ndarray, torch.Tensor): Observed tensor data. If the data is masked, Tensor completion is performed.
            Ls (list of np.ndarray, torch.Tensor): List of laplacian matrices of the graph structures.
            **kwargs: Additional parameters for the algorithm.
                lda (float): Sparsity regularization parameter. Defaults to 1/sqrt(max(n)).
                psis (list of float): List of nuclear norm regularization parameters for each mode. Defaults to [1]*N.
                alphas (list of float): List of smoothness regularization parameters for each mode. Defaults to [1]*N.
                rho (float): Step size of the ADMM algorithm. Defaults to 0.01.
                err_tol (float): Convergence criteria for the algorithm. Defaults to 1e-5.
                maxit (int): Maximum number of iterations allowed for the algorithm. Defaults to 100.
                step_size_growth (float): Step size update coefficient of the ADMM algorithm. Defaults to 1.2.
                    If set to 1, disables step size update.
                mu (float): Step size update treshold of the ADMM algorithm
                verbose (int): Verbosity level of the algorithm.
                device (str): Device to run the algorithm. Defaults to 'cuda' if available.
                dtype (torch.dtype): Data type of the tensors. Defaults to torch.float32.
                lr_modes (list of int): List of modes to apply low-rank regularization.
                smooth_modes (list of int): List of modes to apply smoothness regularization.
                report_freq (int): Frequency of reporting the algorithm progress.
                metric_tracker (Tracker): Tracker object to track metrics over algorithms progress.
                    Please see the Tracker class for more information.
                
        """
        self.dtype = kwargs.get('dtype', torch.float32)
        self.device = kwargs.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        if isinstance(Y, np.ndarray):
            if hasattr(Y, 'mask'):
                self.partially_observed = True
                self.obs = torch.tensor(~Y.mask, device=self.device, dtype=bool)
                self.unobs = torch.tensor(Y.mask, device=self.device, dtype=bool)
                self.Y = torch.tensor(Y, device=self.device, dtype=self.dtype) # Y.get_data().to(self.device, self.dtype)
            else:
                self.partially_observed = False
                self.Y = torch.tensor(Y, device=self.device, dtype=self.dtype)
        else:
            if hasattr(Y, 'get_mask'):
                self.partially_observed = True
                self.obs = torch.tensor(~Y.mask, device=self.device, dtype=bool)
                self.unobs = torch.tensor(Y.mask, device=self.device, dtype=bool)
            else:
                self.partially_observed = False
            self.Y = Y.to(self.device, self.dtype)
        
        self.n = Y.shape
        self.lr_modes = kwargs.get('lr_modes', [i+1 for i in range(len(self.n))])
        self.Ls = [torch.tensor(L, device=self.device, dtype=self.dtype) for L in Ls]
        self.smooth_modes = kwargs.get('smooth_modes', [i+1 for i in range(len(self.Ls))])
        self.N = len(self.lr_modes)
        self.M = len(self.smooth_modes)
        # Hyperparameters
        self.lda = kwargs.get('lda', 1/np.sqrt(np.max(self.n))) # Sparsity regularization parameter
        self.psis = kwargs.get('psis', [1]*self.N)              # Tensor tucker rank regularization parameters
        self.alphas = kwargs.get('alphas', [1]*self.M)          # Smoothness regularization parameters
        # Optimization variables
        self.Z = torch.zeros(self.n, dtype=self.dtype, device=self.device)
        self.S = torch.zeros(self.n, dtype=self.dtype, device=self.device)
        
        # Convergence criteria and ADMM step size
        self.obj = []    # objective function value
        self.r = []      # primal residual
        self.s = []      # dual residual
        self.rhos = []   # step size
        self.it = 0      # number of iterations
        self.rho = kwargs.get('rho', 0.01)   # Step size of ADMM
        self.converged = False
        self.err_tol = kwargs.get('err_tol', 1e-5)
        self.maxit = kwargs.get('maxit', 100)
        self.step_size_growth = kwargs.get('step_size_growth', 1)
        self.mu = kwargs.get('mu', 100)
        ## Algorithm and metric tracking
        self.verbose = kwargs.get('verbose', 1)
        self.report_freq = kwargs.get('report_freq', 1)        
        self.times = {'Z': [], # Track how long each step takes
                      'Xi': [[] for _ in range(self.M)],
                      'Wi': [[] for _ in range(self.N)],
                      'S': [],
                      'Ldai': [[] for _ in range(self.M)],
                      'Gammai': [[] for _ in range(self.N)],
                      'Lda': [],
                      'iteration':[]}
        self.metric_tracker = kwargs.get('metric_tracker', None)

    @torch.no_grad
    def __call__(self):
        """Solves the HoRPCA optimization problem

        Returns:
            Z (torch.Tensor): Low-rank tensor.
            S (torch.Tensor): Sparse tensor.
        """
        Xi = [torch.zeros(self.n, dtype=self.dtype, device=self.device) for _ in range(self.M)]
        Ldai = [torch.zeros(self.n, dtype=self.dtype, device=self.device) for _ in range(self.M)]
        Wi = [torch.zeros(self.n, dtype=self.dtype, device=self.device) for _ in range(self.N)]
        Gammai = [torch.zeros(self.n, dtype=self.dtype, device=self.device) for _ in range(self.N)]
        Z_temp = torch.zeros(self.n, dtype=self.dtype, device=self.device)
        Lda = torch.zeros(self.n, dtype=self.dtype, device=self.device)
        r = torch.zeros(self.n, dtype=self.dtype, device=self.device) # Used to calculate primal residual
        r_res = torch.zeros(1, dtype=self.dtype, device=self.device)  # Used to calculate primal residual
        while self.it < self.maxit and not self.converged:
            ## {Z} Block update ================================================
            # Z Update ---------------------------------------------------------
            zstart = perf_counter()
            Z_temp = sum([Wi[i] - Gammai[i]/self.rho for i in range(self.N)])
            Z_temp += sum([Xi[i] - Ldai[i]/self.rho for i in range(self.M)])
            if self.partially_observed:
                Z_temp[self.obs] += self.Y[self.obs] - self.S[self.obs] - Lda[self.obs]/self.rho
                Z_temp[self.obs] /= (self.N + self.M + 1)
                Z_temp[self.unobs] /= (self.N + self.M)
                dual_residual_norm = (LA.norm(Z_temp - self.Z)**2)*(self.N+self.M)
                dual_residual_norm += LA.norm(Z_temp[self.obs] - self.Z[self.obs])**2
                self.s.append(torch.sqrt(dual_residual_norm).item())
            else:
                Z_temp += self.Y - self.S - Lda/self.rho
                Z_temp /= (self.N + self.M + 1)
                self.s.append( torch.sqrt((LA.norm(Z_temp - self.Z)**2)*(self.N + self.M + 1)).item())
            self.Z, Z_temp = Z_temp, self.Z
            self.times['Z'].append(perf_counter() - zstart)

            ## {X_1,...,X_M,W_1,...,W_N,S} Block updates ====================================
            # Wi Update --------------------------------------------------------
            objective = 0
            for i in range(self.N):
                wistart = perf_counter()
                Wi[i], nuc_norm = soft_moden(self.Z + Gammai[i], self.psis[i]/self.rho, self.lr_modes[i])
                objective += nuc_norm*self.psis[i]
                self.times['Wi'][i].append(perf_counter() - wistart)
            # Xi Update --------------------------------------------------------
            for i in range(self.M):
                xistart = perf_counter()
                Xi[i] = m2t(
                    LA.solve(
                    self.alphas[i]*self.Ls[i] + self.rho*torch.eye(self.Ls[i].shape[0], device=self.device, dtype=self.dtype), 
                                 self.rho*t2m(self.Z + Ldai[i]/self.rho, self.smooth_modes[i])
                                 ), self.n, self.smooth_modes[i])
                            
                objective += 0.5*self.alphas[i]* torch.einsum('ij,jk,ki', 
                                             t2m(Xi[i], self.smooth_modes[i]).T, self.Ls[i], t2m(Xi[i], self.smooth_modes[i])).item()
                self.times['Xi'][i].append(perf_counter() - xistart)

            # S Update ---------------------------------------------------------
            sstart = perf_counter()
            if self.partially_observed:
                self.S[self.obs] = softshrink(self.Y[self.obs] - self.Z[self.obs] - Lda[self.obs]/self.rho, self.lda/self.rho)
                objective += torch.sum(torch.abs(self.S[self.obs]))*self.lda
            else:
                self.S = softshrink(self.Y - self.X - Lda/self.rho, self.lda/self.rho)
                objective += torch.sum(torch.abs(self.S))*self.lda
            self.times['S'].append(perf_counter() - sstart)
            self.obj.append(objective.item())

            ## Dual variable updates ===========================================
            # Lda Update -------------------------------------------------------
            r_res.fill_(0)
            r.fill_(0)
            lda_start = perf_counter()
            if self.partially_observed:
                r[self.obs] = self.Z[self.obs] + self.S[self.obs] - self.Y[self.obs]
                Lda[self.obs] += self.rho*r[self.obs]
            else:
                r = self.Z + self.S - self.Y
                Lda += self.rho*r
            self.times['Lda'].append(perf_counter() - lda_start)
            r_res += LA.norm(r)**2

            # Gammai Update ------------------------------------------------------
            for i in range(self.N):
                gammai_start = perf_counter()
                r = self.Z - Wi[i]
                Gammai[i] += self.rho*r
                r_res += LA.norm(r)**2
                self.times['Gammai'][i].append(perf_counter() - gammai_start)
            r_res = torch.sqrt(r_res)
            self.r.append(r_res.item())

            # Ldai Update ------------------------------------------------------
            for i in range(self.M):
                ldai_start = perf_counter()
                r = self.Z - Xi[i]
                Ldai[i] += self.rho*r
                self.times['Ldai'][i].append(perf_counter() - ldai_start)
                r_res += LA.norm(r)**2
            r_res = torch.sqrt(r_res)
            self.r.append(r_res.item())

            ## End Iteration ===================================================
            self._report_iteration()
            self._check_convergence()
            self._call_tracker()
            self.times['iteration'].append(perf_counter() - zstart)
            if not self.converged:
                self._update_step_size()
                self.it += 1
        return self.Z, self.S


    def _check_convergence(self):
        if self.it > 0:
            if self.s[-1] < self.err_tol and self.r[-1] < self.err_tol:
                self.converged = True
                if self.verbose > 0:
                    print(f'Converged in {self.it} iterations.')
        return self.converged
    
    def _update_step_size(self):
        """Updates the step size of the ADMM algorithm based on the residuals.

        The step size is updated based on the residuals of the primal and dual variables.
        If the ratio of the residuals is larger than the threshold mu, the step size is increased.
        If the ratio of the residuals is smaller than the threshold mu, the step size is decreased.
        """
        if self.step_size_growth <1:
            raise ValueError('Step size growth must be larger than 1')
        if self.step_size_growth != 1.0:
            if self.s[-1] > self.mu*self.r[-1]:
                self.rho *= self.step_size_growth
            elif self.r[-1] > self.mu*self.s[-1]:
                self.rho /= self.step_size_growth
            self.rhos.append(self.rho)
    
    def _call_tracker(self):
        """Tracks metrics of the state variables of the algorithm using a MetricTracker object.
        
        Please see the Tracker class for more information.
        """
        if self.metric_tracker is not None:
            if self.it % self.metric_tracker.tracker_frequency == 0 or self.converged:
                self.metric_tracker.track(self)


    def _report_iteration(self):
        if self.verbose > 0 and self.it>0 and self.it % self.report_freq == 0:
            print(f'It-{self.it:03d}\t({self.times["iteration"][-1]:.4f} sec.) --------- obj = {self.obj[-1]:.3e} --------- del_obj = {self.obj[-1]-self.obj[-2]:.3e}' )
            print(f'|r| = {self.r[-1]:.3e}   \t## |s| = {self.s[-1]:.3e}   \t## rho = {self.rho:.3e}')

    def plot_alg_run(self, figsize=(6,6)):
        """Plots the algorithm log in 2x2 subplots."""
        fig, axs = plt.subplots(2, 2, figsize=figsize)
        axs[0,0].plot(self.obj)
        axs[0,0].set_title('Objective function')
        axs[0,1].plot(self.r)
        axs[0,1].set_title('Primal residual')
        axs[1,0].plot(self.s)
        axs[1,0].set_title('Dual residual')
        axs[1,1].plot(self.rhos)
        axs[1,1].set_title('Step size')
        return fig, axs

    
class MetricTracker:
    """Metric tracker for tracking algorithm progress.

    Example:
    >>> def auc_roc(obj, **kwargs):
    >>>     labels = kwargs['labels']
    >>>     calculate_roc_auc(obj.S.ravel(), labels.ravel())
    >>>     return roc_auc
    >>>
    >>> def cardinality(obj):
    >>>     return torch.sum(obj.S != 0)
    >>>
    >>> def sparsity(obj):
    >>>     return torch.sum(obj.S == 0)/obj.S.numel()
    >>>
    >>> metric_functions = [auc_roc, cardinality]
    >>> external_inputs = {'auc_roc': {'labels': labels}}
    """
    def __init__(self, metric_functions, backend='torch', **kwargs):
        """Initializes the MetricTracker object.

        Args:
            metric_functions (list of functions): Pure functionals that take the algorithm object as input and return a scalar.
                Functions must be designed with the algorithm object in mind.
            external_inputs (dict): Dictionary of external inputs for each metric function. Keys must match the function names.
            backend (str, optional): _description_. Defaults to 'torch'.
        """
        self.backend = backend
        self.device = kwargs.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.metric_functions = metric_functions
        if backend == 'torch':
            self.metrics = {func.__name__: torch.tensor([], device=self.device) for func in metric_functions}
        else:
            self.metrics = {func.__name__: [] for func in metric_functions}
        self.external_inputs = kwargs.get('external_inputs', {})
        for metric_function in self.metric_functions:
            if metric_function.__name__ not in self.external_inputs:
                self.external_inputs[metric_function.__name__] = {}
        self.tracker_frequency = kwargs.get('tracker_frequency', 1)
        self.verbose = kwargs.get('verbose', 1)
        self.tb_writer = kwargs.get('tb_writer', None) # TensorBoard writer
    
    def track(self, obj):
        for func in self.metric_functions:
            stat = func(obj, **self.external_inputs[func.__name__])
            if self.verbose > 0:
                print(f'{func.__name__}: {stat:.4e}')
            
            if self.tb_writer is not None:
                self.tb_writer.add_scalar(func.__name__, stat, obj.it)
            else:
                if self.backend == 'torch':
                    self.metrics[func.__name__] = torch.cat((self.metrics[func.__name__], stat.unsqueeze(0)))
                else:
                    self.metrics[func.__name__].append(stat)
    
    def plot(self, **kwargs):
        if self.tb_writer is not None:
            error_msg = 'The metrics are not kept in the memory and written to disk via TensorBoard writer.'
            error_msg += f'\nThe metrics can be accessed via TensorBoard at {self.tb_writer.get_logdir()}'
            raise ValueError(error_msg)
        figsize = kwargs.get('figsize', (4*len(self.metric_functions), 4))
        fig, axs = plt.subplots(1, len(self.metric_functions), figsize=figsize)
        for i, func in enumerate(self.metric_functions):
            if self.backend == 'torch':
                axs[i].plot(self.metrics[func.__name__].cpu().numpy())
            else:
                axs[i].plot(self.metrics[func.__name__])
            axs[i].set_title(func.__name__)
            axs[i].grid()
        return fig, axs