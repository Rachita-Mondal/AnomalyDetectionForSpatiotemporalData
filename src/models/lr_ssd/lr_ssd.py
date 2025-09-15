"""Low-rank, Structured Sparse Decomposition (LR-SSD) model.

This module contains the implementation of the Low-rank, Structured Sparse
Decomposition (LR-SSD) model. The model is based on the following optimization
problem:

    min_{X, S} R_1(X) + R_2(S)
    such that X+S = Y

R_1(X) is a low-rank regularizer and R_2(S) is a structured sparse regularizer.
The model is solved using the Alternating Direction Method of Multipliers
(ADMM) algorithm.
"""

from time import perf_counter

import numpy as np
import torch
import torch.linalg as la
from torch.nn.functional import softshrink
from matplotlib import pyplot as plt

from src.multilinear_ops.matricize import matricize
from src.multilinear_ops.tensorize import tensorize
from src.proximal_ops.prox_overlapping_grouped_l21 import prox_overlapping_grouped_l21
from src.proximal_ops.soft_hosvd import soft_moden


class LR_SSD:
    """Low-rank, Structured Sparse Decomposition (LR-SSD) model"""

    @torch.no_grad
    def __init__(self, Y, B1, B2, Bt, **kwargs):
        """Initialize the LR-SSD model.

        Args:
            Y (torch.Tensor): Observation tensor
            B1 (torch.Tensor): Sparse oriented incidence matrix of the spatial graph
            B2 (torch.Tensor): Sparse oriented incidence matrix of Line Graph of the spatial graph
            **kwargs: Additional arguments
                lr_modes: List of modes to apply the low-rank regularizer
                graph_modes: List of modes to apply the graph regularizer
                group_weights (list of torch.Tensor): weights of the groups. Optional. 
                    Default is sqrt(|g|) where |g| is the number of nodes in the group.
                device (str): Device to use for computations. Default is 'cuda' if available, else 'cpu'.
                dtype (torch.dtype): Data type to use for computations. Default is torch.double.
                metric_tracker (MetricTracker): Metric tracker to log the metrics. Optional.
                report_freq (int): Frequency to track the metrics. Optional. Default is 1.
                verbose (int): Verbosity level. Optional. Default is 1.
        """
        
        self.device = kwargs.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.dtype = kwargs.get('dtype', torch.double)
        if isinstance(Y, torch.Tensor):
            self.Y = Y.to(device=self.device, dtype=self.dtype)          # observation matrix
        else:
            self.Y = torch.tensor(Y, device=self.device, dtype=self.dtype)
        if isinstance(B1, torch.Tensor):
            self.B1 = B1.to(device=self.device, dtype=self.dtype)        # sparse oriented incidence matrix of the spatial graph
        else:
            self.B1 = torch.tensor(B1, device=self.device, dtype=self.dtype)
        if isinstance(B2, torch.Tensor):
            self.B2 = B2.to(device=self.device, dtype=self.dtype)        # sparse oriented incidence matrix of Line Graph of the spatial graph
        else:
            self.B2 = torch.tensor(B2, device=self.device, dtype=self.dtype)
        if isinstance(Bt, torch.Tensor):
            self.Bt = Bt.to(device=self.device, dtype=self.dtype)        # sparse oriented incidence matrix of the temporal graph
        else:
            self.Bt = torch.tensor(Bt, device=self.device, dtype=self.dtype)
        
        self.lr_modes = kwargs.get('lr_modes', [i+1 for i in range(len(Y.shape))])
        self.graph_modes = kwargs.get('graph_modes', [1])
        self.time_mode = kwargs.get('time_mode', 2)
        self.N = len(self.lr_modes)
        
        self.G_ind = kwargs.get('G_ind', [None, None])
        self.G_ind_coo = kwargs.get('G_ind_coo', [None, None])
        self.G_ind_T = kwargs.get('G_ind_T', [None, None])
        self.group_weights = kwargs.get('group_weights', [None, None])
    
        # Bookkeeping
        self.times = {"X":[], "S0":[], "W1":[], "W2":[], "St":[],       # First block variables
                      "Xi":[[] for _ in range(self.N)], "S":[], "Wt":[], # Second block variables
                      "iteration":[]}
        self.gaps = [[],[]]
        self.obj = []                           # objective function
        self.rhos = {'Xi': [[] for _ in range(self.N)], "S0":[], "W1":[], "W2":[], 'St':[]}    # step size parameters
        self.r = []                             # norm of primal residual
        self.s = []                             # norm of dual residual
        self.rs = {'Xi': [[] for _ in range(self.N)], "S0":[], "W1":[], "W2":[], 'St':[]}     # norms of primal residuals
        self.ss = {'Xi': [[] for _ in range(self.N)], "S0":[], "W1":[], "W2":[], 'St':[]}     # norms of dual residuals
        self.verbose = kwargs.get('verbose', 1)
        self.report_freq = kwargs.get('report_freq', 1)
        self.metric_tracker = kwargs.get('metric_tracker', None)
        self.timeit = kwargs.get('timeit', False)

        # Dynamic step size parameters
        self.it = 0
        self.converged = False
        self.rho_update_thr = kwargs.get('rho_update_thr',100)
        self.rho_update = kwargs.get('rho_upd',1.1)
        self.verbose = kwargs.get('verbose',3)
        self.benchmark = kwargs.get('benchmark',True)
        self.err_tol = kwargs.get('err_tol', 1e-6)

        self.X = torch.zeros(Y.shape, device=self.device, dtype=self.dtype) # low-rank component
        self.S = torch.zeros(Y.shape, device=self.device, dtype=self.dtype) # sparse component
        self.B1T = B1.t().to_sparse_csr()
        self.B2T = B2.t().to_sparse_csr()
        self.BtT = Bt.t().to_sparse_csr()
        self.B1B1T = torch.matmul(B1, B1.t())
        self.B2B2T = torch.matmul(B2, B2.t())
        self.BtBtT = torch.matmul(Bt, Bt.t())
        self.It = torch.eye(self.BtBtT.shape[0], device=self.device, dtype=self.dtype)
        self.Il = torch.eye(self.B2B2T.shape[0], device=self.device, dtype=self.dtype)

    def __call__(self, rho=0.01, max_iter=100, **kwargs):
        """_summary_

        Args:
            rho (float, optional): _description_. Defaults to 0.01.
            max_iter (int, optional): _description_. Defaults to 100.

        Returns:
            _type_: _description_
        """
        # Hyper-parameters
        psis = kwargs.get('psis', [1 for _ in range(len(self.lr_modes))])
        lda_f = kwargs.get('lda_f', 1)
        lda_0 = kwargs.get('lda_0', 1)
        lda_1 = kwargs.get('lda_1', 1)
        lda_2 = kwargs.get('lda_2', 1)
        lda_g0 = kwargs.get('lda_g0', 0)
        lda_g1 = kwargs.get('lda_g1', 0)
        lda_t = kwargs.get('lda_t', 1)
        # === Initialization
        n = self.Y.shape
        N = self.N
        for key in self.rhos.keys():
            if key == 'Xi':
                for i in range(len(self.rhos[key])):
                    self.rhos[key][i].append(rho)
            else:
                self.rhos[key].append(rho)
        # === Auxiliary variables
        Xi = [torch.zeros(n, device=self.device, dtype=self.dtype) for _ in range(N)]
        S0 = torch.zeros(n, device=self.device, dtype=self.dtype)
        W1 = self.B1T @ matricize(torch.zeros(n, device=self.device, dtype=self.dtype), self.graph_modes)
        W2 = self.B2T @ matricize(torch.zeros(n, device=self.device, dtype=self.dtype), self.graph_modes)
        St = torch.zeros_like( matricize(self.S, [self.time_mode]))
        Wt = torch.zeros_like( self.BtT @ St)
        # === Dual variables
        Gamma_0 = torch.zeros(n, device=self.device, dtype=self.dtype)
        Gamma_1 = torch.zeros_like(W1)
        Gamma_2 = torch.zeros_like(W2)
        Gamma_t1 = torch.zeros_like(St) 
        Gamma_t2 = torch.zeros_like(Wt)
        Gamma_x = [torch.zeros(n, device=self.device, dtype=self.dtype) for _ in range(N)]
        Y_hot0 = None
        Y_hot1 = None
        while self.it < max_iter and not self.converged:
            rho_x = [self.rhos['Xi'][i][-1] for i in range(N)]
            rho_0 = self.rhos['S0'][-1]
            rho_1 = self.rhos['W1'][-1]
            rho_2 = self.rhos['W2'][-1]
            rho_t = self.rhos['St'][-1]
            ## First block updates ========================
            # Update X --------------------------
            xstart = perf_counter()
            X_temp = lda_f*(self.Y - self.S)
            for i, mode in enumerate(self.lr_modes):
                X_temp += rho_x[i]*(Xi[i] - Gamma_x[i]/rho_x[i])
            X_temp = X_temp/(lda_f + sum(rho_x))
            self.X = X_temp
            self.times['X'].append(perf_counter() - xstart)
            
            # Update S0 --------------------------
            s_start = perf_counter()
            S0 = matricize(self.S + Gamma_0/rho_0, self.graph_modes)
            if lda_g0 == 0:
                S0 = softshrink(S0, lda_0/rho_0)
                self.gaps[0].append(0)
            else:
                S0, gap0, _, _, Y_hot0 = prox_overlapping_grouped_l21(S0.t(), self.G_ind[0], lda_0/rho_0, lda_g0/rho_0,
                                        self.group_weights[0], hotstart=Y_hot0, G_ind_T=self.G_ind_T[0], 
                                        G_ind_coo=self.G_ind_coo[0], err_tol=1e-5, max_iter=50, 
                                        algo='APGM', attenuate='constant', verbose= self.verbose>2)
                self.gaps[0].append(gap0[-1])
                S0 = S0.t()
            S0 = tensorize(S0, n, self.graph_modes)
            self.times['S0'].append(perf_counter() - s_start)

            # Update W1 ---------------------------
            w1_start = perf_counter()
            W1 = self.B1T @ matricize(self.S, self.graph_modes) + Gamma_1/rho_1
            if lda_g1 == 0:
                W1 = softshrink(W1, lda_1/rho_1)
                self.gaps[1].append(0)
            else:
                W1, gap1, _, _, Y_hot1 = prox_overlapping_grouped_l21(W1.t(), self.G_ind[1], lda_1/rho_1, lda_g1/rho_1,
                                        self.group_weights[1], hotstart=Y_hot1, G_ind_T=self.G_ind_T[1], 
                                        G_ind_coo=self.G_ind_coo[1], err_tol=1e-5, max_iter=50,
                                        algo='APGM', attenuate='constant', verbose= self.verbose>2)
                self.gaps[1].append(gap1[-1])
                W1 = W1.t()
            self.times['W1'].append(perf_counter() - w1_start)
            # self.ss['W1'].append()

            # Update W2 ---------------------------
            w2_start = perf_counter()
            W2 = softshrink( self.B2T @ matricize(self.S, self.graph_modes) + Gamma_2/rho_2, lda_2/rho_2)
            self.times['W2'].append(perf_counter()-w2_start)

            # Update St ---------------------------
            st_start = perf_counter()
            St = la.solve(self.BtBtT + self.It, 
                        matricize(self.S, [self.time_mode])+ Gamma_t1/rho_t + self.Bt @ (Wt - Gamma_t2/rho_t))
            self.times['St'].append(perf_counter() - st_start)

            ## Second block updates ===========================================
            # Update Xi --------------------------
            obj = 0
            for i, mode in enumerate(self.lr_modes):
                xi_start = perf_counter()
                Xi_temp, nuc_norm = soft_moden(self.X + Gamma_x[i]/rho_x[i], psis[i]/rho_x[i], mode)
                obj += nuc_norm
                self.times['Xi'][i].append(perf_counter() - xi_start)
                self.ss['Xi'][i].append(la.norm(Xi_temp - self.X)*rho_x[i])
                Xi[i] = Xi_temp
            
            # Update S ---------------------------
            s_start = perf_counter()
            inv_term = (lda_f + rho_0 + rho_t)*self.Il + rho_1*self.B1B1T + rho_2*self.B2B2T
            S_temp = lda_f*matricize(self.Y - self.X, self.graph_modes)
            S_temp += rho_0*matricize(S0 - Gamma_0/rho_0, self.graph_modes)
            S_temp += rho_1*self.B1 @ (W1 - Gamma_1/rho_1)
            S_temp += rho_2*self.B2 @ (W2 - Gamma_2/rho_2)
            S_temp += rho_t*matricize(tensorize(St - Gamma_t1/rho_t, n, [self.time_mode]), self.graph_modes)
            S_temp = la.solve(inv_term, S_temp)
            self.times['S'].append(perf_counter() - s_start)
            S_dif = S_temp - matricize(self.S, self.graph_modes)
            s_dif_norm = la.norm(S_dif)
            self.ss['S0'].append(s_dif_norm*rho_0)
            self.ss['W1'].append(la.norm(self.B1T@S_dif)*rho_1)
            self.ss['W2'].append(la.norm(self.B2T@S_dif)*rho_2)
            self.S = tensorize(S_temp, n, self.graph_modes)

            # Update Wt --------------------------
            wt_start = perf_counter()
            Wt_temp = Wt.clone().detach()
            Wt = softshrink(self.BtT @ matricize(self.S, [self.time_mode]) + Gamma_t2/rho_t, lda_t/rho_t)
            self.times['St'].append(perf_counter() - wt_start)
            self.ss['St'].append( rho_t*torch.sqrt((s_dif_norm.pow(2)+ la.norm( self.Bt@(Wt - Wt_temp)).pow(2))))

            ## Dual updates ===========================================
            for i in range(N):
                r = self.X - Xi[i]
                Gamma_x[i] = Gamma_x[i] + rho_x[i]*r
                self.rs['Xi'][i].append(la.norm(r))

            r = self.S - S0
            Gamma_0 = Gamma_0 + rho_0*r
            self.rs['S0'].append(la.norm(r))

            r = self.B1T@ matricize(self.S, self.graph_modes) - W1
            Gamma_1 = Gamma_1 + rho_1*r
            self.rs['W1'].append(la.norm(r))

            r = self.B2T@ matricize(self.S, self.graph_modes) - W2
            Gamma_2 = Gamma_2 + rho_2*r
            self.rs['W2'].append(la.norm(r))

            r = matricize(self.S, [self.time_mode]) - St
            Gamma_t1 = Gamma_t1 + rho_t*r
            a = la.norm(r)

            r = self.BtT @ matricize(self.S, [self.time_mode]) - Wt
            Gamma_t2 = Gamma_t2 + rho_t*r
            self.rs['St'].append( torch.sqrt(la.norm(r).pow(2) + a.pow(2)))

            # obj += lda_f*la.norm(self.Y - self.X - self.S) + lda_0*la.vector_norm(S0, ord=1)\
            #     +
            r = []
            s = []
            for key in self.rs.keys():
                if key != 'Xi':
                    r.append( self.rs[key][-1].pow(2))
                    s.append( self.ss[key][-1].pow(2))
                else:
                    for i in range(N):
                        r.append( self.rs[key][i][-1].pow(2))
                        s.append( self.ss[key][i][-1].pow(2))
               
            self.r.append( torch.sqrt(sum(r)) )
            self.s.append( torch.sqrt(sum(s)) )
            ## End Iteration --------------------------------
            self._report_iteration()
            self._check_convergence()
            self._call_tracker()
            self.times['iteration'].append(perf_counter() - xstart)
            if not self.converged:
                self._update_step_size()
                self.it += 1

        return self.X, self.S
    
    def _report_iteration(self):
        if self.verbose > 0 and self.it % self.report_freq == 0:
            print(f'It-{self.it}   \t## |r| = {self.r[-1]:.3e}   \t## |s| = {self.s[-1]:.3e}')
        if self.verbose > 1 and self.it % self.report_freq == 0:
            for key in self.rs.keys():
                if key=='Xi':
                    for i in range(len(self.rs[key])):
                        print(f"|r_{key}_{i}| = {self.rs[key][i][-1]:.3e}  \t##|s_{key}_{i}| = {self.ss[key][i][-1]:.3e}  \t## rho_{key}_{i} = {self.rhos[key][i][-1]:.3e}")
                else:
                    print(f"|r_{key}| = {self.rs[key][-1]:.3e}  \t##|s_{key}| = {self.ss[key][-1]:.3e}  \t## rho_{key} = {self.rhos[key][-1]:.3e}")
        if self.verbose > 2 and self.it % self.report_freq == 0:
            print(f"Gap 1: {self.gaps[0][-1]:.3e}  \t## Gap 2: {self.gaps[1][-1]:.3e}")

    def _check_convergence(self):
        if self.it > 0:
            if self.s[-1] < self.err_tol and self.r[-1] < self.err_tol:
                self.converged = True
                if self.verbose > 0:
                    print(f'Converged in {self.it} iterations.')
        return self.converged
    
    def _call_tracker(self):
        """Tracks metrics of the state variables of the algorithm using a MetricTracker object.
        
        Please see the Tracker class for more information.
        """
        if self.metric_tracker is not None:
            if self.it % self.metric_tracker.tracker_frequency == 0 or self.converged:
                self.metric_tracker.track(self)

    def _update_step_size(self):
        """Updates the step size of the ADMM algorithm based on the residuals.

        The step size is updated based on the residuals of the primal and dual variables.
        If the ratio of the residuals is larger than the threshold mu, the step size is increased.
        If the ratio of the residuals is smaller than the threshold mu, the step size is decreased.
        """
        if self.rho_update <1:
            raise ValueError('Step size growth must be larger than 1')
        if self.rho_update != 1.0:
            for key in self.rhos.keys():
                if key == 'Xi':
                    for i in range(len(self.rhos[key])):
                        if self.rs[key][i][-1] > self.rho_update_thr*self.ss[key][i][-1]:
                            self.rhos[key][i].append( self.rhos[key][i][-1]* self.rho_update)
                        elif self.ss[key][i][-1] > self.rho_update_thr*self.rs[key][i][-1]:
                            self.rhos[key][i].append( self.rhos[key][i][-1]/ self.rho_update)
                else:
                    if self.rs[key][-1] > self.rho_update_thr*self.ss[key][-1]:
                        self.rhos[key].append( self.rhos[key][-1]* self.rho_update)
                    elif self.ss[key][-1] > self.rho_update_thr*self.rs[key][-1]:
                        self.rhos[key].append( self.rhos[key][-1]/ self.rho_update)


    def plot_alg_run(self, figsize=(6,6)):
        """Plots the algorithm log in 2x2 subplots."""
        fig, axs = plt.subplots(1, 3, figsize=figsize)
        r = torch.cat([a.unsqueeze(0) for a in self.r]).cpu().numpy()
        s = torch.cat([a.unsqueeze(0) for a in self.s]).cpu().numpy()
        # obj = torch.cat(self.obj).cpu().numpy()
        axs[0].plot(r)
        axs[0].set_title('Primal residual')
        axs[1].plot(s)
        axs[1].set_title('Dual residual')
        # axs[2].plot(obj)
        # axs[2].set_title('Objective function')
        for ax in axs:
            ax.grid(True)
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
