"""Sum of Nuclear Norms, Latent Overlapping Grouped Norms Decomposition Model.

This 
"""

from time import perf_counter
from math import prod

import torch
import torch.linalg as la
import matplotlib.pyplot as plt

from src.multilinear_ops.matricize import matricize
from src.multilinear_ops.tensorize import tensorize
from src.proximal_ops.soft_hosvd import soft_moden
from src.proximal_ops.prox_overlapping_grouped_l21 import group_indicator_matrix



class SNN_LOGS:
    """Sum of Nuclear Norms, Latent Overlapping Grouped Norms Decomposition Model.
    """
    @torch.no_grad
    def __init__(self, Y, G, lr_modes, graph_modes,
                 grouping='neighbor', weighing='size_normalized_inv', group_norm_type='l2',
                 **kwargs):
        """Initialize SNN-LOGS decomposition with low-rank modes and (overlapping) grouped sparsity.

        Args:
            Y (torch.Tensor): Tensor to be decomposed.
            G (nx.Graph): Graph structure to define (overlapping) groups
            lr_modes (list of int): List of low-rank modes. 
            graph_modes (list of int): The dimensions over which the graph spans
            grouping (str, optional): Grouping strategy. Defaults to 'neighbor'.
                Options:
                    'edge': creates groupings as the pair of nodes connected by an edge
                    'neighbor': creates groupings with r-hop neighbours of each node.
                        r is specified by the 'r_hop' parameter.
                    'neighbor_plus_center': creates groupings with r-hop neighbours of each node
                        and the node itself as a separate group.
                    'edge_plus_center': creates groupings as the pair of nodes connected by an edge
                        and each node as a separate group.
            weighing (str, optional): Weighing strategy. Defaults to 'size_normalized_inv'
            group_norm_type (str, optional): Type of group norm to use. Defaults to 'l2'.
                Options:
                    'l2': l2 norm of the group
                    'l_inf': l_inf norm of the group (Not Implemented Yet)
                Options:
                    'size_normalized': weights the group norms by the square root of the group size
                    'group_size': weights the group norms by the group size
                    'uniform': weights the group norms uniformly
                    'size_normalized_inv': weights the group norms by the inverse of the square root of the group size
                    torch.Tensor: weights the group norms by a custom weighting tensor. 
                        The tensor must have the same length as the number of groups.
            kwargs: Additional parameters for the optimization algorithm.
                r_hop (int): Radius for the grouping strategy 'neighbor'
                device (str): Device to run the model on
                dtype (torch.dtype): Data type to use
                verbose (int): Verbosity level
                metric_tracker (MetricTracker): Metric tracker to log metrics, 
                    find information in src/utils/metric_tracker.py
                report_freq (int): Frequency to report metrics
        """
        self.device = kwargs.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.dtype = kwargs.get('dtype', torch.double)
        if isinstance(Y, torch.Tensor):
            self.Y = Y.to(device=self.device, dtype=self.dtype)          # observation matrix
        else:
            self.Y = torch.tensor(Y, device=self.device, dtype=self.dtype)
        self.lr_modes = lr_modes
        self.N = len(self.lr_modes)
        self.graph_modes = graph_modes
        self.graph = G
        self.r_hop = kwargs.get('r_hop', 1)
        self.grouping = grouping
        self.weighing = weighing
        self.group_norm_type = group_norm_type
        self._initialize_groupings()

        # Bookkeeping
        self.it = 0
        self.converged = False
        self.times = {"X":[], "V":[],                               # First block variables
                      "Xi":[[] for _ in range(self.N)], "Z":[],     # Second block variables
                      "iteration":[]}
        self.obj = []                           # objective function
        self.lagrangian = []                    # lagrangian
        self.rhos = {'Xi': [[] for _ in range(self.N)], "V":[], 'f':[]}    # step size parameters
        self.r = []                             # norm of primal residual
        self.s = []                             # norm of dual residual
        self.rs = {'Xi': [[] for _ in range(self.N)], "V":[], 'f':[]} # norms of primal residuals
        self.ss = {'Xi': [[] for _ in range(self.N)], "V":[], 'f':[]} # norms of dual residuals
        self.verbose = kwargs.get('verbose', 1)
        self.report_freq = kwargs.get('report_freq', 1)
        self.metric_tracker = kwargs.get('metric_tracker', None)

        self.X = torch.zeros(Y.shape, device=self.device, dtype=self.dtype) # low-rank component
        self.S = torch.zeros(Y.shape, device=self.device, dtype=self.dtype) # sparse component

        self.psis = None
        self.lda = None
        self.err_tol = None
        self.rho_update = None
        self.rho_update_thr = None


    def __call__(self, psis, lda, max_iter=100, rho=0.01, err_tol=1e-6,
                        rho_update=1.1, rho_update_thr=100, **kwargs):
        """Run the [SNN]-[LOGN] decomposition ADMM algorithm

        Parameters:
        ----------
            psis (list of float): Nuclear norm regularization parameters
            lda (float): Latent Overlapping Group Norm regularization parameter
            max_iter (int, optional): Maximum number of iterations for optimization. Defaults to 100.
            rho (float, optional): Augmented lagrangian penalty parameter (AKA step size). Defaults to 0.01.
            err_tol (float, optional): Convergence criteria. Defaults to 1e-6.
            rho_update (float, optional): Step size update. Defaults to 1.
            rho_update_thr (int, optional): Step size update threshold. Defaults to 100.
        
        Returns:
        -------
            torch.Tensor: Low-rank component
            torch.Tensor: Sparse component
        """
        self.psis = psis
        self.lda = lda
        self.err_tol = err_tol
        self.rho_update = rho_update
        self.rho_update_thr = rho_update_thr
        for i in range(self.N):
            self.rhos['Xi'][i].append(rho)
        self.rhos['V'].append(rho)
        self.rhos['f'].append(rho)

        # Initialize Auxiliary Variables
        Xi = [torch.zeros(self.Y.shape, device=self.device, dtype=self.dtype)
                    for _ in range(self.N)]
        Zbar = matricize(torch.zeros_like(self.Y),  # Batch x 1 x |Vertices|: Dense tensor
                        self.graph_modes).t().unsqueeze(1)
        Vbar = torch.zeros_like(Zbar)               # Batch x 1 x |Vertices|: Dense tensor
        V = (self.expander * Zbar).coalesce()       # Batch x |Groups| x |Vertices|: Sparse tensor
        Z = (self.expander * Zbar).coalesce()       # Batch x |Groups| x |Vertices|: Sparse tensor
        
        # Initialize Dual Variables
        Gamma_vbar = torch.zeros_like(Zbar)
        Gamma_xi = [torch.zeros(self.Y.shape, device=self.device, dtype=self.dtype)
                        for _ in range(self.N)]
        Gamma_f = torch.zeros(self.Y.shape, device=self.device, dtype=self.dtype)
        while self.it< max_iter and not self.converged:
            rho_x = [self.rhos['Xi'][i][-1] for i in range(self.N)]
            rho_v = self.rhos['V'][-1]
            rho_f = self.rhos['f'][-1]
            it_start = perf_counter()
            # First Block Updates {X, V_1, ... V_G}
            # # Update X
            X_temp = sum([ rho_x[i]*(Xi[i]-Gamma_xi[i]/rho_x[i]) for i in range(self.N)])
            self.X = (X_temp + rho_f*(self.Y - self.S- Gamma_f/rho_f))/(sum(rho_x)+rho_f)
            
            v_start = perf_counter()
            self.times['X'].append(v_start-it_start)
            # # Update V
            Vtemp = Z - self.expander*(Gamma_vbar/rho_v) #self.expander*(Zbar - Vbar - Gamma_vbar/rho_v) + V
            if self.group_norm_type == 'l2':
                norms = Vtemp.coalesce().pow(2).sum(dim=2,keepdim=True).coalesce().to_dense().sqrt()
                # norms: Batch x |Groups| x 1
                scaling_factor = torch.where(norms > self.w*lda/rho_v, 1-self.w*lda/(norms*rho_v), 0)
                V = (Vtemp*scaling_factor)
            elif self.group_norm_type == 'l_inf':
                raise NotImplementedError("l_inf norm not implemented yet")
            # lda_0*weights/rho_v = lambda
            # prox_{lambda ||.||_inf}(Vtemp) = Vtemp - lambda* project_{||.||_1 <= 1}(Vtemp/lambda)
            # abs_Vtemp = Vtemp.abs()
            # l1_norms = abs_Vtemp.sum(dim=2, keepdim=True)
            
            # # # Update Vbar
            Vbar = torch.sum(V, dim=1, keepdim=True).to_dense()/self.D_G
            zbar_start = perf_counter()
            self.times['V'].append(zbar_start-v_start)

            # Second Block Updates {Z_1,...,Z_G, X_1, ..., X_N}
            # # Update Zs
            # # # Update Zbar
            Zbar_old = Zbar.clone().detach()
            Zbar = (matricize(rho_f*(self.Y - self.X - Gamma_f/rho_f),
                             self.graph_modes).t().unsqueeze(1) +
                             rho_v*(Vbar + Gamma_vbar/rho_v)
                             )/(rho_f*self.D_G + rho_v)
            # # # Update S
            self.S = tensorize( (self.D_G*Zbar).squeeze().t(),
                                self.Y.shape, self.graph_modes)
            # # # Update Zs
            Z_old = Z.clone().detach()
            Z = (V + self.expander*(Zbar - Vbar)).coalesce()

            self.ss['V'].append(rho_v*(Z-Z_old).coalesce().values().norm())
            self.ss['f'].append(rho_f*la.norm(self.D_G*(Zbar-Zbar_old)))
            self.times['Z'].append(perf_counter()-zbar_start)

            # # Update Xi
            obj = 0
            for i, mode in enumerate(self.lr_modes):
                xi_start = perf_counter()
                Xi_new, nuc_norm = soft_moden(self.X + Gamma_xi[i]/rho_x[i],
                                               psis[i]/rho_x[i], mode)
                self.ss['Xi'][i].append(la.norm(Xi_new - Xi[i])*rho_x[i])
                Xi[i] = Xi_new
                self.times['Xi'][i].append(perf_counter()-xi_start)
                obj += psis[i]*nuc_norm
            l2_norms = V.pow(2).sum(dim=2,keepdim=True).sqrt().to_dense()
            obj += lda*(self.w*l2_norms).sum()
            self.obj.append(obj.cpu().item())
            # Dual Variable Updates
            # # Update Gamma_vbar
            rv = Z - Z_old
            Gamma_vbar += rho_v*(Vbar - Zbar)
            self.rs['V'].append(rv.coalesce().values().norm())
            # # Update Gamma_xi
            for i in range(self.N):
                rx = self.X - Xi[i]
                Gamma_xi[i] += rho_x[i]*rx
                self.rs['Xi'][i].append(la.norm(rx))
            # # Update Gamma_f
            rf = self.X + self.S - self.Y 
            Gamma_f += rho_f*rf
            self.rs['f'].append(la.norm(rf))

            self.r.append(torch.sqrt( sum([rxi[-1]**2 for rxi in self.rs['Xi']]) +
                                     self.rs['V'][-1]**2 + self.rs['f'][-1]**2))
            self.s.append(torch.sqrt( sum([sxi[-1]**2 for sxi in self.ss['Xi']]) +
                                        self.ss['V'][-1]**2 + self.ss['f'][-1]**2))
            
            self._report_iteration()
            self._check_convergence()
            self._call_tracker()
            self.times['iteration'].append(perf_counter()-it_start)
            if not self.converged:
                self._update_step_size()
                self.it += 1
        return self.X, self.S


    def _report_iteration(self):
        if self.verbose > 0 and self.it % self.report_freq == 0:
            print(f"It-{self.it} \t# |r| = {self.r[-1]:.4e} \t|s| = {self.s[-1]:.4e} \t obj = {self.obj[-1]:.4e}")
            if self.verbose>1:
                for i in range(self.N):
                    print(f"\t# |r_x{i}| = {self.rs['Xi'][i][-1]:.4e} \t# |s_x{i}| = {self.ss['Xi'][i][-1]:.4e} \t# rho_x{i} = {self.rhos['Xi'][i][-1]:.4e}")
                print(f"\t# |r_v| = {self.rs['V'][-1]:.4e} \t# |s_v| = {self.ss['V'][-1]:.4e} \t# rho_v = {self.rhos['V'][-1]:.4e}")
                print(f"\t# |r_f| = {self.rs['f'][-1]:.4e} \t# |s_f| = {self.ss['f'][-1]:.4e} \t# rho_f = {self.rhos['f'][-1]:.4e}")


    def _check_convergence(self):
        if self.s[-1] < self.err_tol and self.r[-1] < self.err_tol:
            self.converged = True
            if self.verbose > 1:
                print(f"Converged in {self.it} iterations.")
        return self.converged


    def _call_tracker(self):
        """Tracks metrics of the state variables of the algorithm using a MetricTracker object.
        
        Please see the Tracker class for more information.
        """
        if self.metric_tracker is not None:
            if self.it % self.metric_tracker.tracker_frequency == 0 or self.converged:
                self.metric_tracker.track(self)


    def _update_step_size(self):
        if self.rho_update <1:
            raise ValueError("Step size growth factor must be larger than 1")
        if self.rho_update != 1.0:
            if self.rs['V'][-1] > self.rho_update_thr * self.ss['V'][-1]:
                self.rhos['V'].append(self.rhos['V'][-1]*self.rho_update)
            elif self.ss['V'][-1] > self.rho_update_thr * self.rs['V'][-1]:
                self.rhos['V'].append(self.rhos['V'][-1]/self.rho_update)
            if self.rs['f'][-1] > self.rho_update_thr * self.ss['f'][-1]:
                self.rhos['f'].append(self.rhos['f'][-1]*self.rho_update)
            elif self.ss['f'][-1] > self.rho_update_thr * self.rs['f'][-1]:
                self.rhos['f'].append(self.rhos['f'][-1]/self.rho_update)
            for i in range(self.N):
                if self.rs['Xi'][i][-1] > self.rho_update_thr * self.ss['Xi'][i][-1]:
                    self.rhos['Xi'][i].append(self.rhos['Xi'][i][-1]*self.rho_update)
                elif self.ss['Xi'][i][-1] > self.rho_update_thr * self.rs['Xi'][i][-1]:
                    self.rhos['Xi'][i].append(self.rhos['Xi'][i][-1]/self.rho_update)


    def _initialize_groupings(self):
        """Initialize groupings based on the specified strategy"""
        G_ind, weights = group_indicator_matrix(self.graph, grouping=self.grouping,
                                                weighing='size_normalized',
                                                r_hop=self.r_hop,
                                                device='cpu')
        G_ind = G_ind.to_sparse_coo().to(device=self.device, dtype=torch.float64)
        self.nog = G_ind.shape[0]   # Number of groups
        self.nov = G_ind.shape[1]   # Number of vertices
        self.batch_dim = prod(self.Y.shape)//prod([self.Y.shape[i-1] for i in self.graph_modes])

        if self.weighing == 'size_normalized':
            self.w = weights.to_dense().to(device=self.device, dtype=torch.float64)
        elif self.weighing == 'size_normalized_inv':
            self.w = (1/weights.to_dense()).to(device=self.device, dtype=torch.float64)
        elif self.weighing == 'uniform':
            self.w = torch.ones_like(weights.to_dense()).to(device=self.device, dtype=torch.float64)
        elif isinstance(self.weighing, torch.Tensor):
            self.w = self.weighing.to(device=self.device, dtype=torch.float64)
            if self.w.shape[0] != self.nog:
                raise ValueError("The custom weighting tensor must have"+\
                                 " the same length as the number of groups")
        self.w = self.w.reshape((1,self.nog,1))

        ind = G_ind.indices()
        indices = torch.cat([#torch.zeros((1,ind.shape[1]),
                             #            dtype=torch.int64, device=self.device),
                            ind],
                            dim=0)
        self.expander =  torch.sparse_coo_tensor(indices,
                                                 torch.ones(indices.shape[1],
                                                             dtype=self.dtype, device=self.device),
                                                # size=(1, self.nog, self.nov),
                                                size=(self.nog, self.nov),
                                                device=self.device, dtype=self.dtype).coalesce()
        self.D_G = torch.sum(G_ind, dim=0).to(
                                            device=self.device, dtype=self.dtype
                                            ).to_dense().reshape((1,1,self.nov))
    
    # def bayesian_information_criterion(self, threshold=1e-8):
    #     """Calculates the Bayesian Information Criterion (BIC) of the model.
        
    #     BIC= 2*NLL(X,S) + k*log(N)
    #     where NLL(X,S) is the negative log-likelihood of the model,
    #     k is the number of parameters in the model,
    #     and N is the number of observations in the data.

    #     k = (# non-zero groups) * (group size) +
    #         + sum_{m \in modes} (rank(X_m)*(X_m.shape[0] + X_m.shape[1]) - rank(X_m)**2 )
        
    #     NLL(X,S) = sum_{m \in modes} \psi_m*||X_m||_* + \lambda*||S||_{LOGN}
    #     """
    #     N = self.Y.numel()
    #     bic = 0
    #     nll = 0
    #     k = 0
    #     for i,m in enumerate(self.lr_modes):
    #         nm = self.Xi.shape[m-1]
    #         sv = la.svd(matricize(self.Xi[i], [m]), compute_uv=False)
    #         r = (sv>threshold*sv.max()).sum()
    #         k += r*(nm + N//nm) - r**2
    #         nll += self.psis[i]*sv.sum()
    #     # norms: Batch x |Groups| x 1
    #     norms = Vtemp.coalesce().pow(2).sum(dim=2,keepdim=True).coalesce().to_dense().sqrt()
    #     scaling_factor = torch.where(norms > self.w*lda/rho_v, 1-self.w*lda/(norms*rho_v), 0)




    def plot_alg_run(self, figsize=(6,6)):
        """Plots the algorithm log in 2x2 subplots."""
        fig, axs = plt.subplots(1, 4, figsize=figsize)
        axs[0].plot(self.obj)
        axs[0].set_title('Objective function')
        axs[1].plot(self.r)
        axs[1].set_title('Primal residual')
        axs[2].plot(self.s)
        axs[2].set_title('Dual residual')
        axs[3].plot(self.rhos)
        axs[3].set_title('Step size')
        for ax in axs:
            ax.grid()
        return fig, axs