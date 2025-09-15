"""Low-rank, Latent Overlapping Grouped Sparse Decomposition Model.

This 
"""

from time import perf_counter
from math import prod

import torch
import torch.linalg as la

from src.multilinear_ops.matricize import matricize
from src.multilinear_ops.tensorize import tensorize
from src.proximal_ops.prox_overlapping_grouped_l21 import prox_overlapping_grouped_l21
from src.proximal_ops.soft_hosvd import soft_moden


class LR_LOGS:
    """Low-rank, Latent Overlapping Grouped Sparse Decomposition Model.
    """
    @torch.no_grad
    def __init__(self, Y, G_ind, group_weights, **kwargs):
        """_summary_

        Args:
            Y (torch.Tensor): Observed data tensor
            G_ind (torch.Tensor): Sparse grouping indicator tensor of shape (nog, nov)
            **kwargs: Additional keyword arguments
                lr_modes (list): List of modes to be considered for the low-rank component
                graph_modes (list): List of modes to be considered for the graph structure
                device (str): Device to run the model on
                dtype (torch.dtype): Data type to be used
                group_weights (torch.Tensor): Weights for the groups

        """
        self.device = kwargs.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.dtype = kwargs.get('dtype', torch.double)
        if isinstance(Y, torch.Tensor):
            self.Y = Y.to(device=self.device, dtype=self.dtype)          # observation matrix
        else:
            self.Y = torch.tensor(Y, device=self.device, dtype=self.dtype)
        
        
        self.lr_modes = kwargs.get('lr_modes', [i+1 for i in range(len(Y.shape))])
        self.graph_modes = kwargs.get('graph_modes', [1])
        self.N = len(self.lr_modes)
        self.group_norm_type = kwargs.get('group_norm_type', 'l2')

        self.nog = G_ind.shape[0] # Number of groups
        self.nov = G_ind.shape[1] # Number of vertices
        self.batch_dim = prod(Y.shape)//prod([Y.shape[i-1] for i in self.graph_modes])
        if G_ind.layout is not torch.sparse_coo:
            G_ind = G_ind.to_sparse_coo()
        self.G_ind = G_ind.to(device=self.device, dtype=self.dtype)     # grouping indicator tensor
        self.group_weights = group_weights
        ind = G_ind.indices()
        indices = torch.cat([torch.zeros((1,ind.shape[1]), 
                                         dtype=torch.int64, device=self.device),
                            ind],
                            dim=0)
        self.expander =  torch.sparse_coo_tensor(indices, 
                                                 torch.ones(indices.shape[1],
                                                             dtype=self.dtype, device=self.device), 
                                                size=(1, self.nog, self.nov), 
                                                device=self.device, dtype=self.dtype).coalesce()
        
        # Bookkeeping
        self.it = 0
        self.verbose = kwargs.get('verbose',3)
        self.benchmark = kwargs.get('benchmark',True)
        self.err_tol = kwargs.get('err_tol', 1e-6)
        self.converged = False
        self.times = {"X":[], "V":[],                               # First block variables
                      "Xi":[[] for _ in range(self.N)], "Zbar":[],  # Second block variables
                      "iteration":[]}
        self.obj = []                           # objective function
        self.rhos = {'Xi': [[] for _ in range(self.N)], "V":[]}    # step size parameters
        self.r = []                             # norm of primal residual
        self.s = []                             # norm of dual residual
        self.rs = {'Xi': [[] for _ in range(self.N)], "V":[]}     # norms of primal residuals
        self.ss = {'Xi': [[] for _ in range(self.N)], "V":[]}     # norms of dual residuals
        self.verbose = kwargs.get('verbose', 1)
        self.report_freq = kwargs.get('report_freq', 1)
        self.metric_tracker = kwargs.get('metric_tracker', None)
        self.timeit = kwargs.get('timeit', False)

        # Dynamic step size parameters    
        self.rho_update_thr = kwargs.get('rho_update_thr',100)
        self.rho_update = kwargs.get('rho_upd',1.1)
        
        self.X = torch.zeros(Y.shape, device=self.device, dtype=self.dtype) # low-rank component
        self.S = torch.zeros(Y.shape, device=self.device, dtype=self.dtype) # sparse component
    
    @torch.no_grad
    def __call__(self, rho=0.01, max_iter=100, **kwargs):
        # Hyper-parameters
        psis = kwargs.get('psis', [1 for _ in self.lr_modes])
        lda_f = kwargs.get('lda_f', 100)
        lda_0 = kwargs.get('lda_0', 0.01)
        weights = self.group_weights.to_dense().reshape((1, self.nog, 1))
        for i in range(self.N):
            self.rhos['Xi'][i].append(rho)
            self.rhos['V'].append(rho)
        # Initialize auxiliary variables
        Xi = [torch.zeros(self.Y.shape, device=self.device, dtype=self.dtype) for _ in range(self.N)]
        Zbar = matricize(torch.zeros_like(self.Y),  # Batch x 1 x |Vertices|: Dense tensor 
                        self.graph_modes).t().reshape((self.batch_dim, 1, self.nov))
        V = (self.expander * Zbar).coalesce()# Batch x |Groups| x |Vertices|: Sparse tensor
        
        ## Dual variables:
        Ubar = torch.zeros_like(Zbar)
        Gamma_xi = [torch.zeros(self.Y.shape, device=self.device, dtype=self.dtype) for _ in range(self.N)]
        
        while self.it< max_iter and not self.converged:
            rho_x = [self.rhos['Xi'][i][-1] for i in range(self.N)]
            rho_v = self.rhos['V'][-1]
            it_start = perf_counter()
            # First block updates
            # # Update X
            X_temp = sum([rho_x[i]*(Xi[i]- Gamma_xi[i]/rho_x[i]) for i in range(self.N)])
            X_temp += lda_f*(self.Y-self.nog*tensorize(Zbar.squeeze().t(), self.Y.shape, self.graph_modes))
            self.X = X_temp/(sum(rho_x)+lda_f)
            self.times['X'].append(perf_counter()-it_start)
            # # Update V
            v_start = perf_counter()
            Vbar = (torch.sum(V, dim=1, keepdim=True)/self.nog).to_dense()
            Vtemp = self.expander*( Zbar - Vbar - Ubar) + V
            if self.group_norm_type == 'l2':
                norms = Vtemp.pow(2).sum(dim=2,keepdim=True).sqrt().to_dense()
                # scaling_factor = torch.where(norms > weights*lda_0, 1-weights*lda_0/(norms*rho_v), 0)
                scaling_factor = torch.where(norms > weights*lda_0/rho_v, 1-weights*lda_0/(norms*rho_v), 0)
                V = Vtemp*scaling_factor
            elif self.group_norm_type == 'l_inf':
                raise NotImplementedError("l_inf norm not implemented yet")
                # lda_0*weights/rho_v = lambda
                # prox_{lambda ||.||_inf}(Vtemp) = Vtemp - lambda* project_{||.||_1 <= 1}(Vtemp/lambda)
                # abs_Vtemp = Vtemp.abs()
                # l1_norms = abs_Vtemp.sum(dim=2, keepdim=True)

            self.times['V'].append(perf_counter()-v_start)

            # Second block updates
            # # Update Xi
            for i in range(self.N):
                xi_start = perf_counter()
                Xi_new, nuc_norm = soft_moden(self.X + Gamma_xi[i]/rho_x[i],
                                               psis[i]/rho_x[i], self.lr_modes[i])
                self.ss['Xi'][i].append(la.norm(Xi_new - Xi[i])*rho_x[i])
                Xi[i] = Xi_new
                self.times['Xi'][i].append(perf_counter()-xi_start)
            
            # # Update Zbar
            zbar_start = perf_counter()
            Zbar_old = Zbar.clone().detach()
            Zbar = matricize(lda_f*(self.Y - self.X), self.graph_modes).t().unsqueeze(1) + \
                    rho_v*(torch.sum(V, dim=1, keepdim=True).to_dense()/self.nog + Ubar)
            Zbar /= (self.nog*lda_f + rho_v)
            Vbar = (torch.sum(V, dim=1, keepdim=True)/self.nog).to_dense()
            self.times['Zbar'].append(zbar_start - perf_counter())
            
            self.ss['V'].append(la.norm((Zbar - Zbar_old))*self.nog*rho_v)
            self.S = tensorize((self.nog*Zbar).squeeze().t(), 
                               self.Y.shape, self.graph_modes)
            

            # Update dual variables
            # # Update Gamma_xi
            for i in range(self.N):
                r = (self.X-Xi[i])
                Gamma_xi[i] = Gamma_xi[i] + rho_x[i]*(self.X-Xi[i])
                self.rs['Xi'][i].append(la.norm(r))
            
            # # Update Ubar
            r = Vbar - Zbar
            Ubar = Ubar + rho_v*r
            self.rs['V'].append(la.norm(r)*self.nog)

            self.r.append(torch.sqrt( sum([rxi[-1]**2 for rxi in self.rs['Xi']]) + self.rs['V'][-1]**2))
            self.s.append(torch.sqrt( sum([sxi[-1]**2 for sxi in self.ss['Xi']]) + self.ss['V'][-1]**2))

            self._report_iteration()
            self._check_convergence()
            self._call_tracker()
            self.times['iteration'].append(perf_counter()- it_start)
            if not self.converged:
                self._update_step_size()
                self.it += 1
        
        return self.X, self.S
    
    def _report_iteration(self):
        if self.verbose > 0 and self.it % self.report_freq == 0:
            print(f"It-{self.it} \t# |r| = {self.r[-1]:.4e} \t|s| = {self.s[-1]:.4e}")
            if self.verbose>1:
                for i in range(self.N):
                    print(f"\t# |r_x{i}| = {self.rs['Xi'][i][-1]:.4e} \t# |s_x{i}| = {self.ss['Xi'][i][-1]:.4e} \t# rho_x{i} = {self.rhos['Xi'][i][-1]:.4e}")
                print(f"\t# |r_v| = {self.rs['V'][-1]:.4e} \t# |s_v| = {self.ss['V'][-1]:.4e} \t# rho_v = {self.rhos['V'][-1]:.4e}")
    
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
            for i in range(self.N):
                if self.rs['Xi'][i][-1] > self.rho_update_thr * self.ss['Xi'][i][-1]:
                    self.rhos['Xi'][i].append(self.rhos['Xi'][i][-1]*self.rho_update)
                elif self.ss['Xi'][i][-1] > self.rho_update_thr * self.rs['Xi'][i][-1]:
                    self.rhos['Xi'][i].append(self.rhos['Xi'][i][-1]/self.rho_update)
