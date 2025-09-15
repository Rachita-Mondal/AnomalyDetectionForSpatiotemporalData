"""Low-rank, Latent Overlapping Grouped Sparse Decomposition Model.
"""

from time import perf_counter
from math import prod

import torch
import torch.linalg as la

from src.multilinear_ops.matricize import matricize
from src.multilinear_ops.tensorize import tensorize
from src.proximal_ops.prox_overlapping_grouped_l21 import prox_overlapping_grouped_l21
from src.proximal_ops.soft_hosvd import soft_moden
from torch.functional import F


class LR_LOGS_ST_TF:
    
    def __init__(self, Y, G_ind, group_weights, B1T, B2T, BtT, **kwargs):
        """Initialize the LR-LOGS-ST-TF model.
        
        Args:
            Y (torch.Tensor): Input tensor
            G_ind (list): List of group indicator matrices.
            group_weights (list): List of group weights.
            B1 (torch.Tensor): Oriented incidence matrix for spatial dimensions.
            B2 (torch.Tensor): Oriented incidence matrix for temporal dimension.
            Bt (torch.Tensor): Oriented incidence matrix for the temporal dimension.
        """
        self.device = kwargs.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.dtype = kwargs.get('dtype', torch.double)
        if isinstance(Y, torch.Tensor):
            self.Y = Y.to(device=self.device, dtype=self.dtype)          # observation matrix
        else:
            self.Y = torch.tensor(Y, device=self.device, dtype=self.dtype)
        

        self.lr_modes = kwargs.get('lr_modes', [i+1 for i in range(len(Y.shape))])
        self.graph_modes = kwargs.get('graph_modes', [1])
        self.temporal_mode = kwargs.get('temporal_mode', 2)
        self.N = len(self.lr_modes)

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
        self.B1T = B1T.to(device=self.device, dtype=self.dtype)          # oriented incidence matrix for spatial dimensions
        self.B2T = B2T.to(device=self.device, dtype=self.dtype)          # oriented incidence matrix for temporal dimension
        self.BtT = BtT.to(device=self.device, dtype=self.dtype)          # oriented incidence matrix for the temporal dimension
        self.B1B1T = self.B1T.t() @ self.B1T
        self.B2B2T = self.B2T.t() @ self.B2T
        self.BtBtT = self.BtT.t() @ self.BtT
        self.Il = torch.eye(self.nov, device=self.device, dtype=self.dtype)
        self.It = torch.eye(self.Y.shape[self.temporal_mode-1], device=self.device, dtype=self.dtype)

        # Bookkeeping
        self.it = 0
        self.verbose = kwargs.get('verbose',3)
        self.benchmark = kwargs.get('benchmark',True)
        self.err_tol = kwargs.get('err_tol', 1e-6)
        self.converged = False
        self.times = {"X":[], "V":[], "S0":[], "S1":[], "S2":[], "St":[],  # First block variables
                      "Xi":[[] for _ in range(self.N)], "Zbar":[], "S":[],# Second block variables
                      "W1":[], "W2":[], "Wt":[], 
                      "iteration":[]}
        self.obj = []                           # objective function
        self.rhos = {'Xi': [[] for _ in range(self.N)], # step size parameters
                     0:[], 1:[], 2:[], 't':[]}    
        self.r = []                             # norm of primal residual
        self.s = []                             # norm of dual residual
        self.rs = {'Xi': [[] for _ in range(self.N)], # step size parameters
                     '0a':[], '0b':[], '0c':[], '1a':[], '1b':[], '2a':[], '2b':[],
                     'ta':[], 'tb':[]}
        self.ss = {'Xi': [[] for _ in range(self.N)], # step size parameters
                     '0a':[], '0b':[], '0c':[], '1a':[], '1b':[], '2a':[], '2b':[], 
                     'ta':[], 'tb':[]}
        self.verbose = kwargs.get('verbose', 1)
        self.report_freq = kwargs.get('report_freq', 1)
        self.metric_tracker = kwargs.get('metric_tracker', None)
        self.timeit = kwargs.get('timeit', False)

        # Dynamic step size parameters    
        self.rho_update_thr = kwargs.get('rho_update_thr',100)
        self.rho_update = kwargs.get('rho_upd',1.1)
        
        self.X = torch.zeros(Y.shape, device=self.device, dtype=self.dtype) # low-rank component
        self.S = torch.zeros(Y.shape, device=self.device, dtype=self.dtype) # sparse component


    def __call__(self, max_iter=1000, rho=1.0, **kwargs):
        """Run the optimization algorithm.
        
        Args:
            max_iter (int): Maximum number of iterations.
            rho (float): Initial step size parameter.
        """
        # Hyperparameters
        psis = kwargs.get('psis', [1.0 for _ in self.lr_modes])
        lda_f = kwargs.get('lda_f', 100)
        lda_0 = kwargs.get('lda_0', 0.01)
        lda_1 = kwargs.get('lda_1', 0.01)
        lda_2 = kwargs.get('lda_2', 0.01)
        lda_t = kwargs.get('lda_t', 0.01)
        weights = self.group_weights.to_dense().reshape((1, self.nog, 1))
        # Step-size parameters
        for i in range(self.N):
            self.rhos['Xi'][i].append(rho)
            self.rhos[0].append(rho)
            self.rhos[1].append(rho)
            self.rhos[2].append(rho)
            self.rhos['t'].append(rho)
        
        # Initialize variables
        Xi = [torch.zeros(self.Y.shape, device=self.device, dtype=self.dtype) for _ in range(self.N)]
        S0 = torch.zeros_like(self.Y)
        S1 = torch.zeros_like(self.Y)
        S2 = torch.zeros_like(self.Y)
        St = torch.zeros_like(self.Y)
        W1 = self.B1T @ matricize(S1, self.graph_modes)
        W2 = self.B2T @ matricize(S2, self.graph_modes)
        Wt = self.BtT @ matricize(St, [self.temporal_mode])
        Zbar = matricize(torch.zeros_like(self.Y),  # Batch x 1 x |Vertices|: Dense tensor 
                        self.graph_modes).t().reshape((self.batch_dim, 1, self.nov))
        V = (self.expander * Zbar).coalesce()# Batch x |Groups| x |Vertices|: Sparse tensor
        
        # Dual variables
        Gamma_xi = [torch.zeros_like(self.Y) for _ in range(self.N)]
        Gamma_0 = torch.zeros_like(self.Y)
        Gamma_01 = torch.zeros_like(self.Y)
        Gamma_1 = torch.zeros_like(self.Y)
        Gamma_11 = torch.zeros_like(W1)
        Gamma_2 = torch.zeros_like(self.Y)
        Gamma_21 = torch.zeros_like(W2)
        Gamma_t = torch.zeros_like(self.Y)
        Gamma_t1 = torch.zeros_like(Wt)
        Ubar = torch.zeros_like(Zbar)

        while self.it < max_iter and not self.converged:
            it_start = perf_counter()
            rho_x = [self.rhos['Xi'][i][-1] for i in range(self.N)]
            rho_0 = self.rhos[0][-1]
            rho_1 = self.rhos[1][-1]
            rho_2 = self.rhos[2][-1]
            rho_t = self.rhos['t'][-1]

            # First Block updates
            # # Update X
            X_temp = sum([rho_x[i]*(Xi[i]- Gamma_xi[i]/rho_x[i]) for i in range(self.N)])
            X_temp += lda_f*(self.Y-self.S)
            self.X = X_temp/(sum(rho_x)+lda_f)
            self.times['X'].append(perf_counter()-it_start)

            # # Update S0
            s_start = perf_counter()
            S0 = (self.S - Gamma_0/rho_0 +\
                self.nog*tensorize(Zbar.squeeze(1).t(), self.Y.shape, self.graph_modes) + Gamma_01/rho_0)/(self.nog + lda_0/rho_0) 
                # )/2
            self.times['S0'].append(perf_counter()-s_start)

            # # Update S1
            s_start = perf_counter()
            S1 = tensorize(
                    la.solve(self.B1B1T + self.Il, 
                            matricize(self.S - Gamma_1/rho_1, self.graph_modes) +\
                            self.B1T.t()@(W1 - Gamma_11/rho_1)
                            ),
                        self.Y.shape, self.graph_modes
                        )
            self.times['S1'].append(perf_counter()-s_start)

            # # Update S2
            s_start = perf_counter()
            S2 = tensorize(
                    la.solve(self.B2B2T + self.Il, 
                        matricize(self.S - Gamma_2/rho_2, self.graph_modes) +\
                        self.B2T.t()@(W2 - Gamma_21/rho_2)
                        ),
                    self.Y.shape, self.graph_modes
                    )
            self.times['S2'].append(perf_counter()-s_start)

            # # Update St
            s_start = perf_counter()
            St = tensorize(
                    la.solve(self.BtBtT + self.It,
                        matricize(self.S - Gamma_t/rho_t, [self.temporal_mode]) +\
                        self.BtT.t()@(Wt - Gamma_t1/rho_t)
                        ),self.Y.shape, [self.temporal_mode]
                        )
                        
            self.times['St'].append(perf_counter()-s_start)

            # # Update V
            v_start = perf_counter()
            V_old = V.clone().detach()
            Vbar = (torch.sum(V, dim=1, keepdim=True)/self.nog).to_dense()
            Vtemp = self.expander*( Zbar - Vbar - Ubar) + V
            norms = Vtemp.pow(2).sum(dim=2,keepdim=True).sqrt().to_dense()
            scaling_factor = torch.where(norms > weights*lda_0, 1-weights*lda_0/(norms*rho_0), 0)
            V = Vtemp*scaling_factor
            self.times['V'].append(perf_counter()-v_start)

            # Second Block updates
            # # Update Xi
            for i in range(self.N):
                xi_start = perf_counter()
                Xi_old = Xi[i].clone().detach()
                Xi[i], nuc_norm = soft_moden(self.X + Gamma_xi[i]/rho_x[i], psis[i]/rho_x[i], self.lr_modes[i] )
                self.ss['Xi'][i].append(rho_x[i]*la.norm(Xi[i]-Xi_old))
                self.times['Xi'][i].append(perf_counter()-xi_start)

            # # Update S
            s_start = perf_counter()
            S_old = self.S.clone().detach()
            self.S = (rho_0*(S0 + Gamma_0/rho_0) +\
                    rho_1*(S1 + Gamma_1/rho_1) +\
                    rho_2*(S2 + Gamma_2/rho_2) +\
                    rho_t*(St + Gamma_t/rho_t) +\
                    lda_f*(self.Y - self.X))/(rho_0 + rho_1 + rho_2 + rho_t + lda_f)
            s_s = la.norm(self.S-S_old)
            self.times['S'].append(perf_counter()-s_start)

            # # Update Zbar
            zbar_start = perf_counter()
            Zbar_old = Zbar.clone().detach()
            Zbar = matricize((S0 + Gamma_0/rho_0), self.graph_modes).t().unsqueeze(1) + \
                    (torch.sum(V, dim=1, keepdim=True).to_dense()/self.nog + Ubar)
            Zbar /= (self.nog+1)
            Vbar_old = Vbar.clone().detach()
            Vbar = (torch.sum(V, dim=1, keepdim=True)/self.nog).to_dense()
            s_zbar = la.norm(Zbar-Zbar_old)*self.nog
            s_v = torch.sqrt(
                torch.sum(
                    ((V - V_old) + self.expander*(Vbar - Vbar_old + Zbar - Zbar_old)).coalesce().values().pow(2)
                    )
            )
            self.ss['0a'].append(rho_0*s_s)
            self.ss['0b'].append(rho_0*s_zbar)
            self.ss['0c'].append(rho_0*s_v)
            self.times['Zbar'].append(zbar_start - perf_counter())
            
            # # Update W1
            w1_start = perf_counter()
            W1_old = W1.clone().detach()
            W1 = F.softshrink(self.B1T@ matricize(S1, self.graph_modes) + Gamma_11/rho_1, lda_1/rho_1)
            # self.ss[1].append(rho_1*torch.sqrt(la.norm(self.B1T.t()@(W1-W1_old))**2 + s_s**2))
            self.ss['1a'].append(rho_1*s_s)
            self.ss['1b'].append(rho_1*la.norm(self.B1T.t()@(W1-W1_old)))
            self.times['W1'].append(w1_start - perf_counter())
            
            # # Update W2
            w2_start = perf_counter()
            W2_old = W2.clone().detach()
            W2 = F.softshrink(self.B2T@ matricize(S2, self.graph_modes) + Gamma_21/rho_2, lda_2/rho_2)
            self.ss['2a'].append(rho_2*s_s)
            self.ss['2b'].append(rho_2*la.norm(self.B2T.t()@(W2-W2_old)))
            self.times['W2'].append(w2_start - perf_counter())

            # # Update Wt
            wt_start = perf_counter()
            Wt_old = Wt.clone().detach()
            Wt = F.softshrink(self.BtT@ matricize(St, [self.temporal_mode]) + Gamma_t1/rho_t, lda_t/rho_t)
            self.ss['ta'].append(rho_t*s_s)
            self.ss['tb'].append(rho_t*la.norm(self.BtT.t()@(Wt-Wt_old)))
            self.times['Wt'].append(wt_start - perf_counter())

            # Update Dual Variables
            for i in range(self.N):
                r = self.X - Xi[i]
                Gamma_xi[i] += rho_x[i]*r
                self.rs['Xi'][i].append(la.norm(r))
            
            r = S0-self.S
            Gamma_0 = Gamma_0 + rho_0*r
            self.rs['0a'].append(la.norm(r))
            r = S0 - tensorize(self.nog*Zbar.squeeze(1).t(), self.Y.shape, self.graph_modes)
            Gamma_01 = Gamma_01 + rho_0*r
            self.rs['0b'].append(la.norm(r))
            r = Vbar-Zbar
            Ubar = Ubar + r
            # r_temp += (self.nog*la.norm(r))**2
            self.rs['0c'].append(self.nog*la.norm(r))

            r = S1-self.S
            Gamma_1 = Gamma_1 + rho_1*r
            self.rs['1a'].append(la.norm(r))

            r = self.B1T@matricize(S1, self.graph_modes)-W1
            Gamma_11 = Gamma_11 + rho_1*r
            self.rs['1b'].append(la.norm(r))
            
            r = S2-self.S
            Gamma_2 = Gamma_2 + rho_2*r
            self.rs['2a'].append( la.norm(r))
            r = self.B2T@matricize(S2, self.graph_modes)-W2
            Gamma_21 = Gamma_21 + rho_2*r
            self.rs['2b'].append(la.norm(r))

            r = St-self.S
            Gamma_t = Gamma_t + rho_t*r
            self.rs['ta'].append(la.norm(r))
            r = self.BtT@matricize(St, [self.temporal_mode])-Wt
            Gamma_t1 = Gamma_t1 + rho_t*r
            self.rs['tb'].append(la.norm(r))

            self.r.append(
                torch.sqrt(
                    sum([rxi[-1]**2 for rxi in self.rs['Xi']])+\
                    self.rs['0a'][-1]**2 + self.rs['0b'][-1]**2 + self.rs['0c'][-1]**2 +\
                    self.rs['1a'][-1]**2 + self.rs['1b'][-1]**2 +\
                    self.rs['2a'][-1]**2 + self.rs['2b'][-1]**2 +\
                    self.rs['ta'][-1]**2 + self.rs['tb'][-1]**2
                )
            )
            self.s.append(
                torch.sqrt(
                    sum([rxi[-1]**2 for rxi in self.ss['Xi']]) +\
                    self.ss['0a'][-1]**2 + self.ss['0b'][-1]**2 + self.ss['0c'][-1]**2 +\
                    self.ss['1a'][-1]**2 + self.ss['1b'][-1]**2  +\
                    self.ss['2a'][-1]**2 + self.ss['2b'][-1]**2  +\
                    self.ss['ta'][-1]**2 + self.ss['tb'][-1]**2 
                )
            )
            
            self._report_iteration()
            self._check_convergence()
            self._call_tracker()
            if not self.converged:
                self._update_step_size()
                self.it += 1
        return self.X, self.S


    def _report_iteration(self):
        if self.verbose > 0 and self.it % self.report_freq ==0:
            print(f"It-{self.it} \t# |r| = {self.r[-1]:.4e} \t|s| = {self.s[-1]:.4e}")
            if self.verbose > 1:
                for i in range(self.N):
                    print(f"\t# |r_x{i}| = {self.rs['Xi'][i][-1]:.4e} \t# |s_x{i}| = {self.ss['Xi'][i][-1]:.4e} \t# rho_x{i} = {self.rhos['Xi'][i][-1]:.4e}")
                print(f"\t# |r_0a| = {self.rs['0a'][-1]:.4e} \t# |s_0a| = {self.ss['0a'][-1]:.4e} \t# rho_0 = {self.rhos[0][-1]:.4e}")
                print(f"\t# |r_0b| = {self.rs['0b'][-1]:.4e} \t# |s_0b| = {self.ss['0b'][-1]:.4e} \t#")
                print(f"\t# |r_0c| = {self.rs['0c'][-1]:.4e} \t# |s_0c| = {self.ss['0c'][-1]:.4e} \t#")
                print(f"\t# |r_1a| = {self.rs['1a'][-1]:.4e} \t# |s_1a| = {self.ss['1a'][-1]:.4e} \t# rho_1 = {self.rhos[1][-1]:.4e}")
                print(f"\t# |r_1b| = {self.rs['1b'][-1]:.4e} \t# |s_1b| = {self.ss['1b'][-1]:.4e} \t#")
                print(f"\t# |r_2a| = {self.rs['2a'][-1]:.4e} \t# |s_2a| = {self.ss['2a'][-1]:.4e} \t# rho_2 = {self.rhos[2][-1]:.4e}")
                print(f"\t# |r_2b| = {self.rs['2b'][-1]:.4e} \t# |s_2b| = {self.ss['2b'][-1]:.4e} \t#")
                print(f"\t# |r_ta| = {self.rs['ta'][-1]:.4e} \t# |s_ta| = {self.ss['ta'][-1]:.4e} \t# rho_t = {self.rhos['t'][-1]:.4e}")
                print(f"\t# |r_tb| = {self.rs['tb'][-1]:.4e} \t# |s_tb| = {self.ss['tb'][-1]:.4e} \t#")
    
    def _check_convergence(self):
        if self.s[-1] < self.err_tol and self.r[-1] < self.err_tol:
            self.converged = True
            if self.verbose > 1:
                print(f"Converged in {self.it} iterations.")
        return self.converged
    
    def _update_step_size(self):
        """Update the step size parameter.
        """
        if self.rho_update <1:
            raise ValueError("Step size growth factor must be larger than 1")
        if self.rho_update != 1.0:
            for i in range(self.N):
                if self.rs['Xi'][i][-1] > self.rho_update_thr * self.ss['Xi'][i][-1]:
                    self.rhos['Xi'][i].append(self.rhos['Xi'][i][-1]*self.rho_update)
                elif self.ss['Xi'][i][-1] > self.rho_update_thr * self.rs['Xi'][i][-1]:
                    self.rhos['Xi'][i].append(self.rhos['Xi'][i][-1]/self.rho_update)


    def _call_tracker(self):
        """Tracks metrics of the state variables of the algorithm using a MetricTracker object.
        
        Please see the Tracker class for more information.
        """
        if self.metric_tracker is not None:
            if self.it % self.metric_tracker.tracker_frequency == 0 or self.converged:
                self.metric_tracker.track(self)