from time import perf_counter

import numpy as np
import torch
import torch.linalg as la

from src.multilinear_ops.m2t import m2t
from src.multilinear_ops.t2m import t2m
from src.multilinear_ops.matricize import matricize
from src.multilinear_ops.tensorize import tensorize
from src.proximal_ops.prox_l21 import prox_l21
from src.proximal_ops.prox_overlapping_grouped_l21 import prox_overlapping_grouped_l21
from src.proximal_ops.soft_hosvd import soft_moden


class tLR_GTV:
    """Graph Total Variation Regularized HoRPCA"""

    @torch.no_grad
    def __init__(self, Y, B, **kwargs):
        self.device = kwargs.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.dtype = kwargs.get('dtype', torch.double)
        self.lr_modes = kwargs.get('lr_modes', [i+1 for i in range(len(Y.shape))])
        self.graph_modes = kwargs.get('graph_modes', [1])
        if isinstance(B, torch.Tensor):
            self.B = B.to(self.device) # Full incidence matrix
        else:
            self.B = torch.tensor(B, device=self.device)
        self.BBT = B @ B.T
        
        self.S_grouping = kwargs.get('S_grouping', 'l1')
        self.G_grouping = kwargs.get('G_grouping', 'incoming')
        if self.G_grouping not in ['incoming', 'outgoing', 'l1']:
            raise ValueError('G_grouping must be either incoming or outgoing or ungrouped')
        if self.S_grouping not in ['l1', 'l21', 'overlapping_group_l21']:
            raise ValueError('S_grouping must be either l1 or l21 or overlapping_l21')
        if self.S_grouping == 'overlapping_group_l21':
            self.weights = kwargs.get('group_weights', None)
            if self.weights is None:
                raise ValueError('Weights must be provided for overlapping_group_l21')
            self.G_ind = kwargs.get('G_ind', None)
            if self.G_ind is None:
                raise ValueError('G_ind must be provided for overlapping_group_l21')
            self.G_ind_coo = kwargs.get('G_ind_coo', None)
            if self.G_ind_coo is None:
                raise ValueError('G_ind_coo must be provided for overlapping_group_l21')
            self.G_ind_T = kwargs.get('G_ind_T', None)
            if self.G_ind_T is None:
                raise ValueError('G_ind_T must be provided for overlapping_group_l21')

        # Bookkeeping
        self.times = {"X_update":[],  "S1_update":[], "G_update":[],
                      "Xi_update":[], "S_update":[], "iteration":[]}
        self.obj = []                           # objective function
        self.rhos = {'X':[], 'S':[], 'G':[]}    # step size parameters
        self.r = []                             # norm of primal residual
        self.s = []                             # norm of dual residual
        self.rs = {'X': [], 'S':[], 'G':[]}     # norms of primal residuals
        self.ss = {'X': [], 'S':[], 'G':[]}     # norms of dual residuals
        if isinstance(Y, torch.Tensor):
            self.Y = Y.to(device=self.device, dtype=self.dtype)          # observation matrix
        else:
            self.Y = torch.tensor(Y, device=self.device, dtype=self.dtype)
        # Dynamic step size parameters
        self.it = 0
        self.converged = False
        self.rho_update_thr = kwargs.get('rho_update_thr',100)
        self.rho_update = kwargs.get('rho_upd',-1)
        self.verbose = kwargs.get('verbose',1)
        self.benchmark = kwargs.get('benchmark',True)
        self.err_tol = kwargs.get('err_tol', 1e-6)

        self.X = torch.zeros(Y.shape, device=self.device, dtype=self.dtype) # low-rank component
        self.S = torch.zeros(Y.shape, device=self.device, dtype=self.dtype) # sparse component
    

    @torch.no_grad
    def __call__(self, **kwargs):
        dims = self.Y.shape
        modes = self.lr_modes
        M = len(modes)
        n_nodes = self.B.shape[0]
        n_samps = np.prod(dims)//n_nodes
        G_dims = (n_samps, n_nodes, n_nodes)
        I = torch.eye(n_nodes, device=self.device, dtype=self.dtype)
        hotstart_S1 = None
        # Hyperparameters
        psis = kwargs.get('psis', [1 for _ in modes])
        lda1 = kwargs.get('lda1', 1/np.sqrt(np.max(dims)))
        ldag = kwargs.get('ldag', 0.1/np.sqrt(np.max(dims)))
        lda2 = kwargs.get('lda2', 1)
        maxit = kwargs.get('maxit', 100)
        rho = kwargs.get('rho', 0.1)
        rho_x = rho
        rho_s = rho
        rho_g = rho
        self.rhos['X'].append(rho_x)
        self.rhos['S'].append(rho_s)
        self.rhos['G'].append(rho_g)

        # Initialize variables
        Xs = [torch.zeros(dims, device=self.device, dtype=self.dtype) for _ in modes]      # Auxiliary variables for X
        S1 = torch.zeros(dims, device=self.device, dtype=self.dtype)                       # Auxiliary variable for S1
        G = torch.zeros(G_dims, device=self.device, dtype=self.dtype) # Auxiliary variable for G, matricized form
        Gamma_x = [torch.zeros(dims, device=self.device, dtype=self.dtype) for _ in modes] # Dual variables for X
        Gamma_s1 = torch.zeros(dims, device=self.device, dtype=self.dtype)                 # Dual variable for S1
        Gamma_g = torch.zeros(G_dims, device=self.device, dtype=self.dtype)                # Dual variable for G
        X_temp = torch.zeros(dims, device=self.device, dtype=self.dtype)                   # Temporary variable for X
        S1_temp = torch.zeros(dims, device=self.device, dtype=self.dtype)                  # Temporary variable for S
        G_temp = torch.zeros(G_dims, device=self.device, dtype=self.dtype)                 # Temporary variable for G
        while self.it < maxit and not self.converged:
            ## First Block {X,S1,G} updates
            
            # Update X ----------------------------------------------
            x_start = perf_counter()
            X_temp = lda2*(self.Y - self.S)
            for i, mode in enumerate(modes):
                X_temp += rho_x*Xs[i] - Gamma_x[i]
            X_temp = X_temp/(lda2 + M*rho_x)
            self.times["X_update"].append(perf_counter() - x_start)
            self.ss['X'].append(rho_x*la.norm(X_temp- self.X))
            self.X = X_temp

            # Update S1 ----------------------------------------------
            s1_start = perf_counter()
            S1_temp = self.S - Gamma_s1/rho_s
            if self.S_grouping == 'l1':
                S1_temp = torch.nn.functional.softshrink(S1_temp, lda1/rho_s)
            elif self.S_grouping == 'l21':
                S1_temp = prox_l21(S1_temp, lda1/rho_s, tuple([g-1 for g in self.graph_modes]))
            elif self.S_grouping == 'overlapping_group_l21':
                S1_temp, gap, _, _, hotstart_S1 = prox_overlapping_grouped_l21(S1_temp, 
                                        lda1[0]/rho_s, lda1[1]/rho_s, hotstart = hotstart_S1,
                                        self.weights, G_ind_coo= self.G_ind_coo, G_ind_T=self.G_ind_T,
                                        max_iter=100, err_tol=1e-4)

            self.ss['S'].append(rho_s*la.norm(S1_temp - S1))
            S1 = S1_temp
            self.times["S1_update"].append(perf_counter() - s1_start)

            # Update G ----------------------------------------------
            g_start = perf_counter()
            G_temp = matricize(self.S, self.graph_modes).T @ self.B
            G_temp = tensorize(G_temp, G_dims, [1]) - Gamma_g/rho_g
            if self.G_grouping == 'l1':
                G_temp = torch.nn.functional.softshrink(G_temp, ldag/rho_g)
            elif self.G_grouping == 'incoming':
                G_temp = prox_l21(G_temp, ldag/rho_g, tuple([1]))
            elif self.G_grouping == 'outgoing':
                G_temp = prox_l21(G_temp, ldag/rho_g, tuple([2]))
            self.ss['G'].append(rho_g*la.norm(G_temp - G))
            G = G_temp
            self.times["G_update"].append(perf_counter() - g_start)

            ## Second Block {Xi,S} updates
            # Update Xi ----------------------------------------------
            Xi_upd_times = []
            obj = 0
            for i, mode in enumerate(modes):
                xi_start = perf_counter()
                Xs[i], nuc_norm = soft_moden(self.X+Gamma_x[i]/rho_x, psis[i]/rho_x, mode)
                Xi_upd_times.append(perf_counter() - xi_start)
                obj += nuc_norm
            self.times["Xi_update"].append(Xi_upd_times)

            # Update S ----------------------------------------------
            s_start = perf_counter()
            K1 = lda2*(self.Y - self.X)
            K2 = rho_g* self.B@ t2m((G+Gamma_g/rho_g), 1).T
            K3 = rho_s*(S1 + Gamma_s1/rho_s)
            self.S = tensorize( torch.linalg.solve(
                                                    (rho_s+lda2)*I + rho_g*self.BBT ,
                                                    matricize(K1 + K3, self.graph_modes) + K2),
                                dims, self.graph_modes)
            self.times['S_update'].append(perf_counter()-s_start)

            ## Dual updates
            # X dual update
            r = 0
            for i, mode in enumerate(modes):
                temp = self.X - Xs[i]
                r += torch.linalg.norm(temp)**2
                Gamma_x[i] += rho_x*(temp)
            self.rs['X'].append(rho_x*torch.sqrt(r))
            # S1 dual update
            r = S1 - self.S
            self.rs['S'].append(rho_s*torch.linalg.norm(r))
            Gamma_s1 += rho_s*r
            # G dual update
            r = G - tensorize(matricize(self.S, self.graph_modes).T @ self.B, G_dims, [1])
            self.rs['G'].append(rho_g*torch.linalg.norm(r))
            Gamma_g += rho_g*r

            ## Update objective function
            self.obj.append(obj + lda1*torch.norm(S1, p=1, dim=None) + ldag*torch.norm(G, p=1, dim=None) +  # Objective is not l21 norm
                            0.5*lda2*torch.norm(self.Y-self.X-self.S)**2)
            ## Update residuals
            self.r.append(torch.sqrt((self.rs['X'][-1]**2 + self.rs['S'][-1]**2 + self.rs['G'][-1]**2)))
            self.s.append(torch.sqrt(self.ss['X'][-1]**2 + self.ss['S'][-1]**2 + self.ss['G'][-1]**2))
            if self.verbose and self.it>1:
                self.report_iteration()
            # Check for convergence
            if torch.maximum(self.r[-1], self.s[-1]) < self.err_tol:
                if self.verbose:
                    print("Converged!")
                self.converged = True
            else: # Update rho
                if self.rho_update !=-1:
                    rho_x, rho_s, rho_g = self.update_rho(rho_x, rho_s, rho_g)
                    self.rhos['X'].append(rho_x)
                    self.rhos['S'].append(rho_s)
                    self.rhos['G'].append(rho_g)
            self.times['iteration'].append(perf_counter()-x_start)
            self.it += 1
        return self.X, self.S


    def update_rho(self, rho_x, rho_s, rho_g):
        """Update dynamic step size parameter based on residuals
        """
        if rho_x > 1e-6 and rho_x < 1e6:
            if self.rs['X'][-1] > self.rho_update_thr*self.ss['X'][-1]:
                rho_x = rho_x*self.rho_update
            elif self.ss['X'][-1] > self.rho_update_thr*self.rs['X'][-1]:
                rho_x = rho_x/self.rho_update
        if rho_s > 1e-6 and rho_s < 1e6:
            if self.rs['S'][-1] > self.rho_update_thr*self.ss['S'][-1]:
                rho_s = rho_s*self.rho_update
            elif self.ss['S'][-1] > self.rho_update_thr*self.rs['S'][-1]:
                rho_s = rho_s/self.rho_update
        if rho_g > 1e-6 and rho_g < 1e6:
            if self.rs['G'][-1] > self.rho_update_thr*self.ss['G'][-1]:
                rho_g = rho_g*self.rho_update
            elif self.ss['G'][-1] > self.rho_update_thr*self.rs['G'][-1]:
                rho_g = rho_g/self.rho_update
        return rho_x, rho_s, rho_g
    

    def report_iteration(self):
        print(f"It-{self.it}: \t## {self.times['iteration'][-1]:.2f} sec. \t obj={self.obj[-1]:.5f} \t## del_obj = {self.obj[-1]-self.obj[-2]:.5f}")
        print(f"|r|={self.r[-1]:.5f} \t ## |s|={self.s[-1]:.5f}")
        print(f"|r_x|={self.rs['X'][-1]:.5f} \t ## |s_x|={self.ss['X'][-1]:.5f} \t ## rho_x={self.rhos['X'][-1]:.5f}")
        print(f"|r_s|={self.rs['S'][-1]:.5f} \t ## |s_s|={self.ss['S'][-1]:.5f} \t ## rho_s={self.rhos['S'][-1]:.5f}")
        print(f"|r_g|={self.rs['G'][-1]:.5f} \t ## |s_g|={self.ss['G'][-1]:.5f} \t ## rho_g={self.rhos['G'][-1]:.5f}")