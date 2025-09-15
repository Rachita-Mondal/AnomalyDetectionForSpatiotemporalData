from time import perf_counter

import numpy as np
import scipy as sp
import numpy.linalg as la
from scipy.sparse import csr_array

from src.multilinear_ops.m2t import m2t
from src.multilinear_ops.t2m import t2m
from src.multilinear_ops.matricize import matricize
from src.multilinear_ops.tensorize import tensorize
from src.multilinear_ops.mode_product import mode_product
from src.proximal_ops.soft_hosvd import soft_moden
from src.proximal_ops.soft_treshold import soft_treshold
from src.proximal_ops.prox_l21 import prox_l21

class RPCA_GTV:
    """_summary_
    """

    def __init__(self, Y, B, **kwargs):
        self.Y = Y                          # Observed tensor data
        self.X = np.zeros(Y.shape)          # Low-rank component
        self.S = np.zeros(Y.shape)          # Sparse component
        self.B = B                          # Graph incidence matrix
        self.modes = kwargs.get('modes', [i+1 for i in range(len(Y.shape))])
        self.graph_modes = kwargs.get('graph_modes', [1])
        self.w_V = kwargs.get('w_V', la.eig(B@B.T))
        self.it = 0             # Iteration counter
        self.converged = False  # Convergence flag
        self.obj = [np.inf]     # Objective function values
        self.r = []             # Norm of primal residual
        self.s = []             # Norm of dual residual
        self.times = {'it':[],   # Time taken for each iteration
                      'Xi_update':[],
                      'S_update':[],
                      'Sv_update':[],
                      'S1_update':[],
                      'X_update':[],
                      }
        self.verbose = kwargs.get('verbose', 1)
        self.err_tol = kwargs.get('err_tol', 1e-6)
        self.rho_upd = kwargs.get('rho_upd', -1)
        self.rho_update_thr = kwargs.get('rho_update_thr', 100)
        self.benchmark = kwargs.get('benchmark', False)
        # Book-keeping of step size, primal and dual residuals
        self.rhos = {'rho_x':[], 'rho_s1':[], 'rho_sv':[]}
        self.rs = {'r_x':[], 'r_s1':[], 'r_sv':[]}
        self.ss = {'s_x':[], 's_s1':[], 's_sv':[]}

        self.gtv_norm = kwargs.get('gtv_norm', 'l1')
        self.s_norm = kwargs.get('s_norm', 'l1')
        self.gtv_grouping = kwargs.get('gtv_grouping', 'ungrouped')

        if self.gtv_norm not in ['l1', 'l21']:
            raise ValueError('gtv_norm must be either "l1" or l21"')
        if self.s_norm not in ['l1', 'l21']:
            raise ValueError('s_norm must be either "l1" or "l21"')
        if self.gtv_grouping not in ['incidence', 'incoming', 'outgoing', 'ungrouped']:
            raise ValueError('gtv_grouping must be either undirected or directed')


    def __call__(self, **kwargs):
        dims = self.Y.shape
        sv_dim = (self.B.shape[1], np.prod(dims)//self.B.shape[0])
        modes = kwargs.get('modes', [i+1 for i in range(len(dims))])
        psis = kwargs.get('psis', [1 for _ in modes])
        lda1 = kwargs.get('lda1', 1/np.sqrt(np.max(dims)))
        ldav = kwargs.get('ldav', 0.1/np.sqrt(np.max(dims)))
        lda2 = kwargs.get('lda2', 1)
        maxit = kwargs.get('maxit', 100)
        rho = kwargs.get('rho', 0.1)
        if self.rho_upd==-1:
            inv_term = np.linalg.inv(self.B@self.B.T + np.eye(self.B.shape[0])*(lda2+rho))

        rho_x = rho
        rho_s1 = rho
        rho_sv = rho
        self.rhos['rho_x'].append(rho_x)
        self.rhos['rho_s1'].append(rho_s1)
        self.rhos['rho_sv'].append(rho_sv)

        Xs = [np.zeros(dims) for _ in modes]      # Auxiliary variables for X
        S1 = np.zeros(dims)                       # Auxiliary variable for S1
        Sv = np.zeros(sv_dim)                     # Auxiliary variable for Sv
        Gamma_x = [np.zeros(dims) for _ in modes] # Dual variables for X
        Gamma_s1 = np.zeros(dims)                 # Dual variable for S1
        Gamma_sv = np.zeros(sv_dim)               # Dual variable for Sv
        
        while self.it < maxit and not self.converged:
            #### First block updates ####
            ## Xi updates
            tstart = perf_counter()
            Xi_upd_times = []
            sxi = []
            obj = 0
            for i, mode in enumerate(modes):
                xi_tstart = perf_counter()
                Xs[i], nuc_norm = soft_moden(self.X + Gamma_x[i]/rho_x, psis[i]/rho_x, mode)
                Xi_upd_times.append(perf_counter()-xi_tstart)
                obj += nuc_norm*psis[i]
            self.times['Xi_update'].append(Xi_upd_times)

            ## S update
            s_tstart = perf_counter()
            temp = (self.Y - self.X)*lda2 + (S1 + Gamma_s1/rho_s1)*rho_s1
            temp += tensorize( self.B@matricize(Sv + Gamma_sv/rho_sv, self.graph_modes)*rho_sv, dims, self.graph_modes)
            if self.rho_upd==-1:
                self.S = tensorize( inv_term@(matricize(temp, self.graph_modes)), dims, self.graph_modes)
            else:
                self.S = tensorize(
                    self.w_V[1]@((self.w_V[1].T@ matricize(temp, self.graph_modes))/(lda2+rho_s1+rho_sv*self.w_V[0]).reshape(-1,1)),
                    dims, self.graph_modes)
            self.times['S_update'].append(perf_counter()-s_tstart)

            #### Second Block Updates ####
            ## X update
            x_tstart = perf_counter()
            X = lda2*(self.Y - self.S)
            for i, mode in enumerate(modes):
                X += rho_x*(Xs[i] - Gamma_x[i]/rho_x)
            X = X/(lda2 + len(modes)*rho_x)
            self.ss['s_x'].append(rho_x*np.linalg.norm(X-self.X))
            self.X = X
            self.times['X_update'].append(perf_counter()-x_tstart)

            ## S1 update
            s1_tstart = perf_counter()
            temp = S1_update(self.S-Gamma_s1/rho_s1, lda1/rho_s1, self.s_norm, self.graph_modes)
            self.ss['s_s1'].append(rho_s1*np.linalg.norm(temp-self.S))
            S1 = temp
            self.times['S1_update'].append(perf_counter()-s1_tstart)

            ## Sv update
            sv_tstart = perf_counter()
            temp = S1_update(self.B.T@ matricize(self.S, self.graph_modes) - Gamma_sv/rho_sv,
                            ldav/rho_sv, self.gtv_norm, self.graph_modes)
            self.ss['s_sv'].append(rho_sv*np.linalg.norm(temp-Sv))
            Sv = temp
            self.times['Sv_update'].append(perf_counter()-sv_tstart)

            #### Dual updates ####
            # X dual update
            r = 0
            for i, mode in enumerate(modes):
                temp = X - Xs[i]
                r += np.linalg.norm(temp)**2
                Gamma_x[i] += rho_x*(temp)
            self.rs['r_x'].append(rho_x*np.sqrt(r))

            # S1 dual update
            temp = S1 - self.S
            self.rs['r_s1'].append(rho_s1*np.linalg.norm(temp))
            Gamma_s1 += rho_s1*(temp)

            # Sv dual update
            temp = Sv - self.B.T@ matricize( self.S, self.graph_modes)
            self.rs['r_sv'].append(rho_sv*np.linalg.norm(temp))
            Gamma_sv += rho_sv*(temp)

            ## Update objective function
            self.obj.append(obj + lda1*np.sum(np.abs(S1)) + ldav*np.sum(np.abs(Sv)) + 
                            0.5*lda2*np.linalg.norm(self.Y-self.X-self.S)**2)
            ## Update residuals
            self.r.append(np.sqrt((self.rs['r_x'][-1]**2 + self.rs['r_s1'][-1]**2 + self.rs['r_sv'][-1]**2)))
            self.s.append(np.sqrt(self.ss['s_sv'][-1]**2 + self.ss['s_s1'][-1]**2 + self.ss['s_x'][-1]**2))
            if self.verbose and self.it>1:
                self.report_iteration()
            # Check convergence
            if np.max((self.r[-1], self.s[-1])) < self.err_tol:
                print("Converged!")
                self.converged = True
            else: # Update rho
                if self.rho_upd !=-1:
                    rho_x, rho_s1, rho_sv = self.update_rho(rho_x, rho_s1, rho_sv)
                    self.rhos['rho_x'].append(rho_x)
                    self.rhos['rho_s1'].append(rho_s1)
                    self.rhos['rho_sv'].append(rho_sv)
            self.times['it'].append(perf_counter()-tstart)
            self.it += 1

        return self.X, self.S

    def report_iteration(self):
        print(f"It-{self.it}: \t## {self.times['it'][-1]:.2f} sec. \t obj={self.obj[-1]:.5f} \t## del_obj = {self.obj[-1]-self.obj[-2]:.5f}")
        print(f"|r|={self.r[-1]:.5f} \t ## |s|={self.s[-1]:.5f}")
        print(f"|r_x|={self.rs['r_x'][-1]:.5f} \t ## |s_x|={self.ss['s_x'][-1]:.5f} \t ## rho_x={self.rhos['rho_x'][-1]:.5f}")
        print(f"|r_s1|={self.rs['r_s1'][-1]:.5f} \t ## |s_s1|={self.ss['s_s1'][-1]:.5f} \t ## rho_s1={self.rhos['rho_s1'][-1]:.5f}")
        print(f"|r_sv|={self.rs['r_sv'][-1]:.5f} \t ## |s_sv|={self.ss['s_sv'][-1]:.5f} \t ## rho_sv={self.rhos['rho_sv'][-1]:.5f}")

    def update_rho(self, rho_x, rho_s1, rho_sv):
        """Update dynamic step size parameter based on residuals
        """
        if self.rs['r_x'][-1] > self.rho_update_thr*self.ss['s_x'][-1]:
            rho_x = rho_x*self.rho_upd
        elif self.ss['s_x'][-1] > self.rho_update_thr*self.rs['r_x'][-1]:
            rho_x = rho_x/self.rho_upd

        if self.rs['r_s1'][-1] > self.rho_update_thr*self.ss['s_s1'][-1]:
            rho_s1 = rho_s1*self.rho_upd
        elif self.ss['s_s1'][-1] > self.rho_update_thr*self.rs['r_s1'][-1]:
            rho_s1 = rho_s1/self.rho_upd

        if self.rs['r_sv'][-1] > self.rho_update_thr*self.ss['s_sv'][-1]:
            rho_sv = rho_sv*self.rho_upd
        elif self.ss['s_sv'][-1] > self.rho_update_thr*self.rs['r_sv'][-1]:
            rho_sv = rho_sv/self.rho_upd
        return rho_x, rho_s1, rho_sv

def S1_update(S, lda, s_norm, group_modes):
    """Update S1"""
    dim = S.shape
    if s_norm == 'l1':
        S = soft_treshold(S, lda)
    if s_norm == 'l21':
        S = tensorize( prox_l21(matricize(S, group_modes), lda), dim, group_modes)
    return S


class lrgtv:

    def __init__(self, Y, B, **kwargs):
        self.lr_modes = kwargs.get('lr_modes', [i+1 for i in range(len(Y.shape))])
        self.graph_modes = kwargs.get('graph_modes', [1])
        self.B = B                              # Graph incidence tensor \in R^{n//|V| x |V| x |V|}
        self.S_grouping = kwargs.get('S_grouping', 'l1')
        self.G_grouping = kwargs.get('G_grouping', 'incoming')
        if self.G_grouping not in ['incoming', 'outgoing', 'l1']:
            raise ValueError('G_grouping must be either incoming or outgoing or ungrouped')
        
        self.times = {"X_update":[],  "S1_update":[], "G_update":[],
                      "Xi_update":[], "S_update":[], "it":[]}
        self.obj = []                           # objective function
        self.rhos = {'X':[], 'S':[], 'G':[]}    # step size parameters
        self.r = []                             # norm of primal residual
        self.s = []                             # norm of dual residual
        self.rs = {'X': [], 'S':[], 'G':[]}     # norms of primal residuals
        self.ss = {'X': [], 'S':[], 'G':[]}     # norms of dual residuals
        self.Y = Y                              # observation matrix
        self.X = np.zeros(Y.shape)              # low-rank component
        self.S = np.zeros(Y.shape)              # sparse component

        # Dynamic step size parameters
        self.it = 0
        self.converged = False
        self.rho_update_thr = kwargs.get('rho_update_thr',100)
        self.rho_upd = kwargs.get('rho_upd',-1)
        self.verbose = kwargs.get('verbose',1)
        self.benchmark = kwargs.get('benchmark',True)
        self.err_tol = kwargs.get('err_tol', 1e-6)

    def __call__(self, **kwargs):
        dims = self.Y.shape
        modes = self.lr_modes
        M = len(modes)
        n_nodes = self.B.shape[1]
        n_samps = np.prod(dims)//n_nodes
        G_dims = (n_samps, n_nodes, n_nodes)
        B1 = kwargs.get('B1', t2m(self.B, 1))
        B1s = csr_array(B1)
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
        if self.rho_upd==-1:
            inv_term = csr_array(np.linalg.inv(B1@B1.T*rho_g + np.eye(B1.shape[0])*(lda2+rho_s)))

        Xs = [np.zeros(dims) for _ in modes]      # Auxiliary variables for X
        S1 = np.zeros(dims)                       # Auxiliary variable for S1
        G = np.zeros(G_dims) #csc_array((n_samps, n_nodes*n_nodes))# Auxiliary variable for G, matricized form
        Gamma_x = [np.zeros(dims) for _ in modes] # Dual variables for X
        Gamma_s1 = np.zeros(dims)                 # Dual variable for S1
        Gamma_g = np.zeros(G_dims)               # Dual variable for Sv

        while self.it < maxit and not self.converged:
            #### First block updates ####
            ## X update
            x_tstart = perf_counter()
            X = lda2*(self.Y - self.S)
            for i, mode in enumerate(modes):
                X += rho_x*Xs[i] - Gamma_x[i]
            X = X/(lda2 + M*rho_x)
            self.ss['X'].append(rho_x*np.linalg.norm(X-self.X))
            self.X = X
            self.times['X_update'].append(perf_counter()-x_tstart)

            ## S1 update
            s1_tstart = perf_counter()
            temp = S1_update(self.S-Gamma_s1/rho_s, lda1/rho_s, self.S_grouping, self.graph_modes)
            self.ss['S'].append(rho_s*np.linalg.norm(temp-self.S))
            S1 = temp
            self.times['S1_update'].append(perf_counter()-s1_tstart)

            ## G update
            g_tstart = perf_counter()
            temp = mode_product(self.B, matricize(self.S, self.graph_modes).T, 1) - Gamma_g/rho_g 
            if self.G_grouping == 'l1':
                temp = soft_treshold(temp, ldag/rho_g)
            elif self.G_grouping == 'incoming':
                temp_dim = temp.shape
                temp = m2t( prox_l21(t2m(temp, 2), ldag/rho_g), temp_dim, 2)
            elif self.G_grouping == 'outgoing':
                temp_dim = temp.shape
                temp = m2t( prox_l21(t2m(temp, 3), ldag/rho_g), temp_dim, 3)
            self.ss['G'].append(rho_g*np.linalg.norm(temp-G))
            G = temp
            self.times['G_update'].append(perf_counter()-g_tstart)

            #### Second Block Updates ####
            ## Xi updates
            tstart = perf_counter()
            Xi_upd_times = []
            obj = 0
            for i, mode in enumerate(modes):
                xi_tstart = perf_counter()
                Xs[i], nuc_norm = soft_moden(self.X + Gamma_x[i]/rho_x, psis[i]/rho_x, mode)
                Xi_upd_times.append(perf_counter()-xi_tstart)
                obj += nuc_norm*psis[i]
            self.times['Xi_update'].append(Xi_upd_times)

            ## S update
            s_tstart = perf_counter()
            K1 = lda2*(self.Y - self.X)
            K2 = rho_g*B1s.dot( t2m((G+Gamma_g/rho_g), 1).T)
            K3 = rho_s*(S1 + Gamma_s1/rho_s)
            self.S = tensorize( inv_term.dot(matricize(K1 + K3, self.graph_modes) + K2), dims, self.graph_modes)
            self.times['S_update'].append(perf_counter()-s_tstart)

            #### Dual updates ####
            # X dual update
            r = 0
            for i, mode in enumerate(modes):
                temp = X - Xs[i]
                r += np.linalg.norm(temp)**2
                Gamma_x[i] += rho_x*(temp)
            self.rs['X'].append(rho_x*np.sqrt(r))

            # S1 dual update
            r = S1 - self.S
            self.rs['S'].append(rho_s*np.linalg.norm(r))
            Gamma_s1 += rho_s*r

            # G dual update
            r = G - mode_product(self.B, matricize(self.S, self.graph_modes).T, 1)
            self.rs['G'].append(rho_g*np.linalg.norm(r))
            Gamma_g += rho_g*r

            ## Update objective function
            self.obj.append(obj + lda1*np.sum(np.abs(S1)) + ldag*np.sum(np.abs(G)) +  # Objective is not l21 norm
                            0.5*lda2*np.linalg.norm(self.Y-self.X-self.S)**2)
            ## Update residuals
            self.r.append(np.sqrt((self.rs['X'][-1]**2 + self.rs['S'][-1]**2 + self.rs['G'][-1]**2)))
            self.s.append(np.sqrt(self.ss['X'][-1]**2 + self.ss['S'][-1]**2 + self.ss['G'][-1]**2))
            if self.verbose and self.it>1:
                self.report_iteration()
            # Check convergence
            if np.max((self.r[-1], self.s[-1])) < self.err_tol:
                print("Converged!")
                self.converged = True
            else: # Update rho
                if self.rho_upd !=-1:
                    rho_x, rho_s, rho_g = self.update_rho(rho_x, rho_s, rho_g)
                    self.rhos['X'].append(rho_x)
                    self.rhos['S'].append(rho_s)
                    self.rhos['G'].append(rho_g)
            self.times['it'].append(perf_counter()-tstart)
            self.it += 1
        return self.X, self.S

    def update_rho(self, rho_x, rho_s, rho_g):
        """Update dynamic step size parameter based on residuals
        """
        if self.rs['X'][-1] > self.rho_update_thr*self.ss['X'][-1]:
            rho_x = rho_x*self.rho_upd
        elif self.ss['X'][-1] > self.rho_update_thr*self.rs['X'][-1]:
            rho_x = rho_x/self.rho_upd

        if self.rs['S'][-1] > self.rho_update_thr*self.ss['S'][-1]:
            rho_s = rho_s*self.rho_upd
        elif self.ss['S'][-1] > self.rho_update_thr*self.rs['S'][-1]:
            rho_s = rho_s/self.rho_upd

        if self.rs['G'][-1] > self.rho_update_thr*self.ss['G'][-1]:
            rho_g = rho_g*self.rho_upd
        elif self.ss['G'][-1] > self.rho_update_thr*self.rs['G'][-1]:
            rho_g = rho_g/self.rho_upd
        return rho_x, rho_s, rho_g

    def report_iteration(self):
        print(f"It-{self.it}: \t## {self.times['it'][-1]:.2f} sec. \t obj={self.obj[-1]:.5f} \t## del_obj = {self.obj[-1]-self.obj[-2]:.5f}")
        print(f"|r|={self.r[-1]:.5f} \t ## |s|={self.s[-1]:.5f}")
        print(f"|r_x|={self.rs['X'][-1]:.5f} \t ## |s_x|={self.ss['X'][-1]:.5f} \t ## rho_x={self.rhos['X'][-1]:.5f}")
        print(f"|r_s|={self.rs['S'][-1]:.5f} \t ## |s_s|={self.ss['S'][-1]:.5f} \t ## rho_s={self.rhos['S'][-1]:.5f}")
        print(f"|r_g|={self.rs['G'][-1]:.5f} \t ## |s_g|={self.ss['G'][-1]:.5f} \t ## rho_g={self.rhos['G'][-1]:.5f}")
