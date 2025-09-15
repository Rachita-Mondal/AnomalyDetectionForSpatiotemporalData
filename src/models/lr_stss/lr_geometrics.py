from typing import Any
import numpy as np
from time import time
from numpy.linalg import norm, inv
from scipy.sparse import csr_matrix
from scipy.linalg import eigh
from sklearn.metrics import roc_auc_score
from src.util.soft_hosvd import soft_moden
from src.util.t2m import t2m
from src.util.m2t import m2t
from src.util.soft_treshold import soft_treshold
from src.util.soft_hosvd import soft_moden
from src.util.list_kronecker import list_kronecker


# from src.algos.admm_base_class import TwoBlockADMMBase
class lr_geo_model:
    """Low-rank X, Geometrically smooth sparse S, Separation model.

    Solves the ... optimization problem.

    WRITE ELABORATE DESCRIPTION.
    """
    def __init__(self, Y, Ls, geo_modes, **kwargs):
        tstart = time()
        self.verbose = kwargs.get('verbose', 1)
        # self.max_it = kwargs.get('max_it', 250)
        self.err_tol = kwargs.get('err_tol', 1e-5)
        self.rho_upd = kwargs.get('rho_upd', 1.2)
        self.rho_mu = kwargs.get('rho_mu', 10)
        self.dims = Y.shape
        self.lr_modes = kwargs.get('lr_modes', [i+1 for i in range(len(self.dims))])
        self.geo_modes = geo_modes
        self.Ls = Ls
        self.N = len(self.lr_modes)
        self.K = len(self.geo_modes)
        # Initialize Hyperparameters
        self.lda1 = None
        self.lda2 = None
        self.psis = None
        self.phis = None
        self.rho = None
        
        self.it = 0             # ADMM Iteration
        self.obj = [np.inf]     # Objective function value
        self.r = []             # Norm of primal residual in each iteration
        self.s = []             # Norm of dual residual
        self.rhos = []          # Step size
        self.converged = False

        # Initialize S Update Inverses
        self.eig_decs = kwargs.get('eig_decs', [eigh(Ls[p].T@Ls[p]) for p in range(K)]) # Lda_p, V_p

        # Initialize variables
        self.X_i = [np.zeros(self.dims) for _ in range(N)]
        self.W_p = [np.zeros(self.dims) for _ in range(K)]
        self.W = np.zeros(self.dims)
        self.S = np.zeros(self.dims)
        self.X = np.zeros(self.dims)

        # Initialize dual variables
        self.Lda_i = [np.zeros(self.dims) for _ in range(N)]           # X = X_i
        self.Gamma_p = [np.zeros(self.dims) for _ in range(N)]         # Wp = S xp Lp
        self.Gamma = np.zeros(self.dims)                               # W = S

        # Check convergence
        self.eps_pri = self.err_tol   # Change later
        self.eps_dual = self.err_tol  # Change later
        self.init_time =time()-tstart
        

    def __call__(self, **kwargs):
        max_it = kwargs.get('max_it', 50)
        # Record Hyperparameters
        self.rho = kwargs.get('rho', 0.1)
        self.lda1 = kwargs.get('lda1',1)
        self.lda2 = kwargs.get('lda2',1)
        psis = kwargs.get('psis', tuple([1 for _ in range(N)]))
        phis = kwargs.get('phis', tuple([1 for _ in range(N)]))
        if isinstance(psis,float):
            self.psis = tuple([psis]*N)
        if isinstance(phis,float):
            self.phis = tuple([phis]*K)

        # Eigenvalue decomposition of Graph Laplacians
        # to be used for the S update.
        Ii = [np.ones(n) for n in self.dims]
        gft_coef = np.zeros(self.dims)
        for i,p in enumerate(self.geo_modes):
            Ii[p-1] = self.eig_decs[i][0]
            gft_coef += self.rho*list_kronecker(Ii)
            Ii[p-1] = np.ones(self.dims[p-1])
        gft_coef += (self.rho+self.lda2)*list_kronecker(Ii)
        gft_coef = (1/gft_coef).reshape(self.dims)

        while self.it<max_it:
            ### {Xi, S} Block updates
            ## Xi update
            o = 0
            for i,m in enumerate(self.lr_modes):
                X_temp = self.X_i[i].copy()
                self.X_i[i], o_temp = soft_moden((self.X+self.Lda_i[i]/self.rho), self.psis[i]/self.rho, m)
                o += self.psis[i]*o_temp
                # s_temp = X_i[i] - X_temp
                # s_norm += norm(s_temp)

            ## S update
            # solve for S
            B = np.zeros(self.dims)
            for i,p in enumerate(self.geo_modes):
                B += self.rho*m2t( self.Ls[i].T@ 
                            t2m(self.W_p[i]+self.Gamma_p[i]/self.rho, p),
                            self.dims, p)
            
            B += self.rho*(self.W+self.Gamma/self.rho)
            B += self.lda2*(self.Y-self.X)
            # Take the Graph Fourier Transform in respective modes 
            for i,p in enumerate(self.geo_modes): 
                B = m2t(self.eig_decs[i][1].T@t2m(B,p),
                        self.dims, p)
            # Filter in GFT domain
            B = gft_coef*B
            # Take inverse GFT 
            for i,p in enumerate(self.geo_modes):
                B = m2t(self.eig_decs[i][1]@t2m(B,p),
                        self.dims, p)
            self.S = B.copy()

            ### {X, W, Wp} Block updates
            ## X update
            s_norm = 0
            tempX = self.rho*sum(self.X_i)-sum(self.Lda_i)
            X_temp = (self.lda2*(self.Y-self.S) + tempX)/(self.lda2+N*self.rho)
            
            s_norm += N**2*norm(self.X-X_temp)**2
            self.X = X_temp.copy()
            ## W update
            W_temp = soft_treshold( self.S-self.Gamma/self.rho,
                            self.lda1/self.rho)
            s_norm += norm(self.W-W_temp)**2
            ## Wp updates
            for i,p in enumerate(self.geo_modes):
                W_temp = soft_treshold(m2t( self.Ls[i]@t2m(self.S,p),self.dims,p) -self.Gamma_p[i]/self.rho,
                                self.phis[i]/self.rho)
                s_norm += norm(self.Ls[i].T@(self.W_p[i]-W_temp))**2
                self.W_p[i] = W_temp.copy()
            
            self.s.append(np.sqrt(s_norm))
            ### Dual variable updates and primal residual
            r_norm = 0
            for i in range(N):
                r_temp = self.X-self.X_i[i]
                self.Lda_i[i] = self.Lda_i[i] + self.rho*(self.r_temp)
                r_norm += norm(r_temp)**2
            
            r_temp = self.W-self.S
            self.Gamma += self.rho*r_temp
            r_norm += norm(r_temp)**2

            for i,p in enumerate(self.geo_modes):
                r_temp = self.W_p[i]- m2t(self.Ls[i]@t2m(self.S,p),self.dims,p)
                self.Gamma_p[i] += self.rho*r_temp
                r_norm += norm(r_temp)**2
                o += self.phis[i]*norm(self.W_p[i].ravel(),1)

            self.r.append(np.sqrt(r_norm))
            o += 0.5*self.lda2*norm(self.X+self.S-self.Y)**2 + self.lda1*norm(self.W.ravel(),1)
            self.obj.append(np.sqrt(o))
            self.it +=1
            
            if self.verbose and self.it>1:
                print(f"It-{self.it}:\t## |r|={self.r[-1]:.5f} \t ## |s|={self.s[-1]:.5f} \t"+
                f" ## rho={self.rho:.4f} obj={self.obj[-1]:.4f} \t ## del_obj = {self.obj[-1]-self.obj[-2]:.4f}")
            if self.r[-1]<self.eps_pri and self.s[-1]<self.eps_dual:
                if verbose:
                    print("Converged!")
                break


    def plot_algo_run(self):
        """Plot primal and dual residuals, objective function value and step size for each iteration. 
        """
        fig, axs = plt.subplots(2,2)
        axs[0,0].plot(self.r)
        axs[0,0].set_xlabel("k")
        axs[0,0].set_ylabel("||r||")
        
        axs[0,1].plot(self.s)
        axs[0,1].set_xlabel("k")
        axs[0,1].set_ylabel("||s||")

        axs[1,0].plot(self.obj[1:])
        axs[1,0].set_xlabel("k")
        axs[1,0].set_ylabel("Objective")

        axs[1,1].plot(self.rhos)
        axs[1,1].set_xlabel("k")
        axs[1,1].set_ylabel("rho")


# lr_geos(Y, Ls, geo_modes, **kwargs):
#     """Low-rank 'X', spatio-temporally smooth sparse 'S' separation from incomplete data.

#     WRITE ELABORATE DESCRIPTION

#     Args:
#         Y (np.ndarray): Observed data/signal/tensor
#         Ls (list): List of laplacian matrices of the neighbour graphs
#         geo_modes (integers): Indices of the geometric modes. 
#         lr_modes (integers): Indices of the low-rank modes. Defaults to all modes.
        
#         rho (float): ADMM augmented lagrangian stepsize parameter
#         lda_1 (float): Hyperparameter of the sparsity. Defaults to 1
#         lda_2 (float): Fidelity term that regularizes deviation from original observation
#         psis (np.array): Hyperparameters/weights of the nuclear norms. Defaults to 1
#         phis (np.array): Hyperparameters of the graph variation norms. Defaults to 1
        
#         psis (float, list of floats): 
#         verbose (int): Algorithm verbisity level. Defaults to 1.
#         max_it (int): Maximum number of iterations allowed for the algorithm. Defaults to 250
#         rho_upd (float): Step size update coefficient of the ADMM algorithm. Defaults to 1.2
#         rho_mu (float): Step size update treshold of the ADMM algorithm. Defaults to 10
#         err_tol (float): Convergence criteria for the algorithm. Defaults to 1e-5
#     """
    
    

#     results = {'X':X,
#                 'S':S,
#                 'obj':np.array(obj),
#                 'r':np.array(r),
#                 's':np.array(s),
#                 'it':it,
#                 'rho':np.array(rhos),
#                 }
#     return results


    
