from typing import Any
import numpy as np
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

#from src.algos.admm_base_class import TwoBlockADMMBase
def lr_geos(Y, Ls, **kwargs):
    """Low-rank 'X', spatio-temporally smooth sparse 'S' separation from incomplete data.

    WRITE ELABORATE DESCRIPTION

    Args:
        Y (np.ndarray): Observed data/signal/tensor
        Ls (list): List of laplacian matrices of the neighbour graphs
        geo_modes (integers): Indices of the geometric modes. 
        lr_modes (integers): Indices of the low-rank modes. Defaults to all modes.
        
        rho (float): ADMM augmented lagrangian stepsize parameter
        lda_1 (float): Hyperparameter of the sparsity. Defaults to 1
        lda_2 (float): Fidelity term that regularizes deviation from original observation
        psis (np.array): Hyperparameters/weights of the nuclear norms. Defaults to 1
        phis (np.array): Hyperparameters of the graph variation norms. Defaults to 1
        
        psis (float, list of floats): 
        verbose (int): Algorithm verbisity level. Defaults to 1.
        max_it (int): Maximum number of iterations allowed for the algorithm. Defaults to 250
        rho_upd (float): Step size update coefficient of the ADMM algorithm. Defaults to 1.2
        rho_mu (float): Step size update treshold of the ADMM algorithm. Defaults to 10
        err_tol (float): Convergence criteria for the algorithm. Defaults to 1e-5
    """
    verbose = kwargs.get('verbose', 1)
    max_it = kwargs.get('max_it', 50)
    err_tol = kwargs.get('err_tol', 1e-5)
    rho_upd = kwargs.get('rho_upd', 1.2)
    rho_mu = kwargs.get('rho_mu', 10)
    rho = kwargs.get('rho', 1)

    sz = Y.shape
    lr_modes = kwargs.get('lr_modes', [i+1 for i in range(len(sz))])
    geo_modes = kwargs.get('geo_modes', [i+1 for i in range(len(sz))])
    N = len(lr_modes)
    K = len(geo_modes)
    
    
    # Hyperparameters
    lda1 = kwargs.get('lda1',1)
    lda2 = kwargs.get('lda2',1)
    psis = kwargs.get('psis', tuple([1 for _ in range(N)]))
    phis = kwargs.get('phis', tuple([1 for _ in range(N)]))
    if isinstance(psis,float):
        psis = tuple([psis]*N)
    if isinstance(phis,float):
        phis = tuple([phis]*K)

    # Initialize variables
    X_i = [np.zeros(sz) for _ in range(N)]
    W_p = [np.zeros(sz) for _ in range(K)]
    W = np.zeros(sz)
    S = np.zeros(sz)
    X = np.zeros(sz)

    # Initialize dual variables
    Lda_i = [np.zeros(sz) for _ in range(N)]        # X = X_i
    Gamma_p = [np.zeros(sz) for _ in range(N)]      # Wp = S xp Lp
    Gamma = np.zeros(sz)                            # W = S

    # Initialize S Update Inverses
    eigDecs = [ eigh(Ls[p].T@Ls[p]) for p in range(K)] # Lda_p, V_p
    Ii = [np.ones(s) for s in sz]
    gft_coef = np.zeros(Y.size)
    for i,p in enumerate(geo_modes):
        Ii[p-1] = eigDecs[i][0]
        gft_coef += rho*list_kronecker(Ii)
        Ii[p-1] = np.ones(sz[p-1])
    gft_coef += (rho+lda2)*list_kronecker(Ii)
    gft_coef = (1/gft_coef).reshape(sz)


    it = 0
    obj = [np.inf]
    r = []          # Norm of primal residual in each iteration
    s = []          # Norm of dual residual
    rhos = [rho]    # Step size
    while it<max_it:
        ### {Xi, S} Block updates
        ## Xi update
        o = 0
        for i,m in enumerate(lr_modes):
            X_temp = X_i[i].copy()
            X_i[i], o_temp = soft_moden((X+Lda_i[i]/rho), psis[i]/rho, m)
            o += psis[i]*o_temp
            # s_temp = X_i[i] - X_temp
            # s_norm += norm(s_temp)

        ## S update
        # solve for S
        B = np.zeros(sz)
        for i,p in enumerate(geo_modes):
            B += rho*m2t( Ls[i].T@ 
                         t2m(W_p[i]+Gamma_p[i]/rho, p),
                        sz, p)
        
        B += rho*(W+Gamma/rho)
        B += lda2*(Y-X)
        # Take the Graph Fourier Transform in respective modes 
        for i,p in enumerate(geo_modes): 
            B = m2t(eigDecs[i][1].T@t2m(B,p),
                    sz, p)
        # Filter in GFT domain
        B = gft_coef*B
        # Take inverse GFT 
        for i,p in enumerate(geo_modes):
            B = m2t(eigDecs[i][1]@t2m(B,p),
                    sz, p)
        S = B.copy()

        ### {X, W, Wp} Block updates
        ## X update
        s_norm = 0
        tempX = rho*sum(X_i)-sum(Lda_i)
        X_temp = (lda2*(Y-S) + tempX)/(lda2+N*rho)
        
        s_norm += norm(X-X_temp)**2
        X = X_temp.copy()
        ## W update
        W_temp = soft_treshold( S-Gamma/rho,
                          lda1/rho)
        s_norm += norm(W-W_temp)**2
        ## Wp updates
        for i,p in enumerate(geo_modes):
            W_temp = soft_treshold(m2t( Ls[i]@t2m(S,p),sz,p) -Gamma_p[i]/rho,
                            phis[i]/rho)
            s_norm += norm(W_p[i]-W_temp)**2
            W_p[i] = W_temp.copy()
        
        s.append(np.sqrt(s_norm))
        ### Dual variable updates and primal residual
        r_norm = 0
        for i in range(N):
            r_temp = X-X_i[i]
            Lda_i[i] = Lda_i[i] + rho*(r_temp)
            r_norm += norm(r_temp)**2
        
        r_temp = W-S
        Gamma += rho*r_temp
        r_norm += norm(r_temp)**2

        for i,p in enumerate(geo_modes):
            r_temp = W_p[i]- m2t(Ls[i]@t2m(S,p),sz,p)
            Gamma_p[i] += rho*r_temp
            r_norm += norm(r_temp)**2
            o += phis[i]*norm(W_p[i].ravel(),1)

        r.append(np.sqrt(r_norm))
        o += 0.5*lda2*norm(X+S-Y)**2 + lda1*norm(W.ravel(),1)
        obj.append(np.sqrt(o))
        it +=1
        # Check convergence
        eps_pri = err_tol   # Change later
        eps_dual = err_tol  # Change later
        if verbose and it>1:
            print(f"It-{it}:\t## |r|={r[-1]:.5f} \t ## |s|={s[-1]:.5f} \t ## rho={rho:.4f} obj={obj[-1]:.4f} \t ## del_obj = {obj[-1]-obj[-2]:.4f} ")
        if r[-1]<eps_pri and s[-1]<eps_dual:
            if verbose:
                print("Converged!")
            break

    results = {'X':X,
                'S':S,
                'obj':np.array(obj),
                'r':np.array(r),
                's':np.array(s),
                'it':it,
                'rho':np.array(rhos),
                }
    return results


def plot_alg(r,s,obj, rhos):
    fig, axs = plt.subplots(2,2)
    axs[0,0].plot(r)
    axs[0,0].set_xlabel("k")
    axs[0,0].set_ylabel("||r||")
    
    axs[0,1].plot(s)
    axs[0,1].set_xlabel("k")
    axs[0,1].set_ylabel("||s||")

    axs[1,0].plot(obj[1:])
    axs[1,0].set_xlabel("k")
    axs[1,0].set_ylabel("Objective")

    axs[1,1].plot(rhos)
    axs[1,1].set_xlabel("k")
    axs[1,1].set_ylabel("rho")
