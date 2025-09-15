import numpy as np
import tensorly as tl
from tensorly.tenalg import khatri_rao
from tensorly.decomposition import matrix_product
from scipy.linalg import inv
from src.util.t2m import t2m
from src.util.cp_fac_update_als import cp_fac_update_als
from src.util.m_soft_threshold import m_soft_threshold
from tensorly.decomposition import parafac
from scipy.linalg import svd

def compute_obj(Y, L, S, U, Lam, param):
    """Compute the objective value for the optimization."""
    N = Y.ndim
    lambda_val = param['lambda']
    mu = param['mu']
    
    term1 = np.sum((Y - S - L - Lam[0])**2)
    term2 = sum(mu / 2 * np.linalg.norm(U[i], 'fro')**2 for i in range(N))
    term3 = lambda_val * np.sum(np.abs(S))
    
    return term1 + term2 + term3

def cp_based(Y, param):
    """CP Based Low rank plus Sparse Decomposition."""
    N = Y.ndim
    sz = Y.shape
    mask_Y = np.ones(sz, dtype=bool)
    mask_Y[param['ind_m']] = False
    max_iter = param['max_iter']
    err_tol = param['err_tol']
    mu = param['mu']
    R = param['init_rank']
    
    U = [np.random.randn(sz[n], R) for n in range(N)]
    lambda_val = param['lambda']
    beta_1 = param['beta_1']
    S = np.zeros_like(Y)
    L = np.zeros_like(Y)
    Lam = [np.zeros_like(Y)]
    
    times = []
    obj_val = [compute_obj(Y, L, S, U, Lam, param)]
    iter = 1
    
    while True:
        # L, Fac Update
        tstart = time.time()
        L, U = cp_fac_update_als(Y - S - Lam[0], S, U, param['ind_m'], mu, R)
        times.append([time.time() - tstart])
        
        # S Update
        tstart = time.time()
        temp = Y - L - Lam[0]
        Sold = S.copy()
        S[mask_Y] = m_soft_threshold(temp[mask_Y], lambda_val)
        times[-1].append(time.time() - tstart)
        
        # Dual Updates
        tstart = time.time()
        Lam[0] = Lam[0] + Y - L - S
        times[-1].append(time.time() - tstart)
        
        # Error and objective calculations
        obj_val.append(compute_obj(Y, L, S, U, Lam, param))
        err = np.linalg.norm(S - Sold) / np.linalg.norm(Sold)
        iter += 1
        
        if err <= err_tol:
            print('Converged!')
            break
        if iter > max_iter:
            print('Max iter reached!')
            break
    
    return L, S, times