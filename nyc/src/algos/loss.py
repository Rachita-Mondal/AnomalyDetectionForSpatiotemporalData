import numpy as np
from scipy.linalg import svd, toeplitz, pinv
from scipy.sparse import diags
from time import time as tic
from src.util.m_soft_threshold import m_soft_threshold
from src.util.merge_tensors import merge_tensors
from src.util.runfold import runfold
from src.util.soft_hosvd import soft_hosvd

def loss(Y, param):
    """
    Low rank + Temporally Smooth Sparse Decomposition.
    
    Parameters:
    Y : numpy.ndarray
        Input tensor.
    param : dict
        Dictionary of parameters including:
            - ind_m: Indices of modes to mask.
            - max_iter: Maximum number of iterations.
            - err_tol: Error tolerance for convergence.
            - alpha, lambda, gamma: Regularization parameters.
            - psi: Parameters for soft_HOSVD.
            - beta_1, beta_2, beta_3: Balancing parameters.
            
    Returns:
    L : numpy.ndarray
        Low-rank component.
    S : numpy.ndarray
        Sparse component.
    Nt : numpy.ndarray
        Temporal component.
    times : dict
        Dictionary containing timing information for different operations.
    """
    N = Y.ndim
    sz = Y.shape
    mask_Y = np.ones(sz, dtype=bool)
    mask_Y[tuple(param['ind_m'])] = False
    
    max_iter = param['max_iter']
    err_tol = param['err_tol']
    alpha = param['alpha']
    lambda_ = param['lambda']
    gamma = param['gamma']
    psi = param['psi']
    beta_1 = param['beta_1']
    beta_2 = param['beta_2']
    beta_3 = param['beta_3']
    
    L = [np.zeros(sz) for _ in range(N)]
    S = np.zeros(sz)
    Nt = np.zeros(sz)
    W = np.zeros(sz)
    Z = np.zeros(sz)
    D = np.diff(np.eye(sz[0]), axis=0)
    Lam1 = [np.zeros(sz) for _ in range(N)]
    Lam2 = np.zeros(sz)
    Lam3 = np.zeros(sz)
    
    times = {'L': [], 'S': [], 'N': [], 'W': [], 'Z': [], 'Dual': []}
    iter = 1
    
    while True:
        # L Update
        tstart = tic()
        temp1 = np.zeros(sz)
        temp2 = Y - S - Nt
        temp1[mask_Y] = temp2[mask_Y]
        L, _ = soft_hosvd(temp1, Lam1, psi, 1 / beta_1)
        times['L'].append(tic() - tstart)
        
        # S Update
        tstart = tic()
        temp1 = np.zeros(sz)
        for i in range(N):
            temp1 += beta_1 * (Y - L[i] - Nt + Lam1[i])
        temp1[~mask_Y] = 0
        temp2 = beta_3 * (W + Lam3)
        Sold = S
        S = m_soft_threshold((temp1 + temp2), lambda_) / (N * beta_1 + beta_3)
        times['S'].append(tic() - tstart)
        
        # N Update
        tstart = tic()
        Nt = np.zeros(sz)
        for i in range(N):
            Nt = (beta_1 / (N * beta_1 + alpha)) * (Y + Lam1[i] - L[i] - S)
        times['N'].append(tic() - tstart)
        
        # W Update
        tstart = tic()
        Dtemp = D.T @ D
        Dtemp2 = D.T
        W = pinv(beta_3 * np.eye(sz[0]) + beta_2 * Dtemp) @ (beta_3 * runfold(S - Lam3, 1) + beta_2 * Dtemp2 @ runfold(Z + Lam2,             1))
        W = np.reshape(W, sz)
        times['W'].append(tic() - tstart)
        
        # Z Update
        tstart = tic()
        Z = m_soft_threshold(merge_tensors(D, W, [1], [0]) - Lam2, gamma / beta_2)
        times['Z'].append(tic() - tstart)
        
        # Dual Updates
        tstart = tic()
        temp = 0
        for i in range(N):
            Lam1_up = Y - L[i] - S - Nt
            temp += np.linalg.norm(Lam1_up[mask_Y])**2
            Lam1[i][mask_Y] += Lam1_up[mask_Y]
        temp = np.sqrt(temp) / (np.sqrt(N) * np.linalg.norm(Y))
        Lam2 -= merge_tensors(D, W, [1], [0]) - Z
        Lam3 -= S - W
        times['Dual'].append(tic() - tstart)
        
        # Error calculation
        err = max(np.linalg.norm(S - Sold) / np.linalg.norm(Sold), temp)
        iter += 1
        
        if err <= err_tol:
            print('Converged!')
            break
        if iter >= max_iter:
            print('Max iterations reached')
            break
    
    L = np.mean(L, axis=0)
    
    return L, S, Nt, times
