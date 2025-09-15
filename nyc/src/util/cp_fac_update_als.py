import numpy as np
import tensorly as tl
from tensorly.tenalg import khatri_rao
from tensorly.decomposition import matrix_product
from scipy.linalg import inv
from src.util.t2m import t2m

def cp_fac_update_als(Y, U, ind_m, mu):
    """
    Update CP factors using Alternating Least Squares (ALS).

    Parameters:
    Y (ndarray): The tensor.
    U (list of ndarrays): List of factor matrices.
    ind_m (ndarray): Indices to be updated.
    mu (float): Regularization parameter.

    Returns:
    L (ndarray): Updated tensor.
    U (list of ndarrays): Updated factor matrices.
    """
    N = Y.ndim
    R = U[0].shape[1]
    sz = Y.shape
    
    if ind_m.size == 0:
        i = np.empty((0, N), dtype=int)
    else:
        # Convert linear indices to subscripts
        i = np.array(np.unravel_index(ind_m, sz)).T

    Ui = [None] * N
    Uinv = [None] * N
    
    for n in range(N):
        indices = [i for i in range(N) if i != n]
        Ui[n] = khatri_rao([U[idx] for idx in indices], 'r')
        Uinv[n] = Ui[n] @ inv((Ui[n].T @ Ui[n])+ mu * np.eye(R))

    for n in range(N):
        temp = t2m(Y,n)
        
        for i_n in range(sz[n]):
            if ind_m.size == None or i_n not in i[:, n]:
                U[n][i_n, :] = temp[i_n, :] @ Uinv[n]
            else:
                red_sz = [sz[j] for j in range(len(sz)) if j != n]
                red_ind = np.hstack(i[:,idx] for idx in range(N) if idx !=n)
                temp_ind = np.ravel_multi_index(red_ind, red_sz)
                temp_ind = np.setdiff1d(np.arange(np.prod(sz) // sz[n]), temp_ind)
                temp_Ui = Ui[n][temp_ind, :]
                U[n][i_n, :] = (temp[i_n, temp_ind] @ temp_Ui) @ inv(temp_Ui.T @ temp_Ui)
    
    L = khatri_rao(U[:-1], 'r')
    L = np.reshape(L @ U[-1].T, sz)
    
    return L, U