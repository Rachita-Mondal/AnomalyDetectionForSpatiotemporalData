import numpy as np
import torch
from src.multilinear_ops.t2m import t2m
from src.multilinear_ops.m2t import m2t


def soft_hosvd(L, Lambda, psi, tau):
    """Soft thresholding of singular values using HOSVD.

    Uses thresholding parameter tau*psi[i] at each mode i.

    Parameters:
        L: Data tensor
        Lambda: List of dual variable tensors for each mode.
        psi: List of weights of each mode's nuclear norm.
        tau: Lagrange multiplier.

    Outputs:
        X: List of thresholded tensors for each mode.
        fn_val: Function value.
    """
    n = len(Lambda)
    X = []
    nuc_norm = []
    for i in range(n):
        ten, val = soft_moden(L-Lambda[i], tau*psi[i], i)
        X.append(ten)
        nuc_norm.append(val*psi[i])

    return X, nuc_norm


def soft_moden(T, tau, n):
    """Soft tresholding of the singular values of a tensor in the n'th mode.
    
    With the tresholding parameter tau. 
    Args:
        T (np.ndarray): Tensor matricized in the n'th mode and
        tresholded in it's n'th mode singular values.
        tau (float): Treshold
        n (int): mode index

    Returns:
        Tnew: Thresholded tensor.
    """
    sz = T.shape
    if isinstance(T, np.ndarray):
        U, S, V = np.linalg.svd(t2m(T, n), full_matrices=False)
        s = S-tau
        smask = s > 0
        S = np.diag(s[smask])
        nuc_norm = sum(s[smask])
        X = U[:, smask]@S@V[smask, :]
        return m2t(X, sz, n), nuc_norm
    elif isinstance(T, torch.Tensor):
        try:
            U, S, V = torch.linalg.svd(t2m(T, n), full_matrices=False)
        except:
            U, S, V = torch.linalg.svd(t2m(T, n), full_matrices=False, driver='gesvda')
        s = S-tau
        smask = s > 0
        S = s[smask]
        nuc_norm = sum(s[smask])
        try:
            X = torch.einsum('ik,k,kj', U[:, smask], S, V[smask, :])
        except:
            torch.backends.opt_einsum.enabled = False
            X = torch.einsum('ik,k,kj', U[:, smask], S, V[smask, :])
        return m2t(X, sz, n), nuc_norm

