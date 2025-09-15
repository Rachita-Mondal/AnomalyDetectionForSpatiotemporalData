import torch

from numpy.linalg import svd
from scipy.sparse.linalg import svds
from src.multilinear_ops.t2m import t2m
from src.multilinear_ops.m2t import m2t

def mode_svd(X, n, rank=None):
    """SVD of the n'th mode matricization of X.

    Args:
        X (np.ndarray): Tensor to be decomposed
        n (int): Mode index
        rank (int): Truncated SVD rank

    Returns:
        U (np.ndarray): Left singular vectors
        S (np.ndarray): Singular values
        Vh (np.ndarray): Hermitian of the right singular vectors
    """
    if rank is None:
        rank = X.shape[n-1]
    if rank == X.shape[n-1]:
        if isinstance(X, torch.Tensor):
            U, S, Vh = torch.linalg.svd(t2m(X, n), full_matrices=False)
        else:
            U, S, Vh = svd(t2m(X, n), full_matrices=False)
    else:
        if isinstance(X, torch.Tensor):
            U, S, Vh = torch.linalg.svd(t2m(X, n))
            U = U[:, :rank]
            S = S[:rank]
            Vh = Vh[:rank, :]
        else:
            U, S, Vh = svds(t2m(X, n), rank, which='LM')
            U = U[:,::-1]
            S = S[::-1]
            Vh = Vh[::-1,:]
    return U, S, Vh