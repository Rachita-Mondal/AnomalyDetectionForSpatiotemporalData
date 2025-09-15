import numpy as np
import torch

def m2t(Xm, dims, m=1):
    """Tensorizes the matrix obtained by t2m to its original state.

    Args:
        Xm (np.ndarray): Matrix
        dims (tuple): original dimensions of the tensor
        m (int): The mode for which the matricization was originally made with t2m. Defaults to 1.
    Returns:
        X (np.ndarray): Tensor
    """
    N = len(dims); 
    if m > N or m <1:
        raise ValueError(f"Invalid tensorization order m={m}, N={N}")
    
    old_dest = (np.arange(N) + (m-1))%N
    dims2 = tuple([dims[i] for i in old_dest])
    X = Xm.ravel().reshape(dims2)
    X = __unroll_from_dim(X, m)
    return X

def __unroll_from_dim(X, m):
    n = X.shape
    N = len(n)
    dest = (np.arange(N) + (m-1))%N
    src = np.arange(N)
    if isinstance(X, np.ndarray):
        return np.moveaxis(X, src, dest)
    elif isinstance(X, torch.Tensor):
        return torch.moveaxis(X, tuple(src), tuple(dest))
