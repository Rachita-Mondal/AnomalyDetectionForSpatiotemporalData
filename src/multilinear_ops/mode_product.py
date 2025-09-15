import numpy as np

from src.multilinear_ops.t2m import t2m
from src.multilinear_ops.m2t import m2t

def mode_product(X, A, n):
    """Tensor mode-n product operation

    Args:
        X (np.ndarray): Tensor to be multiplied
        A (np.array): Factor matrix or vector
        n (int): mode (indexing starts from 1 not 0)

    Notes: If A is a vector of length k, (i.e. dim:(k,)),
    the product operation is (A^T . X_(n)), If A is a matrix the
    product operation is A . X_(n)

    Returns:
        X x_n A
    """
    if not isinstance(n,int):
        raise TypeError('n is not an integer')
    
    if n > len(X.shape):
        raise IndexError('Mode-{n} product of X is impossible with dim(X):{X.shape}')
    if len(A.shape)==1:
        if A.shape[0]!=X.shape[n-1]:
            raise IndexError('Factor vector A with dim(A):{A.shape} is incompatible'+
                             ' for mode-{n} product of X with dim(X):{X.shape}')
    elif len(A.shape)==2:
        if A.shape[1]!=X.shape[n-1]:
            raise IndexError('Factor matrix A with dim(A):{A.shape} is incompatible'+
                             ' for mode-{n} product of X with dim(X):{X.shape}'+
                             '\t({X.shape[n-1]}!={A.shape[1]})')
    else:
        raise TypeError('Factor A is not a matrix but a tensor.')
    
    dims = X.shape
    if A.shape ==1:
        k = 1
    else:
        k = A.shape[0]
    newdims = list(dims[:n-1]) +[k] + list(dims[n:])

    X_n = t2m(X,n)
    return m2t(A@X_n,newdims,n)