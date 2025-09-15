import numpy as np
from scipy.linalg import eig, eigh
from numpy.linalg import norm

from src.util.t2m import t2m
from src.util.m2t import m2t

def tensor_outlying_function(X, Sn, maxit=100, err_tol=1e-4, v=2, seed=10, return_Us=False, pre_tensor_pca=True, **kwargs):
    """Returns a samples Rayleigh projection depth for tensors.

    Args:
        X (np.ndarray): Point of interest
        Sn (list): Samples of a probability distribution with the same
        dimensions as X. Will be used to estimate covariance and mean.
        maxit (int): iteration limit for the algorithm
        err_tol (float): convergence criteria
        v (int): verbosity flag. 
        threshold (float, optional): Explainability threshold. Eigenvalues that have lower explainability
        than the threshold are discarded. Defaults to 0.01.
    """

    # if pre_tensor_pca:
    #     threshold = kwargs.get('threshold')
    #     Sn, X = pre_tensor_pca(Sn, X)

    # Initialization
    # rng = np.random.default_rng(seed)
    M = len(Sn)
    dim = X.shape
    N = len(dim)
    # As = [np.zeros((n,n)) for n in dim] # A = (x-E[X])(x-E[X])^T
    # Bs = [np.zeros((n,n)) for n in dim] # B = E[(X-E[X])(X-E[X])^T]
    Us = [np.ones((n,1)) for n in dim]#/rng.normal(0,1,(n,1)) for n in dim]
    Us = [u/norm(u) for u in Us]
    
    it=0
    O_r = []
    O_r_pass = []
    while it<maxit:
        for mode in range(N,0,-1): # For every mode

            # Transform the point of interest
            tmp_dim = list(dim)
            x = X.copy()
            for i in range(N):
                if i!=mode-1:
                    tmp_dim[i]=1
                    x = m2t(Us[i].T@t2m(x,i+1), tmp_dim, i+1)
            x = x.reshape((tmp_dim[mode-1],1))

            # Transform other samples
            sn = np.zeros((dim[mode-1],M))
            for i in range(M):
                s = Sn[i].copy()
                tmp_dim = list(dim)
                for i in range(N):
                    if i!=mode-1:
                        tmp_dim[i]=1
                        s = m2t(Us[i].T@ t2m(s,i+1), tmp_dim,i+1)
                sn[:,i] = s.ravel()#.reshape((tmp_dim[mode-1],1))
            
            e, u = vector_outlying_score(x, sn,return_v=True)
            u = u/norm(u)
            Us[mode-1]= u.reshape((dim[mode-1],1))
            O_r.append(e)
            # Us[mode-1] = U[:,i].reshape((dim[mode-1],1))
            # O_r.append(np.sqrt(e))
            if v==2:
                print(f"It-{it}, mode:{mode}\t O_R = {O_r[-1]}")
        it +=1
        O_r_pass.append(O_r[-1])
        if v==1:
            print(f"It-{it}\t O_R = {O_r_pass[-1]}")
        if it>1 and np.abs(O_r_pass[-2]-O_r_pass[-1])<err_tol:
            break

    if return_Us:
        return O_r_pass[-1],np.array(O_r),Us 
    else:
        return O_r_pass[-1],np.array(O_r)


def pre_tensor_pca(Sn, X, threshold=0.01, rank=None):
    """Apply dimensionality reduction PCA to the samples of Sn and X

    Args:
        Sn (list): List of samples (tensors)
        X (np.ndarray): Point of evaluation
        threshold (float, optional): Explainability threshold. Eigenvalues that have lower explainability
        than the threshold are discarded. Defaults to 0.01.
        rank (tuple, optional): If the rank is provided, instead of using a threshold the dimensionality
        reduction is done via the rank. In other words, the samples in Sn and X are projected to
        a subspace of R^{rank1 x rank2 x ... }. Defaults to None.
    Returns:
        Sn_projected
        X_projected
    """

    sz = X.shape
    Snc = [s.copy() for s in Sn] 
    kept_ranks = []
    for i in range(len(sz)):
        Sn_i = [t2m(s,i+1) for s in Sn] # 
        M = sum(Sn_i)/len(Sn_i)         # Find the mean
        C = sum([(s-M)@(s-M).T for s in Sn_i])/len(Sn)
        lda, u = eigh(C)
        threshold*np.abs(lda)>sum(np.abs(lda))


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

def vector_outlying_score(x, Sn, return_v=False, verbose=0):
    """Rayleigh outlying score

    Args:
        x (np.array): realization of interest
        Sn (list): set of realizations to compute empirical distribution
        return_v (bool, optional): _description_. Defaults to False.
    """
    assert x.size==x.shape[0]
    B = np.cov(Sn)
    mu = np.mean(Sn,1).reshape((len(x),1))
    x = x.reshape((len(x),1))
    A = (x-mu)@(x-mu).T
    try:
        lda, u = eigh(A,B,subset_by_index=[len(x)-1,len(x)-1])
    except:# LinAlgError:
        ldas, us = eigh(B)
        ldas[np.isclose(ldas,np.zeros(ldas.shape))] = 0
        nonzeros = ldas>0
        #B =  us[:,nonzeros]@np.diag(ldas[nonzeros])@us[:,nonzeros].T
        sn_new = us[:,nonzeros]@us[:,nonzeros].T@Sn # Project to nonzero with PCA
        B = np.cov(sn_new)
        mu = np.mean(Sn,1).reshape((len(x),1))
        x = x.reshape((len(x),1))
        x = us[:,nonzeros]@us[:,nonzeros].T@x
        A = (x-mu)@(x-mu).T
        try:
            #B = B + np.eye(B.shape[0])*1e-1
            lda, u = eigh(A,B,subset_by_index=[len(x)-1,len(x)-1])
        except:
            if verbose:
                print(eigh(B))
            try:
                B = B + np.eye(B.shape[0])*1e-9
                lda, u = eigh(A,B,subset_by_index=[len(x)-1,len(x)-1])
            except:
                print(eigh(B))
                raise RuntimeError("weird error")
    if return_v:
        return np.sqrt(lda),u
    else:
        return np.sqrt(lda)


