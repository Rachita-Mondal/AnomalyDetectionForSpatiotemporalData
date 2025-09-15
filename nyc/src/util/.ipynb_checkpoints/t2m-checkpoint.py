import numpy as np


def t2m(X, m=1):
    """Matricisez the tensor X in the m'th mode.
    
    It is done by stacking fibers of mode m as column vectors.
    Order of the other modes follow cyclic order.
    ie ( I_m x I_(m+1). ... .I_N x I_0. ... I_(m-1) ).
    Args:
        X (np.ndarray): Tensor to be matricized
        m (int, optional): The mode whose fibers are stacked as vectors. Defaults to 1.
    Returns:
        M (np.ndarray): Matricized tensor.
    """
    n = X.shape
    if m>len(n) or m<1:
        raise ValueError(f"Invalid unfolding mode provided. m={m}, X shape:{n}")
    Xm = __roll_2_dim(X,m).ravel().reshape((n[m-1], int(np.prod(n)/n[m-1])))
    return Xm

def __roll_2_dim(X, m):
    n = X.shape; N = len(n)
    dest = np.arange(N)
    src = (np.arange(N) + (m-1))%N
    return np.moveaxis(X, src, dest)

def convert_index(n,i,k):
    """Convert the index of an element of a tensor into it's k'th mode unfolding index

    Args:
        n (tuple): Tensor dimension
        i (tuple): Tensor index
        k (int): Mode unfolding
    Returns:
        idx (tuple): Index of the element corresponding to the matricized tensor
    """
    if type(n) != tuple:
        raise TypeError("Dimension of the tensor, n is not a tuple")
    if type(i) != tuple:
        raise TypeError("index of the tensor element, i is not a tuple")
    if len(n) ==1:
        raise ValueError(f"The provided dimension n={n} is for the vector case")
    for j, i_ in enumerate(i):
        if i_>= n[j]:
            raise ValueError(f"Index i exceeds the dimension n in {j+1}'th mode")
    if k > len(n) or k<1:
        raise ValueError(f"Unfolding mode {k} is impossible for {len(n)}'th order tensor.")
    j=0
    idx = [i[k-1]]
    n_ = list(n)
    n_ = n_[k:] + n_[:k-1]
    i_ = list(i)
    i_ = i_[k:] + i_[:k-1]
    for p,i_k in enumerate(i_):
       j+= i_k*np.prod(n_[p+1:])
    idx.append(int(j))
    return tuple(idx)