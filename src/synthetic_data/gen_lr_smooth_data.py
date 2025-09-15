import numpy as np
import networkx as nx
import numpy as np
from numpy.linalg import inv, norm
from src.util.graph import *
from src.util.generate_lr_data import generate_low_rank_data
from scipy.io import savemat

def gen_lr_smooth_data(dim, rank, PG, filter="Heat", alpha=10):
    """Generate a approximately low-rank data that is smooth on a given product graph.
     
    By first generating a low tucker rank tensor with rank 'rank' with the size given by 'dim'.
    This random low-rank tensor is then filtered with respect to given Product Graph (PG) with
    a specified filter.

    Args:
        dim (tuple or list of integers): Dimensions of the tensor
        rank (tuple or list of integers): Rank of the generated tensor
        PG (ProductGraph): ProductGraph object.
        filter (str, optional): Possible filters are 'Heat', 'Tikhonov', 'Gaussian'. Defaults to "Heat".
        alpha (int, optional): Smoothness parameter. Defaults to 10.

    Returns:
        X: Low-rank tensor that is smooth with respect to the product graph PG.
    """
    X = generate_low_rank_data(dim,rank)
    N = len(dim)
    lda = PG.PG.lda
    lda /=np.max(lda)       # The laplacian eigenvalues are normalized.
    h = np.zeros(lda.size)
    V = PG.PG.V
    if filter == "Gaussian":
        h = np.ones(lda.size)
        h[lda > 1e-8] = 1/np.sqrt(alpha*lda[lda>1e-8])
    elif filter == "Tikhonov":
        h = 1/(1+alpha*lda)
    elif filter == "Heat":
        h = np.exp(-alpha*lda)
    else:
        TypeError("Filter type is not recognized. filter given:"+filter) 
    h = h.reshape((h.size,1))
    x = X.reshape([ni for ni in dim]+[1])
    x = t2m(x,len(dim)+1).T # Vectorize the low-rank tensor
    gft_x = V.T@x   # Take the GFT of the low rank signal
    x = (V @ (gft_x*h)).T # Filtered (Smoothed x) 
    X = m2t(x, [ni for ni in dim]+[1],N+1)
    X = np.squeeze(X,N)
    return X