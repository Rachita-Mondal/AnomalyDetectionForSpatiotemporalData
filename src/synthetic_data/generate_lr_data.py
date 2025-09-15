import numpy as np

from src.multilinear_ops.merge_tucker import merge_tucker
from src.multilinear_ops.qmult import qmult


def generate_low_rank_data(dim, ranks, seed=None, return_factors=False):
    '''Generates low-rank tensor data with dimensions `dim` and ranks `ranks`.
    Parameters:
        dim: Dimensions of the tensor
        ranks: Ranks of the tensor
    Outputs:
        T: Tensor of order `len(dim)`.
    '''
    rng = np.random.default_rng(seed)
    n = len(dim)
    C = rng.normal(0,1,ranks)
    U = [qmult(dim[i])[:,:ranks[i]] for i in range(n)]
    if return_factors:
        return merge_tucker(C, U, np.arange(n)), C, U
    return merge_tucker(C, U, np.arange(n))
