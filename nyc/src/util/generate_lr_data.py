import numpy as np
from src.util.merge_Tucker import merge_Tucker
from src.util.t2m import t2m
from src.util.m2t import m2t
from src.util.qmult import qmult


def generate_low_rank_data(dim, ranks, seed=None):
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
    # U = [np.linalg.svd(
    #     np.random.standard_normal((dim[i], ranks[i])),
    #     full_matrices=False
    #     )[0] for i in range(n)]
    U = [qmult(dim[i])[:,:ranks[i]] for i in range(n)]
    return merge_Tucker(C, U, np.arange(n))
