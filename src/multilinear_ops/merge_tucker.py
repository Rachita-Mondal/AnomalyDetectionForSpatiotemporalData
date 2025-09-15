from src.multilinear_ops.t2m import t2m
from src.multilinear_ops.m2t import m2t


def merge_tucker(C, U, dims, transpose=False):
    """Merge tensor and factor matrices in Tucker format.

    Parameters:
        C: numpy.ndarray
            Tensor to be merged. Should have the same dimensions with the
            number of columns in `U`.

        U: list(numpy.ndarray)
            Factor matrices. I(n) x r(n)

        dims: list(numpy.ndarray)
            Dimensions along which the merge will occur. Should have same or
            smaller length then `U`.

        transpose: bool
            Boolean showing if U's should be transposed.

    Outputs:
        X: numpy.ndarray
            Merged tensor. Shape should be the same with row sizes of `U` along
            modes `dims`.
    """
    if hasattr(dims, '__len__'):
        n = len(dims)
    else:
        dims = [dims]
        n = 1
    X = C.copy()
    sizes = list(X.shape)
    for i in range(n):
        V = U[i].copy().T if transpose else U[i].copy()
        sizes[dims[i]] = V.shape[0]
        X = m2t(V @ t2m(X, dims[i]+1), sizes, dims[i]+1)
    return X
