"""Higher-order Singular Value Decomposition Algorithm.

Proposed in:
    L. De Lathauwer, B. De Moor, and J. Vandewalle, A multilinear singular value decomposition,
    SIAM J. Matrix Anal. Appl., 21 (2000), pp. 1253–1278.

"""
import numpy as np

from src.multilinear_ops.mode_svd import mode_svd
from src.multilinear_ops.mode_product import mode_product

class HoSVD:
    """Higher-order Singular Value Decomposition
    """
    def __init__(self, X, n_ranks=None, client=None):
        """Initialize the algorithm.

        Args:
            X (np.ndarray): Tensor to be decomposed
            client (dask.distributed.Client): Dask client for distributed computation. Defaults to None.
            n_ranks (list): List of the truncated svd ranks. Defaults to None.
        """
        if X.dtype == int:
            self.X = X.astype(float)
        else:
            self.X = X
        self.dims = X.shape
        self.N = len(self.dims)
        self.client = client
        self.C = None
        if n_ranks is None:
            self.n_ranks = [self.dims[i] for i in range(self.N)]
            self.Us = [np.zeros((self.dims[i],self.dims[i])) for i in range(self.N)]
        else:
            self.n_ranks = n_ranks
            if len(n_ranks)!=self.N:
                raise ValueError('Rank must be a list of length N')
            self.Us = [np.zeros((self.dims[i],n_ranks[i])) for i in range(self.N)]


    def __call__(self):
        """Decompose the tensor.

        Returns:
            core (np.ndarray): Core tensor of the decomposition
            factors (list): List of the factor matrices
        """
        
        if self.client is None:
            for i in range(self.N):
                self.Us[i], _, _ = mode_svd(self.X, i+1, self.n_ranks[i])
        else:
            futures = []
            for i in range(self.N):
                futures.append(self.client.submit(mode_svd, self.X, i+1, self.n_ranks[i]))
            # results = self.client.gather(futures)
            results = [f.result() for f in futures]
            for i in range(self.N):
                self.Us[i], _, _ = results[i]

        self.C = mode_product(self.X, self.Us[0].T, 1)
        for i in range(1,self.N):
            self.C = mode_product(self.C, self.Us[i].T, i+1)
        return self.C, self.Us
