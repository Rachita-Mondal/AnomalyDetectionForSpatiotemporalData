"""Higher-order orthogonal iteration (HOOI) algorithm for Tucker Decomposition

Solves the Tucker decomposition problem below with alternating least squares (ALS).

    min_{C, A_1, ..., A_N} ||X - C \times_1 U_1 \times_2 ... \times_N U_N||_F^2

as introduced in:
 L. De Lathauwer, B. De Moor, and J. Vandewalle, On the best rank-1 and
 rank-(R1,R2,...,RN) approximation of higher-order tensors, 
 SIAM J. Matrix Anal.
 Appl., 21 (2000), 

Remarks:
    - The algorithm is not guaranteed to converge to the global optimum.
    - The algorithm is sensitive to initialization.
    - The algorithm is not guaranteed to converge to a stationary point.
    - The algorithm is not guaranteed to converge.
"""
from time import perf_counter

import numpy as np
import torch

from src.algorithms.als_base_class import ALSBaseClass
from src.models.tucker_decomp.hosvd import HoSVD
from src.multilinear_ops.mode_svd import mode_svd
from src.multilinear_ops.mode_product import mode_product


class HoOI(ALSBaseClass):
    """Higher-order Orthogonal Iteration (HoOI) algorithm for Tucker Decomposition.
    """
    @torch.no_grad
    def __init__(self, X, n_ranks, max_it=100, verbose=1, err_tol=1e-5, **kwargs):
        """Initialize HooI algorithm with Higher-order Singular Value Decomposition (HOSVD).

        Args:
            X (np.ndarray): Tensor
            n_ranks (_type_): _description_
            max_it (_type_): _description_
            verbose (int, optional): _description_. Defaults to 1.
            err_tol (_type_, optional): _description_. Defaults to 1e-5.

        Raises:
            ValueError: _description_
        """
        self.device = kwargs.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.dtype = kwargs.get('dtype', torch.float64)
        if isinstance(X, np.ndarray):
            self.X = torch.tensor(X, device=self.device, dtype=self.dtype)
        else:
            self.X = X
        
        self.X_norm = torch.linalg.norm(self.X)
        self.dims = X.shape
        self.n_ranks = n_ranks
        self.max_it = max_it
        self.verbose = verbose
        self.err_tol = err_tol
        self.it = 0
        self.converged = False
        self.objective = []
        self.times = []
        self.N = len(X.shape)

        if len(n_ranks) != len(X.shape):
            raise ValueError('Number of ranks and tensor mode do not match.')
        
        # Initialize with HOSVD
        hosvd = HoSVD(X, n_ranks)
        self.C, self.Us = hosvd()

    @torch.no_grad
    def __call__(self):
        """Decompose the tensor.

        Returns:
            core (np.ndarray): Core tensor of the decomposition
            factors (list): List of the factor matrices
        """
        while not self.converged and self.it < self.max_it:
            tic = perf_counter()
            for n in range(self.N):
                Y = torch.tensor(self.X, device=self.device, dtype=self.dtype)
                for m in range(self.N):
                    if m != n:
                        Y = mode_product(Y, self.Us[m].T, m+1)
                
                self.Us[n], s, _ = mode_svd(Y, n+1, self.n_ranks[n])
            
            self.objective.append(self.objective_function(s))
            self.times.append(perf_counter()-tic)
            if self.verbose:
                print(f'Iteration {self.it}: ||X|| - ||C|| = {self.objective[-1]}')
            if self.objective[-1] < self.err_tol:
                self.converged = True

            self.it += 1
        
        self.C = self.X
        for i in range(self.N):
            self.C = mode_product(self.C, self.Us[i].T, i+1)
        
        return self.C, self.Us
    
    def objective_function(self, s):
        """Compute objective function
        """
        return self.X_norm**2 - np.linalg.norm(s)**2
