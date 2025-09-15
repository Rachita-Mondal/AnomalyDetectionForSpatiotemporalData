"""
This module implements the following algorithms for Robust Principal Component Analysis (RPCA)

1. Exact RPCA using Augmented Lagrangian Multiplier (ALM)
2. Inexact RPCA using ALM (TODO)
3. Exact Matrix completion with Robust-RPCA using ALM (TODO)
4. Inexact Matrix completion with Robust-RPCA using ALM (TODO)

Author: Mert Indibi, Jan 2024

References:
-----------
1. Candes, E. J., Li, X., Ma, Y., & Wright, J. (2009). Robust principal component analysis? Journal of the ACM (JACM), 58(3), 11.
"""

from time import perf_counter

import numpy as np
from numpy.linalg import norm

from src.proximal_ops.soft_hosvd import soft_moden
from src.proximal_ops.soft_treshold import soft_treshold

class ExactRpcaALM:
    """Exact Robust Principal Component Analysis using Augmented Lagrangian Multiplier

    Solves the following problem introduced by Candes et al. (2009)
        minimize ||X||_* + lda||S||_1
        X, S
        such that X + S = Y

    Y is observation matrix, X is assumed to be low-rank and S is assumed to be sparse
    """
    
    def __init__(self, **kwargs):
        self.it = 0
        self.times = {"X_update":[],  "S_update":[],
                    "it":[]}
        self.obj = []   # objective function
        self.rhos = []  # step size parameter
        self.r = []     # norm of primal residual
        self.s = []     # norm of dual residual
        self.Y = None   # observation matrix
        self.X = None   # low-rank component
        self.S = None   # sparse component
        self.lda = None # regularization parameter
        self.err_tol = kwargs.get('err_tol', 1e-6)

        # Dynamic step size parameters
        self.rho_update_thr = kwargs.get('rho_update_thr',100)
        self.rho_upd = kwargs.get('rho_upd',-1)
        self.verbose = kwargs.get('verbose',1)
        self.benchmark = kwargs.get('benchmark',True)

    def __call__(self, Y, **kwargs):
        n = Y.shape
        if len(n) != 2:
            raise ValueError("Inappropriate input dimensions")
        lda = kwargs.get('lda', 1/np.sqrt(np.max(n)))
        rho = kwargs.get('rho', n[0]*n[1]*0.25/np.sum(np.abs(Y)))
        self.rhos.append(rho)
        self.lda = lda
        maxit = kwargs.get('maxit',100)

        # Initialize primal variables
        self.X = np.zeros(n)
        self.S = np.zeros(n)
        # Initialize dual variables
        self.Lda = np.zeros(n)
        while self.it < maxit:
            tstart = perf_counter()
            X, nuc_norm = self.X_update(Y, self.S, self.Lda, rho)

            S = self.S_update(Y, X, self.Lda, rho, lda)

            r = X+S-Y
            self.Lda += rho*r
            
            self.obj.append(nuc_norm+lda*np.sum(np.abs(S)))
            self.r.append( norm(r,'fro'))
            self.s.append( rho*norm(S - self.S,'fro') )
            self.X = X
            self.S = S
            self.it +=1
            self.times['it'].append(perf_counter()-tstart)
            

            if self.verbose and self.it>1:
                self.report_iteration()
            # Check convergence
            if np.max((self.r[-1], self.s[-1])) < self.err_tol:
                print("Converged!")
                break
            else: # Update rho
                if self.rho_upd !=-1:
                    rho = self.update_rho(rho)
                self.rhos.append(rho)
        return X,S
    

    def X_update(self, Y, S, Lda, rho):
        """Update X
        """
        if self.benchmark:
            start = perf_counter()
            X, nuc_norm = soft_moden(Y-S-Lda/rho, 1/rho, 1)
            end = perf_counter()
            self.times['X_update'].append(end-start)
        else:
            X, nuc_norm = soft_moden(Y-S-Lda/rho, 1/rho, 1)
        return X, nuc_norm


    def S_update(self, Y, X, Lda, rho, lda):
        """Update S
        """
        if self.benchmark:
            start = perf_counter()
            S = soft_treshold(Y-X-Lda/rho, lda/rho)
            end = perf_counter()
            self.times['S_update'].append(end-start)
        else:
            S = soft_treshold(Y-X-Lda/rho, lda/rho)
        return S


    def update_rho(self, rho):
        """Update dynamic step size parameter based on residuals
        """
        if self.r[-1] > self.rho_update_thr*self.s[-1]:
            rho = rho*self.rho_upd
        elif self.s[-1] > self.rho_update_thr*self.r[-1]:
            rho = rho/self.rho_upd
        return rho


    def report_iteration(self):
        print(f"It-{self.it}: \t## {self.times['it'][-1]:.2f} sec. \t obj={self.obj[-1]:.5f} \t## del_obj = {self.obj[-1]-self.obj[-2]:.5f}")
        print(f"|r|={self.r[-1]:.5f} \t ## |s|={self.s[-1]:.5f} \t ## rho={self.rhos[-1]:.5f}")


class ExactMcRpca:
    """Exact Matrix completion with Robust-RPCA using Augmented Lagrangian Multiplier
    """
    pass

class InexactRpca:
    """Inexact Robust Principal Component Analysis using Augmented Lagrangian Multiplier
    """
    pass

class InexactRpca:
    """Inexact Robust Principal Component Analysis using Augmented Lagrangian Multiplier
    """
    pass
