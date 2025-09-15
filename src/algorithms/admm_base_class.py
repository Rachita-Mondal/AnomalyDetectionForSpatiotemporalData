""" THIS IS NOT USED!
- MERT INDIBI, 20 August 2023"""

from abc import ABC, abstractmethod, abstractproperty
import time
from typing import Any
import numpy as np


class TwoBlockADMMBase(ABC):
    """Abstract base class for ADMM optimization algorithms."""

    @abstractmethod
    def __init__(self, args: Any, **kwargs: Any):
        pass

    @abstractmethod
    def __call__(self, args: Any, **kwargs: Any):
        """Iterate over the algorithm until convergence or reaching maximum iterations."""
        pass

    @property
    def it(self):
        """Current iteration number."""
        pass

    @property
    def r(self):
        """List of primal residuals for each iteration of the ADMM algorithm."""
        pass

    @property
    def s(self):
        """List of dual residuals of each iteration of the ADMM algorithm."""
        pass

    @property
    def objective(self):
        """List of objective values of each iteration of the ADMM algorithm."""
        pass
    
    @property
    def max_it(self):
        """Maximum number of iterations."""
        pass

    @property
    def rhos(self):
        """List of ADMM step sizes in each iteration of the ADMM algorithm."""
        pass


    @property
    def verbosity(self):
        """Verbosity level of the algorithm."""
        pass

    @property
    def err_tol(self):
        pass