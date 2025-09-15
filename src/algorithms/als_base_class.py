from abc import ABC, abstractmethod, abstractproperty
from time import perf_counter

class ALSBaseClass(ABC):
    """Abstract base class for ALS optimization algorithms."""
    @abstractmethod
    def __init__(self, args, **kwargs):
        """Initialize the algorithm."""
        pass

    @abstractmethod
    def __call__(self):
        """Iterate over the algorithm until convergence or reaching maximum iterations."""
        pass


    @abstractmethod
    def objective_function(self):
        pass









