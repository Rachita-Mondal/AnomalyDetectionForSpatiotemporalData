from abc import ABC, abstractmethod
from typing import Any


class CMA_ES(ABC):
    def __init__(self,
                dim: int,    # Dimension of the solution
                mu: int,     # Population size
                lda: int,   # number of offsprings
                seed: int=0, # Seed for random number generator
                max_evals: int = 1000,
                max_gen: int = 100,
                **kwargs: dict[str, Any]):
        self.dim = dim
        self.mu = mu
        self.lda = lda
        self.seed = seed
        self.evaluated = 0
        self.gen = 0
        self.max_evals = max_evals
        self.max_gen = max_gen
        self.population = None
        self._sample()
        self._best_solution = 


    @abstractmethod
    def _sample(self):
        pass

    @abstractmethod
    def _selection(self):
        pass

    @abstractmethod
    def _recombination(self):
        pass

    @abstractmethod
    def ask(self):
        pass

    @abstractmethod
    def tell(self):
        pass
    
    

class BestSolution(ABC):
    """Best solution(s) found by the algorithm"""
    def __init__(self,
                 dim,
                 num_objectives=1,
                 x=None,
                 objective=0,
                 direction='minimize',
                 **kwargs):
        self.dim = dim
        self.x = x
        self.objective = objective
        self._age = 0

    @abstractmethod
    def update(self):
        pass
    
    @abstractmethod
    def fitness(self):
        return self.fitness

    @property
    def age(self):
        """The age of the best solution(s)"""
        return self.age

    
    def __repr__(self):
        return f"BestSolution(x={self.x}, fitness={self.fitness})"
    
    def __str__(self):
        return f"BestSolution(x={self.x}, fitness={self.fitness})"

    def __lt__(self, other):
        return self.fitness < other.fitness
    
    def __gt__(self, other):
        return self.fitness > other.fitness
    
    def __eq__(self, other):
        return self.fitness == other.fitness
    
    def __le__(self, other):
        return self.fitness <= other.fitness
    
    def __ge__(self, other):
        return self.fitness >= other.fitness
    
    def __ne__(self, other):
        return self.fitness != other.fitness
    
    
    