import numpy as np

class CMA_ES:

    def __init__(self,
                 dim,
                 mu=None,
                 lda=None,
                 boundaries=None,
                 direction='minimize',
                 seed=0, max_evals=1000, max_gen=100, **kwargs):
        """_summary_

        Parameters:
        -----------
        dim (int):
            Problem dimension.
        lda (int, optional):
            The size of the population. Defaults to 4 + floor(3 ln(n)).
        boundaries (np.array or torch.tensor, optional):
            If provided, box boundaries for the problem is created. Defaults to None.
        mu (int, optional):
            The number of parents for recombination. Defaults to None.
        seed (int, optional):
            random number generator seed. Defaults to 0.
        max_evals (int, optional):
            Maximum number of function evaluations allowed. Defaults to 1000.
        max_gen (int, optional):
            Maximum number of generations allowed. Defaults to 100.
        kwargs (dict):
            sigma (int, optional): Optimal solution is expected to be within
                mean +- 3*sigma. If boundaries are not provided, sigma is used to
                initialize global step size. Defaults to 1.
            device (str, optional): Device to run the algorithm. Defaults to 'cuda:0'

        """
        self.dim = dim
        self.mu = mu
        self.lda = lda
        self.boundaries = boundaries
        self.sigma = kwargs.get('sigma', 1)
        self.evaluated = 0
        self.gen = 0
        self.max_evals = max_evals
        self.max_gen = max_gen
        self.population = None
        self._best_solution = None
        self._best_objective = None
        self._best_age = None
        self.generation_mean_objective = []
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.cov = np.eye(dim)
        self.direction = direction
        self._initialize()

    def run_on_bbox_func(self, bbox_func):
        """Run the CMA-ES algorithm on a given function"""
        while self.evaluated < self.max_evals and self.gen < self.max_gen:
            self.iterate(bbox_func)    
        return self._best_solution

    def iterate(self,bbox_func):
        if self.evaluated >= self.max_evals or self.gen >= self.max_gen:
            return
        """Run the CMA-ES algorithm for one generation"""
        # Sample the population
        pop_objectives = np.zeros(self.lda)
        direction = -1 if self.direction == 'maximize' else 1
        Z = self.rng.standard_normal((self.dim, self.lda))
        Y = self.B @ (self.D * Z)
        X = self.mean + self.sigma * Y

        # Evaluate the population
        for i in range(self.lda):
            pop_objectives[i]= bbox_func(X[:,i])
            self.evaluated += 1
            if self.evaluated >= self.max_evals:
                print('Max evaluations reached')
                return self._best_solution, self._best_objective
        self.generation_mean_objective.append(pop_objectives.mean())


        # Selection: Sort the population by objective value
        idx = np.argsort(pop_objectives)
        Y = Y[:,idx]
        if self.direction == 'maximize':
            idx = idx[::-1]
        if self._best_objective is None or (pop_objectives[idx[0]]*direction < self._best_objective*direction):
            self._best_solution = X[:,idx[0]]
            self._best_objective = pop_objectives[idx[0]]
            self._best_age = 0
            msg = f"Gen-{self.gen} obj:{self._best_objective}" + f" sol {self._best_solution}"
            print(msg)
        else:
            self._best_age += 1
        

        # Recombination: Update the mean
        y_w_m = np.sum( self.w[:self.mu].reshape((1, self.mu))* Y[:,:self.mu], axis=1).reshape((self.dim, 1))
        # y_w = np.sum( self.w.reshape((1, self.lda))* Y, axis=1).reshape((self.dim, 1))
        self.mean += self.c_m * self.sigma * y_w_m

        # Recombination: Step size control
        self.p_sigma = (1 - self.c_sigma) * self.p_sigma
        self.p_sigma += np.sqrt(self.c_sigma * (2 - self.c_sigma) * self.mu_eff
                                ) * self.B @   (y_w_m/np.sqrt(self.D))
        self.sigma *= np.exp((self.c_sigma / self.d_sigma) * (np.linalg.norm(self.p_sigma) / self.mag_expct - 1))

        # Covariance matrix adaptation
        self.p_c = (1 - self.c_c) * self.p_c
        self.p_c += np.sqrt(self.c_c * (2 - self.c_c) * self.mu_eff) * y_w_m
        
        # h_sigma = 1 if np.linalg.norm(self.p_c) / np.sqrt(1 - (1 - self.c_c)**(2*(self.gen+1)) ) < 1.4 + 2/(self.dim+1) else 0

        self.C = (1- self.c_1 - self.c_mu*sum(self.w))*self.C
        self.C += self.c_1 * (self.p_c @ self.p_c.T )
        self.C += self.c_mu* self.w*Y @ Y.T
        self.D, self.B = np.linalg.eigh(self.C)
        self.D = self.D.reshape((self.dim, 1))
        self.gen += 1
        
    def _sample(self):
        pass

    def _selection(self):
        pass

    def _recombination(self):
        pass

    def ask(self):
        pass

    def tell(self):
        """Tell the algorithm the fitness of the solutions in the population"""
        pass
    

    def _initialize(self):
        """Default parameters for CMA-ES algorithm as per <cite>"""
        if self.lda is None:
            self.lda = int(np.floor(4 + 3 * np.log(self.dim)))
        else:
            self.lda = int(self.lda)
            #self.lda = np.array(self.lda)
        if self.mu is None:
            self.mu = int(np.floor(self.lda / 2))
        else:
            self.mu = int(self.mu)
            #self.mu = self.mu
        if self.boundaries is None:
            self.mean = np.zeros((self.dim,1))
        else:
            self.mean = (self.boundaries[0] + self.boundaries[1]) / 2
            self.sigma = np.abs( self.boundaries[1] - self.boundaries[0]) / 6
        
        w = np.log(self.lda / 2 + 0.5) - np.log(np.arange(1, self.lda + 1))
        w_p = w[w>=0]
        w_n = w[w<0]
        self.mu_eff = w_p.sum()**2 / (w_p**2).sum()
        self.mu_eff_n = w_n.sum()**2 / (w_n**2).sum()
        # Step-size control parameters
        self.c_sigma = (self.mu_eff + 2) / (self.dim + self.mu_eff + 5)
        self.d_sigma = 1 + 2 * np.maximum(0, 
                                          np.sqrt((self.mu_eff - 1) / (self.dim + 1)) - 1
                                          ) + self.c_sigma
        # Covariance matrix control parameters
        alpha_cov = 2 # Normally 2
        self.c_c = (4 + self.mu_eff / self.dim) / (self.dim + 4 + 2 * self.mu_eff / self.dim)
        self.c_1 = alpha_cov / ((self.dim + 1.3)**2 + self.mu_eff)
        self.c_mu = np.min([1 - self.c_1, 
                alpha_cov * (self.mu_eff - 1.75 + (1 / self.mu_eff)) / ((self.dim + 2)**2 + alpha_cov * self.mu_eff / 2)
                            ])
        # Selection and recombination parameters
        alpha_mu_p = w_p.sum()
        alpha_mu_n = 1+ self.c_1/self.c_mu
        alpha_posdef_n = (1-self.c_1-self.c_mu)/(self.dim*self.c_mu)
        self.w = np.where(w >= 0,
                                w/alpha_mu_p, # positive weights sum to 1
                                w*np.min([alpha_mu_p, alpha_mu_n, alpha_posdef_n])/self.mu_eff_n)
        self.c_m = 1
        self.d_sigma = 1 + 2 * np.maximum(0,
                                           np.sqrt((self.mu_eff - 1) / (self.dim + 1)) - 1
                                           ) + self.c_sigma

        self.mag_expct = np.sqrt(self.dim) * (1 - 1 / (4 * self.dim) + 1 / (21 * self.dim**2))
        self.C = np.eye(self.dim)
        self.B = np.eye(self.dim)
        self.D = np.ones((self.dim, 1))
        self.p_sigma = np.zeros((self.dim, 1))
        self.p_c = np.zeros((self.dim, 1))
