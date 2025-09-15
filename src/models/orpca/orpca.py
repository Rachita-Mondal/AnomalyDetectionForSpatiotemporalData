import warnings
from time import perf_counter

import numpy as np
from numpy.linalg import norm

from src.proximal_ops.soft_treshold import soft_treshold
from src.models.rpca.rpca import ExactRpcaALM


class ORPCA:
    """Online Robust PCA via Stochastic Optimization

    Models:
    - 'hard': Loss function assumes laplacian noise prior
    - 'soft': Loss function assumes gaussian and sparse noise priors.

    Operation modes:
    The method used to estimate the loss function with observed samples.
    - 'memoryless': 
    - 'cumulative_avg': cumulative average of observed samples.
    - 'exponential_ma': exponential moving average of observed samples.
    - 'simple_ma': simple moving average of observed samples.

    Optimizers:
    - 'SGD': Stochastic Gradient Descend method
    - 'RDA': Regularized Dual Averaging method
    - 'AdaGrad':
    - 'Adam':

    Initialization:
    - 'random': Initializes the dictionary L of size rank with random values.
    - 'PCP': Initializes the dictionary L with the Principal Component Pursuit method.
    """

    def __init__(self, dim, model='soft', **kwargs):
        """Set up ORPCA model with different operation modes and optimizers
        
        Args:
            dim (int): Dimension of the samples
            model (str, optional): ORPCA model used. Defaults to 'soft'
            mode (str, optional): Operation mode. Defaults to 'running_avg'
            lda_nuc (float, optional): Nuclear norm regularization parameter. Defaults to 1.
            lda1 (float, optional): Sparse error regularization parameter. Defaults to 1.
            rank (int, optional): Rank of the dictionary. If burnin samples are provided,
                the rank is estimated using RPCA on provided samples. Defaults to 1.
            L_initialization (str, optional): Initialization method for the dictionary L. Defaults to 'random'.
            optimizer (str, optional): Optimizer used for updating the dictionary L. Defaults to 'SGD'.
            annealing_mode (str, optional): Annealing mode for the step size. Defaults to 'sqrt'.
            seed (int, optional): Random seed. Defaults to 0.
            re_update_tol (float, optional): Tolerance for the projection minimization. Defaults to 1e-6.
            re_update_maxit (int, optional): Maximum number of iterations for the projection minimization. Defaults to 10000.
        """
        self.model = kwargs.get('model', 'soft')
        self.mode = kwargs.get('mode', 'running_avg')
        self.lda_nuc = kwargs.get('lda_nuc', 1)
        self.lda1 = kwargs.get('lda1', 1)
        self.dim = dim
        self.seed = kwargs.get('seed', 0)
        self.rng = np.random.default_rng(self.seed)
        self.rs_update_tol = kwargs.get('rs_update_tol', 1e-6)
        self.rs_update_maxit = kwargs.get('rs_update_maxit', 10000)
        # Bookkeeping
        self.times = {"projection":[], "L_update":[], "it":[]}
        self.losses = []
        self.L_change = []
        self.t = 0
        self.verbose = kwargs.get('verbose', 1)
        # Initialize the dictionary L
        self.L_init = kwargs.get('L_init', 'random')
        if self.L_init not in ['random', 'PCP']:
            raise ValueError("Invalid initialization")
        self.rank = kwargs.get('rank', None)
        burnin_samples = kwargs.get('burnin_samples', None)
        self._init_L(burnin_samples)

        # Initialize operation mode
        self.mode = kwargs.get('mode', 'cumulative_avg')
        if self.mode not in ['memoryless', 'cumulative_avg', 'simple_ma', 'exponential_ma']:
            raise ValueError("Invalid mode")
        self._init_mode()
        
        # Initialize optimizer
        self.optimizer = kwargs.get('optimizer', 'SGD')
        if self.optimizer not in ['SGD', 'RDA', 'AdaGrad', 'Adam', 'BCD']:
            raise ValueError("Invalid optimizer")
        self.optimizer_param = kwargs.get('optimizer_param', {})
        self._init_optimizer()
        

    def run_sequence(self, Z, **kwargs):
        """Run ORPCA on the columns of matrix Z as samples
        
        Args:
            Z (np.ndarray): Matrix of samples
        
        Returns:
            X (np.ndarray): Reconstructed samples
            R (np.ndarray): Dictionary coefficients for the samples
            E (np.ndarray): Sparse error in samples
        """
        n_samples = Z.shape[1]
        X = np.zeros(Z.shape)
        R = np.zeros((self.rank, n_samples))
        E = np.zeros(Z.shape)
        benchmark = kwargs.get('benchmark', False)
        if benchmark:
            X_gt = kwargs.get('X_gt', None)
            E_gt = kwargs.get('E_gt', None)
            U_gt = kwargs.get('U_gt', None)
            self.expressed_variance = np.zeros(n_samples)
            self.reconstruction_error = np.zeros(n_samples)
            self.sparse_error = np.zeros(n_samples)

        for i in range(n_samples):
            X[:,i], R[:,i], E[:,i] = self.run_sample(Z[:,i])
            if benchmark:
                self.expressed_variance[i] = np.linalg.norm(U_gt@U_gt.T@self.L)/np.linalg.norm(self.L)
                self.reconstruction_error[i] = np.sum((X[:,i] - X_gt[:,i])**2)/np.sum(X_gt[:,i]**2)
                self.sparse_error[i] = np.sum((E[:,i] - E_gt[:,i])**2)/np.sum(E_gt[:,i]**2)
                if self.verbose > 0:
                    print(f"Sample {i+1}/{n_samples}\t EV:{self.expressed_variance[i]:.4f}\t RecErr:{self.reconstruction_error[i]:.4f}\t SpErr:{self.sparse_error[i]:.4f}")
        return X, R, E


    def run_sample(self, y):
        """Run a single sample of ORPCA
        
        Args:
            y (np.ndarray): Sample to be processed
            
        Returns:
            x (np.ndarray): Reconstructed sample
            r (np.ndarray): Dictionary coefficients for the sample
            s (np.ndarray): Sparse error in sample
        """
        self.t += 1
        start = perf_counter()
        r, s, loss, it, converged = self._solve_r_s(y)
        x = np.matmul(self.L, r)
        proj_time = perf_counter() - start
        self.times["projection"].append(proj_time)
        if self.verbose>0:
            print(f"t:{self.t}\t min_[r,e] >> loss:{loss}\t it: {it} \t time:{proj_time:.4f} \t Converged?:{converged}")
        self.losses.append(loss)
        
        self._update_regularized_loss_fn(y, r, s)

        L_start = perf_counter()
        L = self._update_L()
        L_upd_time = perf_counter() - L_start
        self.times["L_update"].append(L_upd_time)
        self.L_change.append(norm(L - self.L)/norm(self.L))
        self.L = L
        if self.verbose>0:
            print(f"t:{self.t}\t L_upd >> L_change:{self.L_change[-1]:.5f}\t time:{L_upd_time:.4f}")
        return x, r, s
    

    def _init_L(self, burnin_samples):
        if self.L_init == 'random':
            if self.rank is None:
                warnings.warn("Rank not provided, using dim//5 as rank")
                self.rank = self.dim//5
            self.L = self.rng.standard_normal(size=(self.dim, self.rank))
        elif self.L_init == 'PCP':
            rpca = ExactRpcaALM()
            X, _ = rpca(burnin_samples, lda=self.lda1)
            U, s, Vh = np.linalg.svd(X)
            rank = np.sum(s > 1e-10)
            self.L = U[:,:rank]@np.diag(np.sqrt(s[:rank]))@Vh[:rank,:]
            self.rank = rank
        else:
            raise ValueError("Invalid initialization")
    

    def _update_L(self):
        """Update the dictionary L according to the optimizer used
        """
        match self.optimizer:
            case 'SGD':
                return run_sample_sgd(self.L, self.A, self.B, self.t, self.lda_nuc, **self.optimizer_param)
            case 'Adam':
                L_new, L_m, L_v = run_sample_adam(self.L, self.A, self.B, self.t, self.lda_nuc, **self.optimizer_param)
                self.optimizer_param['L_m'] = L_m
                self.optimizer_param['L_v'] = L_v
                return L_new
            case 'BCD':
                return run_sample_bcd(self.L, self.A, self.B, self.t, self.lda_nuc, **self.optimizer_param)
            case 'RDA':
                raise NotImplementedError("Not implemented yet")
            case _:
                raise ValueError("Invalid optimizer")
    

    def _solve_r_s(self, y):
        match self.model:
            case 'hard':
                return solve_r_s_hard(self.L, y, self.lda1)
            case 'soft':
                return solve_r_s_soft(self.L, y, self.lda_nuc, self.lda1)
            case _:
                raise ValueError("Invalid model")


    def _update_regularized_loss_fn(self, y, r, s):
        """Estimate the regularized loss function according to operation mode

        Args:
            y (np.ndarray): Observed sample
            r (np.ndarray): Estimated coefficients for the sample
            s (np.ndarray): Estimated sparse error in the sample
        """
        match self.mode:
            case 'memoryless':
                self.A = np.outer(r, r)
                self.B = np.outer(y - s, r)
            case 'cumulative_avg':
                self.A = self.A*(self.t - 1)/self.t + np.outer(r, r)/self.t
                self.B = self.B*(self.t - 1)/self.t + np.outer(y - s, r)/self.t
            case 'simple_ma':
                raise NotImplementedError("Not implemented yet")
            case 'exponential_ma':
                raise NotImplementedError("Not implemented yet")
            case _:
                raise ValueError("Invalid mode")


    def _init_mode(self):
        if self.mode == 'simple_ma':
            self.A = np.zeros((self.rank, self.rank))
            self.B = np.zeros((self.dim, self.rank))
            self.A_old = np.zeros((self.rank, self.rank))
            self.B_old = np.zeros((self.dim, self.rank))
        else:
            self.A = np.zeros((self.rank, self.rank))
            self.B = np.zeros((self.dim, self.rank))

    def _init_optimizer(self):
        if self.optimizer == 'SGD':
            pass
        elif self.optimizer == 'Adam':
            self.optimizer_param['beta1'] = self.optimizer_param.get('beta1', 0.9)
            self.optimizer_param['beta2'] = self.optimizer_param.get('beta2', 0.999)
            self.optimizer_param['epsilon'] = self.optimizer_param.get('epsilon', 1e-8)
            self.L_m = np.zeros(self.L.shape) # First Moment
            self.L_v = np.zeros(self.L.shape) # Second Moment
        elif self.optimizer == 'BCD':
            pass
        elif self.optimizer == 'RDA':
            pass
        else:
            raise ValueError("Invalid optimizer")
    

def run_sample_sgd(L_t, A_t, B_t, t, lda_nuc, **kwargs):
    """Run a single sample of SGD for ORPCA

    Args:
        L_t (np.ndarray): _description_
        A_t (np.ndarray): _description_
        B_t (np.ndarray): _description_
        t (float): _description_
        lda_nuc (float): _description_
        annealing_mode (str, optional): _description_. Defaults to 'constant'.
        step_size (float, optional): _description_. Defaults to 1e-4.
    """
    annealing_mode = kwargs.get('annealing_mode', 'sqrt')
    step_size = kwargs.get('step_size', 1e-3)
    L_new = L_t.copy()
    if annealing_mode == 'constant':
        pass
    elif annealing_mode == 'sqrt':
        step_size /= np.sqrt(t)
    A = A_t + lda_nuc*np.identity(A_t.shape[0])/t
    gt = np.matmul(L_new, A) - B_t
    L_new -= step_size*gt
    return L_new


def run_sample_bcd(L_t, A_t, B_t, t, lda_nuc, **kwargs):
    annealing_mode = kwargs.get('annealing_mode', 'sqrt')
    step_size = kwargs.get('step_size', 1e-3)
    A = A_t + lda_nuc*np.identity(A_t.shape[0])/t
    L_new = L_t.copy()
    if annealing_mode == 'sqrt':
        step_size /= np.sqrt(t)
        for i in range(L_new.shape[1]):
            gt = np.matmul(L_new, A[:,i]) - B_t[:,i]
            L_new[:,i] -= step_size*gt
    elif annealing_mode == 'constant':
        for i in range(L_new.shape[1]):
            gt = np.matmul(L_new, A[:,i]) - B_t[:,i]
            L_new[:,i] -= gt/A[i,i]
    return L_new

def run_sample_adam(L_t, A_t, B_t, t, lda_nuc, **optimizer_param):
    beta1 = optimizer_param.get('beta1', 0.9)
    beta2 = optimizer_param.get('beta2', 0.999)
    epsilon = optimizer_param.get('epsilon', 1e-8)
    step_size = optimizer_param.get('step_size', 1e-3)
    L_m = optimizer_param.get('L_m', np.zeros(L_t.shape))
    L_v = optimizer_param.get('L_v', np.zeros(L_t.shape))
    L_new = L_t.copy()
    A = A_t + lda_nuc*np.identity(A_t.shape[0])/t
    if optimizer_param.get('bcd', False):
        for i in range(L_new.shape[1]):
            gt = np.matmul(L_new, A[:,i]) - B_t[:,i]
            L_m[:,i] = beta1*L_m[:,i] + (1 - beta1)*gt
            L_v[:,i] = beta2*L_v[:,i] + (1 - beta2)*gt**2
            m_hat = L_m[:,i]/(1 - beta1**t)
            v_hat = L_v[:,i]/(1 - beta2**t)
            L_new[:,i] -= step_size*m_hat/(np.sqrt(v_hat) + epsilon)
    else:
        gt = np.matmul(L_new, A) - B_t
        L_m = beta1*L_m + (1 - beta1)*gt
        L_v = beta2*L_v + (1 - beta2)*gt**2
        m_hat = L_m/(1 - beta1**t)
        v_hat = L_v/(1 - beta2**t)
        L_new -= step_size*m_hat/(np.sqrt(v_hat) + epsilon)
    return L_new, L_m, L_v


def run_sample_rda():
    pass


def run_sample_adagrad():
    pass
    


def sample_loss_hard(L, y, r, lda1):
    r_mag = np.sum(r**2)
    e_mag = np.sum(np.abs(y - np.matmul(L, r)))
    return 0.5*r_mag + lda1*e_mag


def sample_loss_soft(L, y, r, e, lda_nuc, lda1):
    err = np.sum((y - np.matmul(L, r))**2)
    r_mag = np.sum(r**2)
    e_mag = np.sum(np.abs(e))
    return 0.5*err + 0.5*lda_nuc*r_mag + lda1*e_mag


def solve_r_s_soft(L, y, lda_nuc, lda1, maxit=10000, tol=1e-6):
    """Solves the problem
    min_{r,e} 0.5||y-Lr-e||_2^2 + 0.5*lambda_nuc||r||_2^2 + lambda_1*||e||_1
    
    """
    # intialization
    p, rank = L.shape
    r = np.zeros(rank)
    r_new = np.zeros(rank)
    s = np.zeros(p)
    s_new = np.zeros(p)
    I = np.identity(rank)
    converged = False
    it = 0

    LLt = np.matmul(
            np.linalg.inv(L.T@L + lda_nuc*I),
            L.T)
    
    while not converged and it < maxit:
        it += 1
        r_new = np.matmul(LLt, y - s)
        s_new = soft_treshold(y - np.matmul(L, r_new), lda1)
        r, r_new = r_new, r
        s, s_new = s_new, s

        stopc = np.max((np.linalg.norm(r - r_new), np.linalg.norm(s - s_new)))/np.linalg.norm(y)
        if stopc < tol:
            converged = True
    
    loss = sample_loss_soft(L, y, r, s, lda_nuc, lda1)
    return r, s, loss, it, converged


def solve_r_s_hard(L, y, lda1, maxit=10000, tol=1e-6):
    """Solves the problem,
    min_{r,s} 0.5||r||_2^2 + lambda1||s||_1
    s.t. Lr + s - y = 0

    with ALM
    """
    raise NotImplementedError("Not implemented yet")
    return r, s, loss, it, converged

