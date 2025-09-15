from time import perf_counter

import numpy as np
from numpy.linalg import inv, norm

from src.proximal_ops.soft_hosvd import soft_moden
from src.proximal_ops.soft_treshold import soft_treshold


class ORPCA:
    """Online Robust PCA via Stochastic Optimization

    Solves the following problem introduced by 
    """
    def __init__(self, lda1, lda2, dim, rank, **kwargs):
        seed = kwargs.get('seed', None)
        rng = np.random.default_rng(seed)
        self.mode = kwargs.get('mode', 'naive')
        if self.mode not in ['naive', 'dynamic', 'windowed']:
            raise ValueError("Invalid mode")
        self.initially_slow_L_update = kwargs.get('initially_slow_L_update', True)
        t0 = kwargs.get('t0', 1)

        self.lda1 = lda1
        self.lda2 = lda2
        self.dim = dim
        self.rank = rank
        self.err_tol = kwargs.get('err_tol', 1e-6)
        self.times = {"projection":[], "L_update":[], "it":[]}
        self.sample_loss = []
        self.surrogate_loss = []
        self.L_change = []
        self.sample_count = 0


        self.L = rng.standard_normal(size=(dim, rank))

        if self.initially_slow_L_update:    
            for j in range(rank):
                self.L[:,j] = self.L[:,j]/ np.max( (1, norm(self.L[:,j]) ))
            
            self.A = t0*np.eye(rank)
            self.B = t0*self.L
        else:
            self.A = np.zeros((rank,rank))
            self.B = np.zeros((dim,rank))
    
        self.A_prev = None
        self.B_prev = None
        self.forget_factor = kwargs.get('forget_factor', 0.98)
        self.window_size = kwargs.get('window_size', 100)
        self.projection_maxit = kwargs.get('projection_maxit', 100)

        self.benchmark = kwargs.get('benchmark', True)
        self.verbose = kwargs.get('verbose', 1)


    def run_sequence(self, Z, **kwargs):
        """Run ORPCA on the columns of matrix Z as samples
        """
        use_online = kwargs.get('use_online', True)
        n_samples = Z.shape[1]
        X = np.zeros(Z.shape)
        R = np.zeros((self.rank, n_samples))
        E = np.zeros(Z.shape)
        for i in range(n_samples):
            if self.benchmark:
                tstart = perf_counter()
            
            X[:,i], R[:,i], E[:,i] = self.run_sample(Z[:,i])

            if self.benchmark:
                self.times["it"].append(perf_counter() - tstart)
            if self.verbose > 0:
                rprt = "-"*30+'\n'
                rprt = rprt + f"t={self.sample_count}# \t loss={self.sample_loss[-1]:.5f}"
                rprt = rprt + f"# \t surrogate loss={self.surrogate_loss[-1]:.5f}"
                print(rprt)
        
        if use_online:
            return X, R, E
        else:
            return self.L@R.T, R, E


    def run_sample(self, z):
        r, e, loss = self.project_r(z)
        self.sample_loss.append(loss)
        self.A = self.update_A(r)
        self.B = self.update_B(z, r, e)
        x = self.L@r
        self.L = self.update_L(z,r,e)
        self.surrogate_loss.append(loss + self.lda1*0.5*norm(self.L,ord='fro'))
        self.sample_count += 1
        return x.ravel(), r.ravel(), e.ravel()


    def project_r(self, z):
        """Solves equation (7) in the paper. (Not fast, but simple)
        """
        if self.benchmark:
            tstart = perf_counter()
        z = z.reshape((self.dim,1))
        r = np.zeros((self.rank,1))
        e = np.zeros((self.dim,1))
        r_new = np.zeros((self.rank,1))
        e_new = np.zeros((self.dim,1))
        it = 0
        converged = False
        inv_term = inv(self.L.T@self.L + self.lda1*np.eye(self.rank))@self.L.T
        while it<self.projection_maxit:
            r_new = inv_term@(z - e)
            e_new = soft_treshold(z - self.L@r_new, self.lda2)
            
            r, r_new = r_new, r
            e, e_new = e_new, e
            if norm(r_new - r)/self.dim < self.err_tol and norm(e_new - e)/self.dim < self.err_tol:
                converged = True
                break
            it += 1
        
        loss = 0.5*norm(z - self.L@r - e, ord=2) + 0.5*self.lda1*norm(r, ord=2) + self.lda2*norm(e, ord=1)
        if self.benchmark:
            self.times["projection"].append(perf_counter() - tstart)
        if self.verbose > 1:
            print(f"r,e update:\tloss={loss:.4f}\tIterations: {it}\t Converged? {converged}")
        return r, e, loss


    def update_A(self, r):
        if self.mode == 'naive':
            return self.A + r@r.T
        
        elif self.mode == 'dynamic':
            return self.A *self.forget_factor+ r@r.T
        
        elif self.mode == 'windowed':
            raise NotImplementedError("Windowed mode not implemented")


    def update_B(self, z, r, e):
        z = z.reshape((self.dim,1))
        e = e.reshape((self.dim,1))
        r = r.reshape((self.rank,1))
        if self.mode == 'naive':
            return self.B + (z - e)@r.T
        
        elif self.mode == 'dynamic':
            return self.B*self.forget_factor + (z - e)@r.T
        
        elif self.mode == 'windowed':
            raise NotImplementedError("Windowed mode not implemented")


    def update_L(self, z=None, r=None, e=None):
        """The basis update in the paper. Solves the equation (8)
        """
        if self.benchmark:
            tstart = perf_counter()
        A = self.A + self.lda1*np.eye(self.rank)
        L_old = self.L.copy()
        L = self.L.copy()
        L_new = self.L.copy()
        # gL = (L@A - self.B)#/(self.sample_count+1)
        r = r.reshape((self.rank,1)); e = e.reshape((self.dim,1)); z = z.reshape((self.dim,1));
        gL = ((self.L@r + e - z)@r.T + self.lda1*self.L)/(self.sample_count+1)
        L_new = L - gL@inv(A)*(self.sample_count+1)
        it = 0
        converged = False
        # L_change = np.inf
        
        # while it < 100:
        # for j in range(self.rank):
        #     temp = (self.B[:,j] - L_new@A[:,j])/A[j,j] + L_new[:,j]
        #     L_new[:,j] = temp#/max((1, norm(temp)))
            # L_new[:,j] = L[:,j] - gL[:,j]/(A[j,j]) 
            
            # else:
            # L_new[:,j] = temp
        
        # if norm(L_new) > self.clip_threshold:
        #     L_new = self.clip_threshold*L_new/norm(L_new)
        #     print("Clipping")
        
        L_change = norm(L_new-L, ord='fro')
    
        L = L_new.copy()
            # it += 1
        # L = self.B@inv(A)
        if self.benchmark:
            self.times["L_update"].append(perf_counter() - tstart)
        
        self.L_change.append(norm(L - L_old, 'fro'))
        if self.verbose >1:
            print(f"L update:\t||Lt - Lt-1||={self.L_change[-1]:.4f}\t Iterations: {it}\t Converged? {converged}")
        return L
