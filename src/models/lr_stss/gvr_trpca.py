import numpy as np
from time import time
from numpy.linalg import norm#, inv
# from scipy.sparse import csr_matrix
# from scipy.linalg import eigh
import scipy.io as sio

from src.util.m2t import m2t
from src.util.t2m import t2m
from src.util.matricize import matricize
from src.util.tensorize import tensorize
from src.util.soft_treshold import soft_treshold
from src.util.soft_hosvd import soft_moden


class gvr_trpca:
    """Graph Variance Regularized Tensor Robust PCA.
    """
    def __init__(self, L_t, L_l, loc_g_type="cartesian",
                **kwargs):
        self.it = 0
        self.times = {"X":[], "W":[], "Wt":[], "Wl":[],
                      "Xi":[], "S":[], "S_gft":[],
                      "S_mult":[], "S_add":[], "S_dual":[], "it":[]}
        self.obj = []
        # norms of primal residual
        self.r1 = []
        self.r2 = []
        # norms of dual residual 
        self.s1 = []
        self.s2 = []
        # Initialize the variables.
        self.Y = None
        self.X = None
        self.S = None
        self.rho1s = []
        self.rho2s = []
        self.rho1_mu = kwargs.get('rho1_mu',20)
        self.rho2_mu = kwargs.get('rho2_mu',20)
        self.rho1_upd = kwargs.get('rho1_upd',-1)
        self.rho2_upd = kwargs.get('rho2_upd',-1)
        
        # Initialize inverse operator
        self.L_t = L_t
        self.L_l = L_l
        self.V_r = kwargs.get('V_r', None) # Row mode GFT
        self.V_c = kwargs.get('V_c', None) # Column mode GFT
        self.V_l = kwargs.get('V_l', None) # Location modes GFT
        self.V_t = kwargs.get('V_t', None) # Time mode FT
        self.spec_r = kwargs.get('spec_r', None) # Row Spectrum
        self.spec_c = kwargs.get('spec_c', None) # Column Spectrum
        self.spec_l = kwargs.get('spec_l', None) # Location Laplacian Spectrum
        self.spec_t = kwargs.get('spec_t', None) # Time Spectrum
        self.loc_g_type = loc_g_type

        if loc_g_type != "direct": # Location factor graphs are provided
            if loc_g_type == "cartesian":
                Ir = np.ones(len(self.spec_r))
                Ic = np.ones(len(self.spec_c))
                self.spec_l = np.kron(self.spec_r, Ic) + np.kron(Ir, self.spec_c)
            elif loc_g_type == "kronecker":
                raise NotImplementedError("Kronecker product graph solution not yet implemented!")
            elif loc_g_type == "strong":
                raise NotImplementedError("Strong product graph solution not yet implemented!")
            else:
                raise ValueError("Invalid location graph type!")
        else: # Location graph is provided
            if isinstance(self.V_l, type(None)) or isinstance(self.spec_l, type(None)):
                raise ValueError("Location GFT is not provided!")
            else:
                self.loc_g_type = "direct"

    def _S_gft(self, B, loc_modes, temp_mode):
        if self.loc_g_type != "direct":
            B = m2t(self.V_r.T@t2m(B, loc_modes[0]),B.shape, loc_modes[0])
            B = m2t(self.V_c.T@t2m(B, loc_modes[1]),B.shape, loc_modes[1])
            B = m2t(self.V_t.T@t2m(B, temp_mode),B.shape, temp_mode)
        else:
            B = tensorize(self.V_l.T@matricize(B, loc_modes), B.shape, loc_modes)
            B = m2t(self.V_t.T@t2m(B, temp_mode),B.shape, temp_mode)
        return B
    
    def _S_inv_gft(self, B, loc_modes, temp_mode):
        # Inverse GFT
        if self.loc_g_type != "direct":
            B = m2t(self.V_r@t2m(B, loc_modes[0]),B.shape, loc_modes[0])
            B = m2t(self.V_c@t2m(B, loc_modes[1]),B.shape, loc_modes[1])
            B = m2t(self.V_t@t2m(B, temp_mode),B.shape, temp_mode)
        else:
            B = tensorize(self.V_l@matricize(B, loc_modes), B.shape, loc_modes)
            B = m2t(self.V_t@t2m(B, temp_mode),B.shape, temp_mode)
        return B
            

    def __call__(self, Y, lda_1, lda_2, lda_l, lda_t, psis,
                 lr_modes, loc_modes, temp_mode,
                 maxit=100, err_tol=1e-4, verbose=1, **kwargs):
        n = Y.shape
        N = len(n)
        eps_pri = err_tol
        eps_dual = err_tol
        # Initialize Hyperparameters
        self.lda_1 = lda_1
        self.lda_2 = lda_2
        self.lda_l = lda_l
        self.lda_t = lda_t
        self.psis = psis
        rho = kwargs.get('rho', 1)
        rho1 = kwargs.get('rho_1', rho)
        rho2 = kwargs.get('rho_2', rho)
        self.rho1s.append(rho1)
        self.rho2s.append(rho2)

        # Initialize primal variables
        X = np.zeros(n)
        S = np.zeros(n)
        Xi = [np.zeros(n) for i in lr_modes]
        W = np.zeros(n)
        Wl = np.zeros(n)
        Wt = np.zeros(n)
        
        # Initialize dual variables
        Lda_i = [np.zeros(n) for i in lr_modes]
        Lda = np.zeros(n)
        Lda_t = np.zeros(n)
        Lda_l = np.zeros(n)
        
        # Initialize Filter coefficients for S update
        loc_coeffs = np.kron(self.spec_l**2, np.ones(n[temp_mode-1]))
        time_coeffs = np.kron(np.ones(len(self.spec_l)), self.spec_t**2)
        filter_coeffs = 1/(rho2*(loc_coeffs+time_coeffs)+(rho2+lda_2))
        filter_coeffs = filter_coeffs.reshape((len(self.spec_l)*len(self.spec_t),1))
        while self.it < maxit:
            # First block updates
            # {X, W, Wl, Wt}
            tstart= time()
            t = tstart
            X = (rho1*sum(Xi)-sum(Lda_i) + lda_2*(Y-S))/(N*rho1+lda_2)
            self.X = X
            self.times["X"].append(time()-t)
            t = time()
            W = soft_treshold(S-Lda/rho2, lda_1/rho2)
            self.times["W"].append(time()-t)
            t = time()
            Wl = soft_treshold(tensorize(self.L_l@matricize(S, loc_modes), n, loc_modes) - Lda_l/rho2,
                               lda_l/rho2)
            self.times["Wl"].append(time()-t)
            t = time()
            Wt = soft_treshold(m2t(self.L_t@t2m(S, temp_mode), n, temp_mode) - Lda_t/rho2,
                               lda_t/rho2)
            self.times["Wt"].append(time()-t)
            
            # Second block updates
            # {X1, ..., XN, S}
            # Xi update
            t = time()
            o = 0
            for i,m in enumerate(lr_modes):
                X_temp = Xi[i].copy()
                Xi[i], o_temp = soft_moden(X+Lda_i[i]/rho1,psis[i]/rho1,
                                    m)
                o += psis[i]*o_temp
            self.times["Xi"].append(time()-t)
            self.s1.append(norm(rho1* (sum(Xi[i]) - sum(X_temp)))) #
            # S update
            t = time()
            B = rho2*W+Lda
            B += tensorize(self.L_l.T@ matricize((rho2*Wl+Lda_l), loc_modes), n, loc_modes)
            B += m2t(self.L_t@t2m((rho2*Wt+Lda_t),temp_mode), n, temp_mode)
            B += lda_2*(Y-X)
            self.times['S_add'].append(time()-t)
            tt = time()
            # Graph Fourier Transform
            B = self._S_gft(B, loc_modes, temp_mode)
            self.times['S_gft'].append(time()-tt)
            # Filter
            tt = time()
            B = matricize(B, loc_modes+[temp_mode])
            B = tensorize(filter_coeffs*B, n, loc_modes+[temp_mode])
            self.times['S_mult'].append(time()-tt)
            # Inverse GFT
            B = self._S_inv_gft( B, loc_modes, temp_mode)
            Schange = B-S
            # Schange_l
            tt = time()
            self.s2.append( rho2*np.sqrt(norm(Schange)**2 +\
                            norm(tensorize(self.L_l.T@ matricize(Schange, loc_modes), n, loc_modes))**2 +\
                            norm(m2t(self.L_t@t2m((rho2*Wt+Lda_t),temp_mode), n, temp_mode) )**2)
                            ) 
            self.times['S_dual'].append(time()-tt)
            S = B
            self.S = S
            self.times["S"].append(time()-t)
            
            r1 = 0
            # Update dual variables
            for i in range(N):
                rtemp = X-Xi[i]
                Lda_i[i] += rho1*rtemp
                r1 += norm(rtemp)**2
            self.r1.append(np.sqrt(r1))

            r2 = 0
            rtemp = W-S
            Lda += rho2*rtemp
            r2 += norm(rtemp)**2
            
            rtemp = Wl - tensorize(self.L_l@ matricize(S, loc_modes), n, loc_modes)
            Lda_l += rho2*rtemp
            r2 += norm(rtemp)**2

            rtemp = Wt - m2t(self.L_t@t2m(S,temp_mode),n, temp_mode)
            Lda_t +=rho2*rtemp
            r2 += norm(rtemp)**2
            self.r2.append(np.sqrt(r2))
            self.times["it"].append(time()-tstart)
            self.it +=1
            if verbose and self.it>1:
                print(f"It-{self.it}: \t## {self.times['it'][-1]:.2f} sec. \t") ##  obj={obj[-1]:.5f} \t## del_obj = {obj[-1]-obj[-2]:.5f}")
                print(f"|r1|={self.r1[-1]:.5f} \t ## |s1|={self.s1[-1]:.5f} \t ## rho1={rho1:.5f}")
                print(f"|r2|={self.r2[-1]:.5f} \t ## |s2|={self.s2[-1]:.5f} \t ## rho2={rho2:.5f}")
            # Check for convergence
            if self.r1[-1]<eps_pri and self.s1[-1]<eps_dual and self.r2[-1]<eps_pri and self.s2[-1]<eps_dual:
                if verbose:
                    print("Converged!")
                break
            else:
                if self.rho1_upd !=-1:
                    if self.r1[-1]>self.rho1_mu*self.s1[-1]:
                        rho1=rho1*self.rho1_upd
                    elif self.s1[-1]>self.rho1_mu*self.r1[-1]:
                        rho1=rho1/self.rho1_upd
                if self.rho2_upd !=-1:
                    if self.r2[-1]>self.rho2_mu*self.s2[-1]:
                        rho2=rho2*self.rho2_upd
                        filter_coeffs = 1/(rho2*(loc_coeffs+time_coeffs)+(rho2+lda_2))
                        filter_coeffs = filter_coeffs.reshape((len(self.spec_l)*len(self.spec_t),1))
                    elif self.s2[-1]>self.rho2_mu*self.r2[-1]:
                        rho2=rho2/self.rho2_upd
                        filter_coeffs = 1/(rho2*(loc_coeffs+time_coeffs)+(rho2+lda_2))
                        filter_coeffs = filter_coeffs.reshape((len(self.spec_l)*len(self.spec_t),1))
                self.rho2s.append(rho2)
                self.rho1s.append(rho1)
        return X, S

    def plot_run(self):
        fig, axs = plt.subplots(2,2)
        axs[0,0].plot(self.r1,)
        axs[0,0].set_xlabel("k")
        axs[0,0].set_ylabel("||r1||")
        
        axs[0,1].plot(s)
        axs[0,1].set_xlabel("k")
        axs[0,1].set_ylabel("||s||")

        axs[1,0].plot(obj[1:])
        axs[1,0].set_xlabel("k")
        axs[1,0].set_ylabel("Objective")

        axs[1,1].plot(rhos)
        axs[1,1].set_xlabel("k")
        axs[1,1].set_ylabel("rho")

    def save_result(self, name):
        pass