import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm
from src.util.m2t import m2t
from src.util.t2m import t2m
from src.util.soft_treshold import soft_treshold
from src.util.soft_hosvd import soft_moden

def horpca_singleton(B, lda1=-1, rho=-1, verbose=1, err_tol=1e-5, maxit=100, step_size_growth=1.2, mu=30):
    """Solves the HoRPCA algorithm.

    Args:
        B (np.ma.masked_array): Observed tensor data
        lda1 (float): Hyperparameter of the l_1 norm.
        verbose (bool): Algorithm verbisity level. Defaults to True.
        max_it (int): Maximum number of iterations allowed for the algorithm. Defaults to 100.
        step_size_growth (float): Step size update coefficient of the ADMM algorithm. Defaults to 1.2.
        mu (float): Step size update treshold of the ADMM algorithm. Defaults to 10.
        err_tol (float): Convergence criteria for the algorithm. Defaults to 1e-5.

    Returns:
        results (dict): Dictionary of the results with algorithm parameters and
        algorithm logs. The dictionary structure is the following,
            Z (np.ndarray): Low-rank and smooth part.
            S (np.ndarray): Sparse part.
            lda1 (float): Sparsity (l_1 norm) hyperparameter of the algorithm.
            obj (np.array): Score log of the algorithm iterations.
            r (np.array): Logs of the frobenius norm of the residual part in the ADMM algortihm.
            s (np.array): Logs of the frobenius norm of the dual residual part in the ADMM algortihm.
            it (int): number of iterations.
            rho (np.array): Logs of the step size in each iteration of the ADMM algorithm.
    """
    n = B.shape
    if lda1 ==-1:
        lda1 = 1/np.sqrt(np.max(n))
    
    try:
        mask = B.mask
        if mask.size ==1:
            B = np.ma.masked_array(B, mask=np.zeros(B.shape))
        result = _horpca_singleton_m(B, lda1, rho, verbose, err_tol, maxit, step_size_growth, mu)
    except AttributeError:
        B = np.ma.masked_array(B,np.zeros(B.shape))
        result = _horpca_singleton_m(B, lda1, rho, verbose, err_tol, maxit, step_size_growth, mu)
    return result



def _horpca_singleton_m(B, lda1, rho, verbose, err_tol, maxit, step_size_growth, mu):
    n = B.shape; N = len(n)
    obs_idx = ~B.mask
    unobs_idx = B.mask
    if rho==-1:
        rho=0.001

    # init 
    Xi = [np.zeros(n, dtype=np.float64) for _ in range(N)]; Ldai = [np.zeros(n, dtype=np.float64) for _ in range(N)]
    E = np.zeros(n, dtype=np.float64); Lda = np.zeros(n, dtype=np.float64)
    Z = np.zeros(n, dtype=np.float64); Zold = np.zeros(n, dtype=np.float64)
    
    it = 0
    r =[]; s=[] # Primal and Dual residual norms for each iteration.
    obj =[np.inf]; # Objective function and lagrangian function values for each iteration.
    rhos = [rho]; # 
    while it <maxit:
        nuclear_norm = 0
        for i in range(N):
            Xi[i],nn  = soft_moden(Z-Ldai[i]/rho, 1/rho, i+1)
            nuclear_norm += np.abs(nn)  

        
        E[obs_idx] = soft_treshold(B[obs_idx]-Z[obs_idx]-Lda[obs_idx]/rho, lda1/rho)
        
        obj.append( nuclear_norm + lda1*np.sum(np.abs(E[obs_idx])) )
        
        Zold = Z.copy()
        Xi_ = sum(Xi); Ldai_ = sum(Ldai)
        Z[obs_idx] = ( Xi_[obs_idx] + Ldai_[obs_idx]/rho +  B[obs_idx] - E[obs_idx] - Lda[obs_idx]/rho)/(N+1)
        Z[unobs_idx] = (Xi_[unobs_idx] + Ldai_[unobs_idx]/rho )/N
        
        
        
        # Update dual variables and calculate primal and dual residuals:
        pri_residual_norm_k = 0
        for i in range(N):
            ri = Xi[i]-Z
            Ldai[i] = Ldai[i] + rho*(ri)
            pri_residual_norm_k += np.linalg.norm(ri)
        
        ri = E[obs_idx] + Z[obs_idx]- B[obs_idx]
        Lda[obs_idx] = Lda[obs_idx] + rho*(ri)
        pri_residual_norm_k += np.linalg.norm(ri)
        s_k = Z-Zold
        s.append( rho*( np.sqrt(N+1)*np.linalg.norm(s_k[obs_idx]) + np.sqrt(N)*np.linalg.norm(s_k[unobs_idx])))
        r.append(pri_residual_norm_k)
        
        it +=1
        # Check convergence
        eps_pri = err_tol*(N+1)*norm(Z)
        eps_dual = err_tol*sum([norm(y) for y in Ldai]+[norm(Lda)])
        if verbose:
            print(f"It-{it}: obj={obj[-1]:.4f} \t ## del_obj = {obj[-1]-obj[-2]:.4f} \t## |r|={r[-1]:.5f} \t ## |s|={s[-1]:.5f} \t ## rho={rho:.4f}")

        if r[-1]<eps_pri and s[-1]<eps_dual:
            if verbose:
                print("Converged!")
            break
        else:
            if step_size_growth !=-1:
                if r[-1]>mu*s[-1]:
                    rho=rho*step_size_growth
                    rhos.append(rho)
                elif s[-1]>mu*r[-1]:
                    rho=rho/step_size_growth
                    rhos.append(rho)
                else:
                    rhos.append(rho)

    results = {'Y':Z,
                'E':E,
                'lda1':lda1,
                'obj':np.array(obj),
                'r':np.array(r),
                's':np.array(s),
                'it':it,
                'rho':np.array(rhos)}
    return results

def plot_alg(r,s,obj, rhos):
    """ Plots the algorithm log in 2x2 subplots."""
    fig, axs = plt.subplots(2,2)
    axs[0,0].plot(r)
    axs[0,0].set_xlabel("k")
    axs[0,0].set_ylabel("||r||")
    
    axs[0,1].plot(s)
    axs[0,1].set_xlabel("k")
    axs[0,1].set_ylabel("||s||")

    axs[1,0].plot(obj[1:])
    axs[1,0].set_xlabel("k")
    axs[1,0].set_ylabel("Objective")

    axs[1,1].plot(rhos)
    axs[1,1].set_xlabel("k")
    axs[1,1].set_ylabel("rho")
    



