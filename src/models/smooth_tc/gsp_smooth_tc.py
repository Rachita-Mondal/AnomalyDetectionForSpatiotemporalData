import numpy as np
from numpy.linalg import norm
from src.util.m2t import m2t
from src.util.t2m import t2m
from src.util.soft_treshold import soft_treshold
import matplotlib.pyplot as plt
from numba import njit
#from scipy.linalg import cho_factor, cho_solve
#@njit
def robust_smooth_tc(B, Ls, **kwargs):
    """Solves the Smooth and Sparse Seperation (SSS) algorithm.

    Args:
        B (np.ma.masked_array): Observed tensor data
        Ls (list): List of laplacian matrices of the graph structures
        modes (list): List of modes to apply the graph laplacian for the smoothness regularization.
        defaults to [1,2,3,...,len(Ls)]
        lda1 (float): Hyperparameter of the l_1 norm.
        alpha (list): Hyperparameter of the Smoothness term.
        verbose (bool): Algorithm verbisity level. Defaults to True.
        max_it (int): Maximum number of iterations allowed for the algorithm. Defaults to 250
        rho_upd (float): Step size update coefficient of the ADMM algorithm. Defaults to 1.2
        rho_mu (float): Step size update treshold of the ADMM algorithm. Defaults to 10
        err_tol (float): Convergence criteria for the algorithm. Defaults to 1e-5

    Returns:
        results (dict): Dictionary of the results with algorithm parameters and
        algorithm logs. The dictionary structure is the following,
            Z (np.ndarray): Low-rank and smooth part
            S (np.ndarray): Sparse part
            lda1 (float): Sparsity (l_1 norm) hyperparameter of the algorithm
            alpha (float): Smoothness hyperparameter of the algorithm. 
            obj (np.array): Score log of the algorithm iterations. 
            r (np.array): Logs of the frobenius norm of the residual part in the ADMM algortihm
            s (np.array): Logs of the frobenius norm of the dual residual part in the ADMM algortihm
            it (int): number of iterations
            rho (np.array): Logs of the step size in each iteration of the ADMM algorithm.
    """
    n = B.shape
    modes = kwargs.get('modes', [i+1 for i in range(len(Ls))])
    N = len(modes)
    rho = kwargs.get('rho',0.1)
    lda1 = kwargs.get('lda1',1)
    alpha = kwargs.get('alpha', [1 for _ in range(N)])
    verbose = kwargs.get('verbose',1)
    max_it = kwargs.get('max_it',100)
    rho_upd = kwargs.get('rho_upd', 1.2)
    rho_mu = kwargs.get('rho_mu', 30)
    err_tol = kwargs.get('err_tol',1e-5)
    assert len(alpha) == N
    obs_idx = ~B.mask
    unobs_idx = B.mask

    # Initialize variables:
    X_i = [np.zeros(n) for _ in range(N)]
    S = np.zeros(n); 
    Z =np.zeros(n)
    ri = [np.zeros(n) for _ in range(N)]
    Lda_i = [np.zeros(n) for _ in range(N)]
    Lda_s = np.zeros(n)
    

    # Inverse terms
    Inv = []
    for i,m in enumerate(modes):
        Inv.append( np.linalg.inv(alpha[i]*Ls[i] + rho*np.eye(n[m-1])) )

    it = 0
    r = []; s=[] # Primal and dual residuals norms for each iteration
    obj =[np.inf]
    rhos =[rho] # To record the step size (penalty parameter)
    while it<max_it:
        for i,m in enumerate(modes):
            X_i[i] = m2t(
                        Inv[i]@ t2m( (rho*Z-Lda_i[i]) ,m)
                        ,n, m)
        
        S[obs_idx] = soft_treshold(B[obs_idx] - Z[obs_idx] - Lda_s[obs_idx]/rho, lda1/rho )
        
        Xbar = sum(X_i)+sum(Lda_i)/rho
        Zold = Z.copy()
        Z[obs_idx] = (Xbar[obs_idx] + B[obs_idx] - S[obs_idx] - Lda_s[obs_idx]/rho)/(N+1)
        Z[unobs_idx] = Xbar[unobs_idx]/N

        # Update dual variables and calculate primal and dual residuals
        pri_residual_norm_k = 0
        for i in range(N):
            ri[i] = X_i[i]-Z
            Lda_i[i]=Lda_i[i] + rho*(ri[i])
            pri_residual_norm_k += norm(ri[i])**2
        rs = S[obs_idx]+Z[obs_idx] -B[obs_idx]
        Lda_s[obs_idx] = Lda_s[obs_idx] + rho*rs
        pri_residual_norm_k += norm(rs)**2
        s.append(rho*np.sqrt( (N+1)*norm(Z[obs_idx]-Zold[obs_idx])**2 + N*norm(Z[unobs_idx] - Zold[unobs_idx])**2 ) )
        r.append(np.sqrt(pri_residual_norm_k))
        

        
        it +=1
        # Check convergence
        eps_pri = err_tol*(N+1)*norm(Z)
        eps_dual = err_tol*sum([norm(y) for y in Lda_i]+[norm(Lda_s)])
        #if verbose:
        #    print(f"It-{it}:\t## |r|={r[-1]:.5f} \t ## |s|={s[-1]:.5f} \t ## rho={rho:.4f}")
         #  obj={obj[-1]:.4f} \t ## del_obj = {obj[-1]-obj[-2]:.4f} 
        if r[-1]<eps_pri and s[-1]<eps_dual:
            if verbose:
                print("Converged!")
            break
        else: # Update step size if needed
            if rho_upd !=-1:
                if r[-1]>rho_mu*s[-1]:
                    rho=rho*rho_upd
                    rhos.append(rho)
                    for i,m in enumerate(modes):
                        Inv[i] = np.linalg.inv(alpha[i]*Ls[i] + rho*np.eye(n[m-1]))
                elif s[-1]>rho_mu*r[-1]:
                    rho=rho/rho_upd
                    rhos.append(rho)
                    for i,m in enumerate(modes):
                        Inv[i] = np.linalg.inv(alpha[i]*Ls[i] + rho*np.eye(n[m-1]))
                else:
                    rhos.append(rho)

    results = {'Z':Z,
                'S':S,
                'lda1':lda1,
                'alpha':alpha,
                'obj':np.array(obj),
                'r':np.array(r),
                's':np.array(s),
                'it':it,
                'rho':np.array(rhos)}
    return results

# def smooth_tc(B, Ls, **kwargs):
#     n = B.shape
#     modes = kwargs.get('modes', [i+1 for i in range(len(Ls))])
#     N = len(modes)
#     rho = kwargs.get('rho',0.1)
#     alpha = kwargs.get('alpha', [1 for _ in range(N)])
#     verbose = kwargs.get('verbose',1)
#     max_it = kwargs.get('max_it',100)
#     rho_upd = kwargs.get('rho_upd', 1.1)
#     rho_mu = kwargs.get('rho_mu', 10)
#     err_tol = kwargs.get('err_tol',1e-5)
#     assert len(alpha) == N
#     obs_idx = ~B.mask
#     unobs_idx = B.mask

#     # Initialize variables:
#     X_i = [np.zeros(n) for _ in range(N)]
#     Z =np.zeros(n)
#     ri = [np.zeros(n) for _ in range(N)]
#     Lda_i = [np.zeros(n) for _ in range(N)]
#     Lda_z = np.zeros(n)
    

#     # Inverse terms
#     Inv = []
#     for i,m in enumerate(modes):
#         Inv.append( np.linalg.inv(alpha[i]*Ls[i] + rho*np.eye(n[m-1])) )


#     it = 0
#     r = []; s=[] # Primal and dual residuals norms for each iteration
#     obj =[np.inf]
#     rhos =[rho] # To record the step size (penalty parameter)
#     while it<max_it:
#         for i,m in enumerate(modes):
#             X_i[i] = m2t(
#                         Inv[i]@ t2m( (rho*Z-Lda_i[i]) ,m)
#                         ,n, m)

#         Xbar = sum(X_i)+sum(Lda_i)/rho
#         Zold = Z.copy()
#         Z[obs_idx] = (Xbar[obs_idx] + B[obs_idx] - Lda_z[obs_idx]/rho)/(N+1)
#         Z[unobs_idx] = Xbar[unobs_idx]/N

#         # Update dual variables and calculate primal and dual residuals
#         pri_residual_norm_k = 0
#         for i in range(N):
#             ri[i] = X_i[i]-Z
#             Lda_i[i]=Lda_i[i] + rho*(ri)
#             pri_residual_norm_k += norm(ri[i])**2




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