import numpy as np
from numpy.linalg import norm
from src.util.m2t import m2t
from src.util.t2m import t2m
from src.util.soft_treshold import soft_treshold
from src.util.soft_hosvd import soft_moden
import matplotlib.pyplot as plt


def grhorpca(B,Ls, **kwargs):
    """Solve the Graph Regularized Higher-order Principal Component Analysis algorithm.

    Args:
        B (np.ma.masked_array): Observed tensor data
        Ls (list): List of laplacian matrices of the graph structures
        modes (list): List of modes to apply the graph laplacian for the smoothness regularization.
        defaults to [1,2,3,...,len(Ls)]
        lda1 (float): Hyperparameter of the l_1 norm.
        lda_nucs (list): Hyperparemeter of the nuclear norm term.
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
    M = len(modes)
    N = len(n)
    rho = kwargs.get('rho',0.1)
    lda1 = kwargs.get('lda1',1)
    lda_nucs = kwargs.get('lda_nucs', [1 for _ in range(N)])
    alpha = kwargs.get('alpha', [1 for _ in range(M)])
    verbose = kwargs.get('verbose',1)
    max_it = kwargs.get('max_it',250)
    rho_upd = kwargs.get('rho_upd', 1.2)
    rho_mu = kwargs.get('rho_mu', 10)
    err_tol = kwargs.get('err_tol',1e-5)

    assert len(alpha) == M
    obs_idx = ~B.mask
    unobs_idx = B.mask

    # Initialize variables:
    X_i = [np.zeros(n) for _ in range(M)]
    Y_i = [np.zeros(n) for _ in range(N)]
    S = np.zeros(n) 
    Z =np.zeros(n); Zold =np.zeros(n) 
    ri = np.zeros(n)
    
    Lda_1i = [np.zeros(n) for _ in range(M)]
    Lda_2i = [np.zeros(n) for _ in range(N)]
    Lda_s = np.zeros(n)

    # Inverse terms
    Inv = []
    for i,m in enumerate(modes):
        Inv.append( np.linalg.inv(alpha[i]*Ls[i] + rho*np.eye(n[m-1])))

    it = 0
    r = []; s=[] # Primal and dual residuals norms for each iteration
    obj =[np.inf]
    rhos =[rho] # To record the step size (penalty parameter)
    while it<max_it:
        o = 0
        for i in range(M):
            X_i[i] = Inv[i]@ t2m( (rho*Z+Lda_1i[i]), modes[i] )
            o+= alpha[i]* np.trace(X_i[i].T@Ls[i]@X_i[i])
            X_i[i] = m2t(X_i[i],n, modes[i])
        
        for i in range(N):
            #print(f"N={N}, i={i}, ")
            Y_i[i], nuc_norm = soft_moden(Z+Lda_2i[i]/rho, lda_nucs[i]/rho, i+1)
            o += nuc_norm

        ss = soft_treshold(B-Z-Lda_s,lda1/rho)
        S[obs_idx] = ss[obs_idx]
        o += np.sum(np.abs(S))

        Zold = Z.copy()
        Xbar = sum([Xi for Xi in X_i]+[-Lda/rho for Lda in Lda_1i]+[Yi for Yi in Y_i]+[-Lda/rho for Lda in Lda_2i])
        Z[obs_idx] = (Xbar[obs_idx]+ B[obs_idx] - S[obs_idx] - Lda_s[obs_idx]/rho)/(M+N+1)
        Z[unobs_idx] = Xbar[unobs_idx]/(M+N)

        rtmp = 0
        for i in range(M):
            ri = Z - X_i[i]
            Lda_1i[i]=Lda_1i[i]+rho*ri
            rtmp=+norm(ri)**2
        
        for i in range(N):
            ri = Z - Y_i[i]
            Lda_2i[i]=Lda_2i[i]+rho*ri
            rtmp=+norm(ri)**2
        
        
        ri.fill(0)
        ri[obs_idx] = S[obs_idx] + Z[obs_idx] -B[obs_idx]
        Lda_s[obs_idx] =Lda_s[obs_idx] +rho*ri[obs_idx]
        rtmp += norm(ri)**2
        
        r.append(np.sqrt(rtmp))
        stmp = (N+M)*norm(Z-Zold)**2
        stmp += norm(Z[obs_idx]-Zold[obs_idx])**2
        s.append(np.sqrt(stmp))
        obj.append(o)
        
        it +=1
        # Check convergence
        eps_pri = err_tol*(M+N+1)*norm(Z)
        eps_dual = err_tol*np.sqrt(sum([norm(y)**2 for y in Lda_1i]+[norm(Lda_s)**2]+[norm(y)**2 for y in Lda_2i]))
        if verbose:
            print(f"It-{it}:\t## |r|={r[-1]:.5f} \t ## |s|={s[-1]:.5f} \t ## rho={rho:.4f} obj={obj[-1]:.4f} \t ## del_obj = {obj[-1]-obj[-2]:.4f} ")
    
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


def plot_alg(r,s,obj, rhos):
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