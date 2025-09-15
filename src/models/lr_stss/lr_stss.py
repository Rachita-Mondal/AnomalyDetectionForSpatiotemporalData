import numpy as np
from numpy.linalg import norm, inv
from src.util.soft_hosvd import soft_moden
from src.util.t2m import t2m
from src.util.m2t import m2t
from src.util.soft_treshold import soft_treshold
from src.util.soft_hosvd import soft_moden
from scipy.sparse import csr_matrix

def lr_stss(Y, A, temp_m, spat_m, **kwargs):
    """Low-rank separation algorithm from spatio-temporally smooth sparse.

    WRITE ELABORATE DESCRIPTION

    Args:
        Y (np.ndarray): Observed data/signal/tensor
        A (np.array): Adjecancy matrix of the neighbour locations
        temp_m (int): temporal mode of the tensor
        spat_m (int, list of ints): spatial mode(s) of the tensor
        lda_1 (float): Hyperparameter of the sparsity. Defaults to 1
        lda_2 (float): Fidelity term that regularizes deviation from original observation
        lda_loc (float): Hyperparameter of the local smoothness. Defaults to 1
        lda_t (float): Hyperparameter of the temporal smoothness. Defaults to 1
        psis (float, list of floats): Hyperparameters/weights of the nuclear norms. Defaults to 1
        verbose (int): Algorithm verbisity level. Defaults to 1.
        max_it (int): Maximum number of iterations allowed for the algorithm. Defaults to 250
        rho_upd (float): Step size update coefficient of the ADMM algorithm. Defaults to 1.2
        rho_mu (float): Step size update treshold of the ADMM algorithm. Defaults to 10
        err_tol (float): Convergence criteria for the algorithm. Defaults to 1e-5
    """
    sz = Y.shape
    lr_modes = kwargs.get('lr_modes', [i+1 for i in range(len(sz))])
    N = len(lr_modes)
    lda_loc = kwargs.get('lda_loc',1)
    lda_1 = kwargs.get('lda1',1)
    lda_2 = kwargs.get('lda2',100)
    lda_t = kwargs.get('lda_t',100)
    psis = kwargs.get('psis', tuple([1 for _ in range(N)]))
    if isinstance(psis,float):
        psis = tuple([psis]*N)
    
    verbose = kwargs.get('verbose',1)
    max_it = kwargs.get('max_it',250)
    max_it2 = kwargs.get('max_it2',30)
    rho = kwargs.get('rho',1)
    rho2 = kwargs.get('rho2',1)
    rho_upd = kwargs.get('rho_upd', 1.2)
    rho_mu = kwargs.get('rho_mu', 10)
    err_tol = kwargs.get('err_tol',1e-5)

    # Initialize variables
    X_i = [np.zeros(sz) for _ in range(N)]
    W_t = np.zeros(sz)
    W_loc = np.zeros(sz)
    W = np.zeros(sz)
    S = np.zeros(sz)
    X = np.zeros(sz)
    obs_idx = ~Y.mask
    unobs_idx = Y.mask

    # Initialize dual variables
    Lda_i = [np.zeros(sz) for _ in range(N)]    # X = X_i
    Lda_t = np.zeros(sz)                        # W_t = D xt S
    Lda_loc = np.zeros(sz)                      # W_loc = (I-A)xl S
    Lda = np.zeros(sz)                          # W = S

    # Initialize Delt matrix
    Delt = np.eye(sz[temp_m-1]) + np.diag(-np.ones(sz[temp_m-1]-1), 1)
    Delt[-1,0]=-1
    Delt = csr_matrix(Delt)
    I_A = csr_matrix(np.eye(A.shape[0])-A)
    inv_T = inv(rho2*np.eye(sz[temp_m-1])+ rho*Delt.T@Delt)
    inv_L = inv(rho2*np.eye(sz[spat_m-1])+ rho*I_A.T@I_A)

    it = 0
    obj = [np.inf]
    r = []  # Norm of primal residual in each iteration
    s = []  # Norm of dual residual
    rhos = [rho] # Step size
    while it<max_it:
        # {X, Wt, Wl, W} Block updates
        tempX = rho*sum(X_i)-sum(Lda_i)
        X[obs_idx] = (lda_2*(Y[obs_idx]-S[obs_idx]) + tempX[obs_idx])/(lda_2+N*rho)
        X[unobs_idx] = tempX[unobs_idx]/(N*rho) # X update
        ## Wt update
        W_t = soft_treshold( m2t(Delt@t2m(S,temp_m),sz,temp_m)-Lda_t/rho,
                             lda_t/rho)
        ## Wl update
        W_loc = soft_treshold( m2t((I_A)@t2m(S,spat_m),sz,spat_m)-Lda_loc/rho,
                              lda_loc/rho)
        ## W update
        W = soft_treshold( S-Lda/rho,
                          lda_1/rho)
        
        # {S, Xi} Block updates
        #        ###### Solve for S ####
        Z = S
        Ss = [np.zeros(sz) for _ in range(4)]
        Gamma = [np.zeros(sz) for _ in range(4)]
        it2 =0
        obj2 = [np.inf]
        r2 = []
        s2 = []
        while it2<max_it2:
            Ss[0][obs_idx] = (lda_2*(Y[obs_idx]-X[obs_idx])+ \
                              rho2*(Z[obs_idx]+Gamma[0][obs_idx]/rho2))/(lda_2+rho2) 
            Ss[0][unobs_idx] = (rho2*(Z[unobs_idx]+Gamma[0][unobs_idx]/rho2))/(lda_2+rho2)

            Ss[1] = m2t(np.asarray(inv_T@\
                            (rho2*t2m(Z+Gamma[1]/rho2,temp_m)+
                            rho*Delt.T@t2m(W_t+Lda_t/rho,temp_m)
                            )),
                        sz, temp_m)

            Ss[2] = m2t(np.asarray(inv_L@\
                            (rho2*t2m(Z+Gamma[2]/rho2,spat_m)+
                             rho*I_A.T@t2m(W_loc+Lda_loc/rho,spat_m)))
                        ,sz, spat_m)
        
            Ss[3] = (rho2*(Z+Gamma[3]/rho2)+rho*(W+Lda/rho))/(rho2+rho)

            Zkp1 = sum(Ss)/4
            s2.append(norm(Zkp1-Z))
            Z = Zkp1.copy()
            r2_norm = 0
            for i in range(4):
                r2_temp = Z-Ss[i]
                r2_norm += norm(r2_temp)**2
                Gamma[i] = Gamma[i] +rho2*(r2_temp)
            r2.append(np.sqrt(r2_norm))
            # obj2.append(0.5(lda2*norm(X+Z-Y)**2+rho*))
            obj2.append(np.inf)
            if verbose>1:
                print(f"It:{it}\tS-ADMM it:{it2}\t## |r2|={r2[-1]:.5f} \t ## |s2|={s2[-1]:.5f}"+
                      f"\t ## rho2={rho2:.4f}")##\t ## obj={obj2[-1]:.4f} \t ## del_obj = {obj2[-1]-obj2[-2]:.4f}")
        
            it2 +=1
            if r2[-1]<err_tol and s2[-1]<err_tol:
                if verbose>1:
                    print(f"It:{it}\tS-ADMM it:{it2} S ADMM converged.")
                break
        
        #        ###### Solve for S ####
        s_norm = 0
        s_norm += norm(S-Z)**2
        S = Z.copy()
        ## Update Xi
        o = 0
        for i,m in enumerate(lr_modes):
            X_temp = X_i[i].copy()
            X_i[i], o_temp = soft_moden(X+Lda_i[i]/rho,psis[i]/rho,
                                m)
            o += psis[i]*o_temp
            s_temp = X_i[i] - X_temp
            s_norm += norm(s_temp)
        s.append(s_norm)
        r_norm = 0
        # Dual variable updates and primal residual
        for i in range(N):
            r_temp = X-X_i[i]
            Lda_i[i] = Lda_i[i] + rho*(r_temp)
            r_norm += norm(r_temp)**2
        
        r_temp = W_t- m2t(Delt@t2m(S,temp_m),sz,temp_m)
        Lda_t += rho*(r_temp)
        r_norm += norm(r_temp)**2
        
        r_temp = W_loc - m2t((I_A)@t2m(S,spat_m),sz,spat_m)
        Lda_loc += rho*r_temp
        r_norm += norm(r_temp)**2
        
        r_temp = W-S
        Lda += rho*r_temp
        r_norm += norm(r_temp)**2
        r.append(np.sqrt(r_norm))
        
        obj.append(o + lda_t*norm((Delt@t2m(S,temp_m)).ravel(),1)+
                   lda_loc*norm((I_A@t2m(S,spat_m)).ravel(),1)+
                   lda_1*norm(S.ravel(),1)+
                   0.5*lda_2*norm(X+S-Y)**2)
        it +=1
        # Check convergence
        eps_pri = err_tol   # Change later
        eps_dual = err_tol  # Change later
        if verbose and it>1:
            print(f"It-{it}:\t## |r|={r[-1]:.5f} \t ## |s|={s[-1]:.5f} \t ## rho={rho:.4f} obj={obj[-1]:.4f} \t ## del_obj = {obj[-1]-obj[-2]:.4f} ")
        
        if r[-1]<eps_pri and s[-1]<eps_dual:
            if verbose:
                print("Converged!")
            break
        # else: # Update step size if needed
        #     if rho_upd !=-1:
        #         if r[-1]>rho_mu*s[-1]:
        #             rho=rho*rho_upd
        #             rhos.append(rho)
        #             inv_T = inv(rho2*np.eye(sz[temp_m-1])+ rho*Delt.T@Delt)
        #             inv_L = inv(rho2*np.eye(sz[spat_m-1])+ rho*I_A.T@I_A)
        #         elif s[-1]>rho_mu*r[-1]:
        #             rho=rho/rho_upd
        #             rhos.append(rho)
        #             inv_T = inv(rho2*np.eye(sz[temp_m-1])+ rho*Delt.T@Delt)
        #             inv_L = inv(rho2*np.eye(sz[spat_m-1])+ rho*I_A.T@I_A)
        #         else:
        #             rhos.append(rho)

    results = {'X':X,
                'S':S,
                'obj':np.array(obj),
                'r':np.array(r),
                's':np.array(s),
                'r2':np.array(r2),
                's2':np.array(s2),
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