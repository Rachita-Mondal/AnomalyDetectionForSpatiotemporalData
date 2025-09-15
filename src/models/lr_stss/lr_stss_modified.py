import numpy as np
from numpy.linalg import norm, inv
from scipy.sparse import csr_matrix
from scipy.linalg import eigh
from sklearn.metrics import roc_auc_score
from src.proximal_ops.soft_hosvd import soft_moden
from src.multilinear_ops.t2m import t2m
from src.multilinear_ops.m2t import m2t
from src.proximal_ops.soft_treshold import soft_treshold
from src.proximal_ops.soft_hosvd import soft_moden
from src.multilinear_ops.list_kronecker import list_kronecker


def lr_stss_modified(Y, A, temp_m, spat_m, **kwargs):
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
    lda_2 = kwargs.get('lda2',1)
    lda_t = kwargs.get('lda_t',100)
    psis = kwargs.get('psis', tuple([1 for _ in range(N)]))
    L = kwargs.get('L', None)
    if isinstance(psis,float):
        psis = tuple([psis]*N)
    verbose = kwargs.get('verbose',1)
    max_it = kwargs.get('max_it',250)
    rho = kwargs.get('rho',1)
    err_tol = kwargs.get('err_tol',1e-5)

    anomaly_mask_gt = kwargs.get('anomaly_mask_gt', None)
    detect_events = kwargs.get('detect_events', None)
    detected_events = {'3':[], '2':[], '1':[], '0.7':[], 'sum':[]}
    ratios = np.array([0.014, 0.07, 0.14, 0.3, 0.7, 1, 2, 3])/100
    auc = []

    # Initialize variables
    X_i = [np.zeros(sz) for _ in range(N)]
    W_t = np.zeros(sz)
    W_loc = np.zeros(sz)
    W = np.zeros(sz)
    S = np.zeros(sz)
    X = np.zeros(sz)
    # obs_idx = ~Y.mask
    # unobs_idx = Y.mask

    # Initialize dual variables
    Lda_i = [np.zeros(sz) for _ in range(N)]    # X = X_i
    Lda_t = np.zeros(sz)                        # W_t = D xt S
    Lda_loc = np.zeros(sz)                      # W_loc = (I-A)xl S
    Lda = np.zeros(sz)                          # W = S

    # Initialize Delt matrix (Temporal first order difference operator)
    Delt = np.eye(sz[temp_m-1]) + np.diag(-np.ones(sz[temp_m-1]-1), 1)
    Delt[-1,0]=-1
    Delt = csr_matrix(Delt)
    if not isinstance(L, type(None)):
        I_A = L
    else:
        I_A = csr_matrix(np.eye(A.shape[0])-A)

    rho_hat = (rho+lda_2)/rho
    # S update inverses
    w_l, V_l = eigh(I_A.T@I_A+np.eye(sz[spat_m-1])*0.5*rho_hat)
    w_t, V_t = eigh(Delt.T@Delt+np.eye(sz[temp_m-1])*0.5*rho_hat)
    li = [np.ones(s) for s in sz]
    li[temp_m-1] = w_t
    I_wt = list_kronecker(li)
    li[temp_m-1] = np.ones(sz[temp_m-1])
    li[spat_m-1] = w_l
    I_wl = list_kronecker(li)
    I_w = 1/(I_wt + I_wl)
    I_w = I_w.reshape(sz)


    it = 0
    obj = [np.inf]
    r = []          # Norm of primal residual in each iteration
    s = []          # Norm of dual residual
    rhos = [rho]    # Step size
    while it<max_it:
        # {X, Wt, Wl, W} Block updates
        tempX = rho*sum(X_i)-sum(Lda_i)
        X = (lda_2*(Y-S) + tempX)/(lda_2+N*rho)
        # X[obs_idx] = (lda_2*(Y[obs_idx]-S[obs_idx]) + tempX[obs_idx])/(lda_2+N*rho)
        # X[unobs_idx] = tempX[unobs_idx]/(N*rho) # X update
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
        B_t = m2t( Delt.T@ t2m(W_t-Lda_t/rho, temp_m), sz, temp_m)
        B_l = m2t( I_A.T@ t2m(W_loc-Lda_loc/rho, spat_m), sz, spat_m)
        B_1 = W - Lda/rho
        B = Y-X
        B_tilda = rho*(B_t+B_l+B_1) + lda_2*B
        
        B_tilda = I_w* m2t( # Take Graph Fourier Transform of B_tilda and filter it
            V_l.T @ t2m(m2t( 
                            V_t.T @ t2m(B_tilda/rho, temp_m), sz, temp_m),
                        ), sz, spat_m)
        Z = m2t(V_l@ t2m(m2t(V_t@t2m(
                                B_tilda, temp_m), sz, temp_m),
                                spat_m), sz, spat_m)
        
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

        if not isinstance(detect_events, type(None)):
            num_detected_events = np.array([detect_events(np.abs(S), r) for r in ratios])
            detected_events['3'].append(num_detected_events[-1])
            detected_events['2'].append(num_detected_events[-2])
            detected_events['1'].append(num_detected_events[-3])
            detected_events['0.7'].append(num_detected_events[-4])
            detected_events['sum'].append(sum(num_detected_events))
        
        if not isinstance(anomaly_mask_gt, type(None)):
            auc.append(roc_auc_score(anomaly_mask_gt.ravel(),np.abs(S).ravel()))
        
        if r[-1]<eps_pri and s[-1]<eps_dual:
            if verbose:
                print("Converged!")
            break

    results = {'X':X,
                'S':S,
                'obj':np.array(obj),
                'r':np.array(r),
                's':np.array(s),
                'it':it,
                'rho':np.array(rhos),
                'detected_events': detected_events,
                'auc': np.array(auc)}
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