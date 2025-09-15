import numpy as np
from numpy.linalg import norm, inv
from scipy.sparse import csr_matrix
from scipy.linalg import eigh
# from src.algos.admm_base_class import TwoBlockADMMBase
from src.multilinear_ops.t2m import t2m
from src.multilinear_ops.m2t import m2t
from src.proximal_ops.soft_treshold import soft_treshold
from src.proximal_ops.soft_hosvd import soft_moden

def lr_sts_hard(Y, temp_m, **kwargs):
    """Low-rank, Sparse and Temporally smooth term separation with hard constraint.

    Args:
        Y (np.ndarray): Observed data/signal/tensor
        temp_m (int): temporal mode of the tensor
        psis (float, list of floats): Weights of the nuclear norms. Defaults to 1
        lda_1 (float): Hyperparameter of the sparsity. Defaults to 1
        lda_t (float): Hyperparameter of the temporal smoothness. Defaults to 1
        verbose (int): Algorithm verbisity level. Defaults to 1.
        max_it (int): Maximum number of iterations allowed for the algorithm. Defaults to 250
        rho_upd (float): Step size update coefficient of the ADMM algorithm. Defaults to 1.2
        rho_mu (float): Step size update treshold of the ADMM algorithm. Defaults to 10
        err_tol (float): Convergence criteria for the algorithm. Defaults to 1e-5
    Returns:
        X (np.ndarray): Seperated low-rank and smooth part
        S (np.ndarray): Seperated sparse part 
    """
    # Parse keyword arguments
    sz = Y.shape
    lr_modes = kwargs.get('lr_modes', [i+1 for i in range(len(sz))])
    N = len(lr_modes)
    lda_1 = kwargs.get('lda1',1)
    lda_t = kwargs.get('lda_t',100)
    psis = kwargs.get('psis', tuple([1 for _ in range(N)]))
    detect_events = kwargs.get('detect_events', None)
    detected_events = {'3':[], '2':[], '1':[], '0.7':[], 'sum':[]}
    ratios = np.array([0.014, 0.07, 0.14, 0.3, 0.7, 1, 2, 3])/100
    if isinstance(psis,float):
        psis = tuple([psis]*N)
    verbose = kwargs.get('verbose',1)
    max_it = kwargs.get('max_it',250)
    rho = kwargs.get('rho',1)
    err_tol = kwargs.get('err_tol',1e-5)
    eps_pri = err_tol   # Change later
    eps_dual = err_tol  # Change later
    # Initialize variables
    X_i = [np.zeros(sz) for _ in range(N)]
    W_t = np.zeros(sz)
    W = np.zeros(sz)
    S = np.zeros(sz)
    X = np.zeros(sz)

    # Initialize dual variables
    Lda_i = [np.zeros(sz) for _ in range(N)]    # X = X_i
    Lda_t = np.zeros(sz)                        # W_t = D xt S
    Lda = np.zeros(sz)                          # W = S
    Lda_h = np.zeros(sz)                        # X + S = Y

    # Initialize Delt matrix (Temporal first order difference operator)
    Delt = np.eye(sz[temp_m-1]) + np.diag(-np.ones(sz[temp_m-1]-1), 1)
    Delt[-1,0]=-1
    S_inv = inv(np.eye(sz[temp_m-1])*2 + Delt.T@Delt)

    it = 0
    obj = [np.inf]
    r = []  # Norm of primal residual in each iteration
    s = []  # Norm of dual residual
    rhos = [rho] # Step size
    while it<max_it:
        # {X, Wt, Wl, W} Block updates
        tempX = sum(X_i)-sum(Lda_i)/rho
        X = (tempX + Y-S-Lda_h/rho)/(N+1)

        ## Wt update
        W_t = soft_treshold( m2t(Delt@t2m(S,temp_m),sz,temp_m)-Lda_t/rho,
                             lda_t/rho)
        ## W update
        W = soft_treshold( S-Lda/rho,
                          lda_1/rho)
        
        # {X_1,X_2, ... X_N , S} Block updates
        # S update
        Z = m2t( S_inv@t2m(((W+Lda/rho) + m2t(Delt.T@t2m( (W_t+Lda_t/rho) ,temp_m),sz,temp_m) + (Y-X-Lda_h/rho)),temp_m),
                sz, temp_m)
        s_norm = 0
        s_norm += norm(S-Z)**2
        S = Z.copy()
        o = 0
        for i,m in enumerate(lr_modes):
            X_temp = X_i[i].copy()
            X_i[i], o_temp = soft_moden(X+Lda_i[i]/rho,psis[i]/rho,
                                m)
            o += psis[i]*o_temp
            s_temp = X_i[i] - X_temp
            s_norm += norm(s_temp)**2

        s.append(np.sqrt(s_norm))
        r_norm = 0
        # Dual variable updates and primal residual
        for i in range(N):
            r_temp = X-X_i[i]
            Lda_i[i] = Lda_i[i] + rho*(r_temp)
            r_norm += norm(r_temp)**2
        
        r_temp = W_t- m2t(Delt@t2m(S,temp_m),sz,temp_m)
        Lda_t += rho*(r_temp)
        r_norm += norm(r_temp)**2
        
        r_temp = W-S
        Lda += rho*r_temp
        r_norm += norm(r_temp)**2

        r_temp = X+S-Y
        Lda_h = rho*r_temp
        r_norm += norm(r_temp)**2
        r.append(np.sqrt(r_norm))

        obj.append(o + lda_t*norm((Delt@t2m(S,temp_m)).ravel(),1)+
                   lda_1*norm(S.ravel(),1))
        it +=1
        
        if verbose and it>1:
            print(f"It-{it}:\t## |r|={r[-1]:.5f} \t ## |s|={s[-1]:.5f} \t ## rho={rho:.4f} obj={obj[-1]:.4f} \t ## del_obj = {obj[-1]-obj[-2]:.4f} ")
        
        if not isinstance(detect_events, type(None)):
            num_detected_events = np.array([detect_events(np.abs(S), r) for r in ratios])
            detected_events['3'].append(num_detected_events[-1])
            detected_events['2'].append(num_detected_events[-2])
            detected_events['1'].append(num_detected_events[-3])
            detected_events['0.7'].append(num_detected_events[-4])
            detected_events['sum'].append(sum(num_detected_events))

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
                'detected_events': detected_events}
    return results

def lr_sts_fidelity(Y, temp_m, **kwargs):
    """Low-rank, Sparse and Temporally smooth term separation with hard constraint.

    Args:
        Y (np.ndarray): Observed data/signal/tensor
        temp_m (int): temporal mode of the tensor
        psis (float, list of floats): Weights of the nuclear norms. Defaults to 1
        lda_1 (float): Hyperparameter of the sparsity. Defaults to 1
        lda_t (float): Hyperparameter of the temporal smoothness. Defaults to 1
        lda_2 (float): Hyperparameter controlling the fidelity. Defaults to 1
        psis (tuple): Hyperparameters of the nuclear norms. Defaults to 1s
        verbose (int): Algorithm verbisity level. Defaults to 1.
        max_it (int): Maximum number of iterations allowed for the algorithm. Defaults to 250
        rho_upd (float): Step size update coefficient of the ADMM algorithm. Defaults to 1.2
        rho_mu (float): Step size update treshold of the ADMM algorithm. Defaults to 10
        err_tol (float): Convergence criteria for the algorithm. Defaults to 1e-5
    Returns:
        X (np.ndarray): Seperated low-rank and smooth part
        S (np.ndarray): Seperated sparse part 
    """
    # Parse keyword arguments
    sz = Y.shape
    lr_modes = kwargs.get('lr_modes', [i+1 for i in range(len(sz))])
    N = len(lr_modes)
    lda_1 = kwargs.get('lda_1',1)
    lda_2 = kwargs.get('lda_2',1)
    lda_t = kwargs.get('lda_t',1)

    psis = kwargs.get('psis', tuple([1 for _ in range(N)]))
    if isinstance(psis,float):
        psis = tuple([psis]*N)
    verbose = kwargs.get('verbose',1)
    max_it = kwargs.get('max_it',250)
    rho = kwargs.get('rho',1)
    err_tol = kwargs.get('err_tol',1e-5)
    eps_pri = err_tol   # Change later
    eps_dual = err_tol  # Change later
    # Initialize variables
    X_i = [np.zeros(sz) for _ in range(N)]
    W_t = np.zeros(sz)
    W = np.zeros(sz)
    S = np.zeros(sz)
    X = np.zeros(sz)

    # Initialize dual variables
    Lda_i = [np.zeros(sz) for _ in range(N)]    # X = X_i
    Lda_t = np.zeros(sz)                        # W_t = D xt S
    Lda = np.zeros(sz)                          # W = S

    # Initialize Delt matrix (Temporal first order difference operator)
    Delt = np.eye(sz[temp_m-1]) + np.diag(-np.ones(sz[temp_m-1]-1), 1)
    Delt[-1,0]=-1
    S_inv = inv(np.eye(sz[temp_m-1])*(lda_2+rho) + rho*Delt.T@Delt)

    it = 0
    obj = [np.inf]
    r = []  # Norm of primal residual in each iteration
    s = []  # Norm of dual residual
    rhos = [rho] # Step size
    while it<max_it:
        # {X, Wt, W} Block updates
        tempX = sum(X_i)-sum(Lda_i)/rho
        X = (rho*tempX + lda_2*(Y-S))/(rho*N+lda_2)

        ## Wt update
        W_t = soft_treshold( m2t(Delt@t2m(S,temp_m),sz,temp_m)-Lda_t/rho,
                             lda_t/rho)
        ## W update
        W = soft_treshold( S-Lda/rho,
                          lda_1/rho)
        
        # {X_1,X_2, ... X_N , S} Block updates
        # S update
        Z = lda_2*(Y-X) + rho*(W+Lda/rho) + rho*m2t(Delt.T@t2m(W_t+Lda_t/rho, temp_m),sz, temp_m)
        Z = m2t(S_inv@ t2m( Z,temp_m),sz, temp_m )
        # Z = m2t( S_inv@t2m(((W+Lda/rho) + m2t(Delt.T@t2m( (W_t+Lda_t/rho) ,temp_m),sz,temp_m) + (Y-X-Lda_h/rho)),temp_m),
        #         sz, temp_m)
        s_norm = 0
        s_norm += norm(S-Z)**2
        S = Z.copy()
        o = 0
        for i,m in enumerate(lr_modes):
            X_temp = X_i[i].copy()
            X_i[i], o_temp = soft_moden(X+Lda_i[i]/rho,psis[i]/rho,m)
            o += psis[i]*o_temp
            s_temp = X_i[i] - X_temp
            s_norm += norm(s_temp)**2

        s.append(np.sqrt(s_norm))
        r_norm = 0
        # Dual variable updates and primal residual
        for i in range(N):
            r_temp = X-X_i[i]
            Lda_i[i] = Lda_i[i] + rho*(r_temp)
            r_norm += norm(r_temp)**2
        
        r_temp = W_t- m2t(Delt@t2m(S,temp_m),sz,temp_m)
        Lda_t += rho*(r_temp)
        r_norm += norm(r_temp)**2
        
        r_temp = W-S
        Lda += rho*r_temp
        r_norm += norm(r_temp)**2
        r.append(np.sqrt(r_norm))

        obj.append(o + lda_t*norm((Delt@t2m(S,temp_m)).ravel(),1)+
                   lda_1*norm(S.ravel(),1)+ lda_2*norm(Y-X-S)**2)
        it +=1
        
        if verbose and it>1:
            print(f"It-{it}:\t## |r|={r[-1]:.5f} \t ## |s|={s[-1]:.5f} \t ## rho={rho:.4f} obj={obj[-1]:.4f} \t ## del_obj = {obj[-1]-obj[-2]:.4f} ")

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
                'rho':np.array(rhos)}
    return results