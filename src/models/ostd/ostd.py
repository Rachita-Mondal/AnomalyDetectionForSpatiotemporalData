"""Online Stochastic Tensor Decomposition
"""

from time import perf_counter

import numpy as np
from numpy.linalg import inv, norm

from src.proximal_ops.soft_treshold import soft_treshold
from src.multilinear_ops.m2t import m2t
from src.multilinear_ops.t2m import t2m


def ostd_init(Y, ranks, modes, **kwargs):
    """Initializes the online robust tensor decomposition
    """
    seed = kwargs.get('seed', None)
    rng = np.random.default_rng(seed)
    dims = Y.shape
    size = Y.size
    Lt_s = []
    At_s = []
    Bt_s = []
    for i, m in enumerate(modes):
        
        ni = dims[m-1]
        pi = np.prod(dims)//ni
        
        At_s.append(np.zeros((ranks[i], ranks[i])))
        Bt_s.append(np.zeros((ni, ranks[i])))

        # Random basis initialization
        Lt_s.append(rng.standard_normal(size=(ni, ranks[i])))

        # Bilateral Random Projections Initialization
        # y = t2m(Y, m).reshape((size,1))
        # Y2 = np.random.standard_normal(size=(1, ranks[i]))
        # for _ in range(2):
        #     Y1 = y@Y2
        #     Y2 = y.T@Y1
        

    return Lt_s, At_s, Bt_s

def ostd_sample(Yt, Lt_s, At_s, Bt_s, lda1, lda2, ranks, modes, verbose=2):
    """Sample-wise update of the online robust tensor decomposition
    """
    dims = Yt.shape
    Xs = []
    Es = []
    #Rs = []
    for i, m in enumerate(modes):
        Yit = t2m(Yt,m)
        ni, pi = Yit.shape
        Xs.append(np.zeros((ni,pi)))
        Es.append(np.zeros((ni,pi)))
        #Rs.append(np.zeros((pi, ranks[i])))
        # Project Yit[:,] vectors to the column space of Lt_s[i]
        for j in range(pi):
            y = Yit[:,j].reshape((ni,1))
            r, e, loss = project_r(y, Lt_s[i], lda1, lda2)
            At_s[i] += r@r.T            # update_A(At_s[i], r)
            Bt_s[i] += (y-e)@r.T # update_B(Bt_s[i], Yit[:,j], r, e)
            Xs[i][:,j] = (Lt_s[i]@r).ravel()
            Es[i][:,j] = e.ravel()
         #   Rs[i][j,:] = r

            Lt_s[i], L_change = update_L(Lt_s[i], At_s[i], Bt_s[i], lda1)
            if verbose > 2:
                print(f"Mode {i}, sample {j}:\tL change: {L_change:.4f}")
        Xs[i] = m2t(Xs[i], dims, m)
        Es[i] = m2t(Es[i], dims, m)

    return sum(Xs)/len(modes), sum(Es)/len(modes), Lt_s, At_s, Bt_s


def project_r(y, L, lda1, lda2, err_tol=1e-4, maxit=20000, verbose=1, benchmark=False):
    """Solves equation (7) in the paper. (Not fast, but simple)
    """
    if benchmark:
        tstart = perf_counter()
    d = y.shape[0]
    rank = L.shape[1]
    y = y.reshape((d,1))
    r = np.zeros((rank,1))
    e = np.zeros((d,1))
    r_new = np.zeros((rank,1))
    e_new = np.zeros((d,1))
    it = 0
    converged = False
    LLt = inv(L.T@L + lda1*np.eye(rank))@L.T
    while it<maxit:
        r_new = LLt@(y - e)
        e_new = soft_treshold(y - L@r_new, lda2)
        
        r, r_new = r_new, r
        e, e_new = e_new, e
        if norm(r_new - r)/d < err_tol and norm(e_new - e)/d < err_tol:
            converged = True
            break
        it += 1
    
    loss = 0.5*norm(y - L@r - e, ord=2) + 0.5*lda1*norm(r, ord=2) + lda2*norm(e, ord=1)
    # if benchmark:
    #     times["projection"].append(perf_counter() - tstart)
    if verbose > 1:
        print(f"r,e update:\tloss={loss:.4f}\tIterations: {it}\t Converged? {converged}")
    return r, e, loss


def update_L(L, A, B, lda1, method='sgd'):
    rank = A.shape[0]
    L_old = L.copy()
    A = A + np.eye(rank) * lda1

    if method == 'bcd':
        for j in range(rank):
            temp = (B[:,j] - L@A[:,j])/A[j,j] + L[:,j]
            L[:,j] = temp # /max(1, norm(temp))
    elif method == 'sgd_hessian':
        gL = (L@A - B) # / sample_count
        L = L - gL@inv(A)
    L_change = norm(L - L_old)
    return L, L_change