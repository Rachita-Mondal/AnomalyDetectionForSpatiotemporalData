import numpy as np

def project_to_simplex(y):
    """ Algorithm that projects the vector y onto the simplex based on the Projection Onto A 
    Simplex (Chen,Ye 2011)
        let y in R^n
        x = argmin ||x-y||
            s.t x in simplex^n
    Args:
        y (): vector

    Returns:
        x vector: y's projection onto simplex
    """
    n = len(y)
    idx = np.argsort(y)
    y_sorted = y[idx]
    t = np.zeros(n)
    i = n-1 # i = n-1
    while True:
        if i>=1:
            t_tmp = (sum(y[i:])-1)/(n-i)
            if t_tmp >= y[i-1]:
                t = t_tmp
                break
            else:
                i=i-1
        else:
            t = (sum(y)-1)/n
            break
    x = y-t
    x[x<0]=0
    return x

# def vec2simplex(vecX, l=1.):
#     m = vecX.size
#     vecS = np.sort(vecX)[::-1]
#     vecC = np.cumsum(vecS)-l
#     vecH = vecS - vecC /(np.arange(m)+1)
#     r = np.max(np.where(vecH>0)[0])
#     t = vecC[r]/(r+1)
#     return np.maximum(0,vecX-t)

def vec2simplex(y):
    """projsplx projects a vector to a simplex
    by the algorithm presented in 
    (Chen an Ye, "Projection Onto A Simplex", 2011)"""
    assert len(y.shape) == 1
    N = y.shape[0]
    y_flipsort = np.flipud(np.sort(y))
    cumsum = np.cumsum(y_flipsort)
    t = (cumsum - 1) / np.arange(1,N+1).astype('float')
    
    t_iter = t[:-1]
    t_last = t[-1]    
    y_iter = y_flipsort[1:]
    
    if np.all((t_iter - y_iter) < 0):
        t_hat = t_last
    else:
        # find i such that t>=y
        eq_idx = np.searchsorted(t_iter - y_iter, 0, side='left')
        t_hat = t_iter[eq_idx]

    x = y - t_hat
    # there may be a numerical error such that the constraints are not exactly met.
    x[x<0.] = 0.
    x[x>1.] = 1.
    assert np.abs(x.sum() - 1.) <= 1e-5
    assert np.all(x >= 0) and np.all(x <= 1.)
    return x