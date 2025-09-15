import numpy as np

def mysign(A):
    """Generate sign tensor that assign 1 to every non-negative element."""
    S = np.sign(A)
    try:
        S[S==0]=1
    except:
        if S==0: 
            S=1
    return S

def qmult(A, seed=None):
    """QMULT Pre-multiply matrix by random orthogonal matrix.

        QMULT(A) returns Q*A where Q is a random real orthogonal matrix
        from the Haar distribution of dimension the number of rows in A.
        Special case: if A is a scalar then QMULT(A) is the same as QMULT(EYE(A)).
        
        Reference:
        G. W. Stewart, The efficient generation of random
        orthogonal matrices with an application to condition estimators,
        SIAM J. Numer. Anal., 17 (1980), 403-409.
    
        Nicholas J. Higham
        Copyright 1984-2005 The MathWorks, Inc.
        $Revision: 1.4.4.2 $  $Date: 2005/11/18 14:15:22 $   

    Args:
        A (int,np.array): Array to be pre-multiplied or an integer.

    Returns:
        B (np.array): Orthonormal array or the result of pre-multiplication of A
        with on orthonormal array.
    """
    if type(A)== type(np.zeros(2)):
        n = A.shape[0]
    elif type(A)==type(4):
        assert A>0
        n = A
        A = np.eye(n)
    
    d = np.zeros((n, 1))
    for k in range(n-1, 0, -1):
        # Generate random Householder transformation.
        x = np.random.randn(n-k+1, 1)
        s = np.linalg.norm(x)
        sgn = mysign(x[0])
        s = sgn*s
        d[k-1] = -sgn
        x[0] = x[0] + s
        beta = s*x[0]
        # Apply the transformation to A.
        y = x.T @ A[k-1:n, :]
        A[k-1:n, :] = A[k-1:n, :] - x @ (y/beta)
    # Tidy up signs.
    for i in range(n-1):
        A[i, :] = d[i]*A[i, :]
    A[n-1, :] = A[n-1, :] * mysign(np.random.randn())
    B = A
    return B

