import numpy as np

def m_soft_threshold(X, sigma):
    """
    Soft thresholding function.
    
    Parameters:
    X : numpy.ndarray
        Input array to apply soft thresholding.
    sigma : float
        The threshold value.
        
    Returns:
    numpy.ndarray
        The thresholded array.
    """
    temp = np.abs(X) - sigma
    temp[temp < 0] = 0
    X = temp * np.sign(X)
    return X