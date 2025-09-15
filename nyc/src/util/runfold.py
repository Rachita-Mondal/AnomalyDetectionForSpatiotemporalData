import numpy as np

def runfold(A, first_mode=None):
    """
    Right unfolding operator for a tensor.
    
    Parameters:
    A : numpy.ndarray
        The input tensor to be unfolded.
    first_mode : int, optional
        The mode along which to unfold. If None, use the first mode of the tensor.
        
    Returns:
    numpy.ndarray
        The unfolded tensor.
    """
    if first_mode is not None:
        # Get the shape of the tensor
        shape = A.shape
        # Reshape the tensor
        A = np.reshape(A, (shape[first_mode], -1))
    else:
        # Default unfolding along the first mode
        shape = A.shape
        A = np.reshape(A, (shape[0], -1))
    
    return A