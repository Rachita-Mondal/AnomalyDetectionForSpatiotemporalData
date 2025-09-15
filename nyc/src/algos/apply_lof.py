import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import cdist
from src.util.LOF import lof

def apply_lof(Y, K):
    # Initialize the result array with the same shape as Y
    L = np.zeros(Y.shape)
    
    # Loop through the dimensions of the input tensor
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            for k in range(Y.shape[3]):
                # Extract the 3rd mode fiber and apply LOF
                data_slice = Y[i, j, :, k]
                suspicious_index, lof_values = lof(data_slice.reshape(-1,1), K)
                
                # Assign LOF values to the result tensor
                L[i, j, :, k] = lof_values
    
    return L
