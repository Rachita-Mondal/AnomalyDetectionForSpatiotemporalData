import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import cdist

def lof(A, k):
    # Convert k to an integer if it's a fraction
    if k < 1:
        k = round(k * A.shape[0])
    
    # Initialize the nearest neighbors model
    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='auto').fit(A)
    distances, indices = nbrs.kneighbors(A)
    
    # Ignore the first column which is the point itself
    indices = indices[:, 1:]
    distances = distances[:, 1:]
    
    # Compute k-distance
    k_dist = distances[:, -1]
    
    # Initialize lrd_values
    lrd_values = np.zeros(A.shape[0])
    
    # Compute lrd for each element
    for i in range(A.shape[0]):
        lrd_values[i] = compute_lrd(A, i, k_dist[i], indices[i], k) + np.finfo(float).eps
    
    # Compute LOF
    lof_values = np.zeros(A.shape[0])
    for i in range(A.shape[0]):
        lrd_i = lrd_values[i]
        lrd_neighbors = lrd_values[indices[i]]
        lof_values[i] = np.sum(lrd_neighbors / lrd_i) / k
    
    # Rank indices based on LOF values (descending order)
    suspicious_index = np.argsort(-lof_values)
    
    return suspicious_index, lof_values

def compute_lrd(A, index_p, k_dist_p, k_index_p, k):
    # Compute the reachability distance
    distances = cdist([A[index_p]], A[k_index_p], metric='euclidean').flatten()
    reach_dist = np.maximum(distances, k_dist_p)
    
    # Compute the local reachability density
    lrd_value = k / np.sum(reach_dist)
    return lrd_value