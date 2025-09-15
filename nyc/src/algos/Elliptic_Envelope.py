import numpy as np
from sklearn.covariance import EllipticEnvelope

def robust_cov(X):
    """Compute robust covariance matrix and mean."""
    model = EllipticEnvelope(support_fraction=0.9)
    model.fit(X)
    robust_mean = model.location_
    robust_cov = model.covariance_
    return robust_mean, robust_cov
def mahal_dist(Y):
    """
    Computes Mahalanobis distance of each entry of each of the mode-3 fibers.
    Computes a robust covariance with:
    mahal_Y : Mahalanobis distance from mean.
    est_Y : Mean of each third mode fiber.
    """
    np.random.seed(123)
    s = Y.shape
    mahal_Y = np.zeros(s)
    est_Y = np.zeros(s)

    zero_ind_tensor = np.sum(np.abs(Y), axis=2)

    for i in range(s[0]):
        for j in range(s[1]):
            for k in range(s[3]):
                if zero_ind_tensor[i, j, k] == 0:
                    mahal_Y[i, j, :, k] = 0
                else:
                    X = Y[i, j, :, k].reshape(-1, 1)
                    robust_mean, robust_cov = robust_cov(X)
                    
                    # Compute Mahalanobis distance
                    centered_X = X - robust_mean.reshape(-1, 1)
                    inv_cov = np.linalg.inv(robust_cov)
                    mahal_Y[i, j, :, k] = np.sqrt((centered_X.T @ inv_cov @ centered_X).flatten())
                    est_Y[i, j, :, k] = robust_mean.flatten()
                    
                    # Ensure non-zero Mahalanobis distance
                    if np.all(mahal_Y[i, j, :, k] == 0):
                        X_noisy = X + np.sqrt(0.01) * np.random.randn(X.shape[0], 1)
                        robust_mean, robust_cov = robust_cov(X_noisy)
                        centered_X = X - robust_mean.reshape(-1, 1)
                        mahal_Y[i, j, :, k] = np.sqrt((centered_X.T @ np.linalg.inv(robust_cov) @ centered_X).flatten())
                        est_Y[i, j, :, k] = robust_mean.flatten()
    
    return mahal_Y # est_Y

