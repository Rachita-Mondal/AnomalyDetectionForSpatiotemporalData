import numpy as np
from sklearn.svm import OneClassSVM
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import StandardScaler

def one_class_svm(Y, out_fr):
    """
    Applies one-class SVM on the third mode fibers of a four-mode tensor Y.
    
    Parameters:
    Y : numpy.ndarray
        Input tensor of shape (sz_1, sz_2, sz_3, sz_4).
    out_fr : float
        Outlier fraction parameter for the one-class SVM.
        
    Returns:
    numpy.ndarray
        An array of the same shape as Y with SVM scores.
    """
    sz_1, sz_2, sz_3, sz_4 = Y.shape
    scr = np.zeros_like(Y)
    
    for i in range(sz_1):
        for j in range(sz_2):
            for k in range(sz_4):
                # Extract the third mode fibers
                data = Y[i, j, :, k].reshape(-1, 1)  # Reshape to (n_samples, n_features)
                labels = np.ones(sz_3)  # All ones for one-class SVM
                
                # Standardize the data
                scaler = StandardScaler()
                data = scaler.fit_transform(data)
                
                # Fit One-Class SVM
                model = OneClassSVM(nu=out_fr, kernel='rbf', gamma='auto')
                # Cross-validation prediction
                predictions = cross_val_predict(model, data, cv=5, method='predict')
                
                # Assign predictions to the result tensor
                scr[i, j, :, k] = predictions
    
    # Negate the scores
    scr = -scr
    
    return scr