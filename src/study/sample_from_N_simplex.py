import numpy as np


def sample_from_N_simplex(trial, N):
    """Sample a point from the N-dimensional unit simplex.
    
    Favors the points that are close to 1,0,0,...,0
    Args:
        trial: The Optuna trial object.
        N: The dimension of the simplex.
    Returns:
        A point sampled from the N-dimensional unit simplex.
    """

    thetas = np.array([trial.suggest_float(f"theta_{i}", 0, np.pi/2) for i in range(1, N)]).reshape((1, N-1))
    sin_thetas = np.concatenate([np.ones((1,1)), np.sin(thetas)], axis=1)
    cos_thetas = np.concatenate([np.cos(thetas), np.ones((1,1))], axis=1)
    x = (np.cumprod(sin_thetas, axis=1)*cos_thetas)**2
    return x