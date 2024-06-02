import numpy as np

def resample_data(X, y, n_samples=None, random_state=None):
    """
    Resample the data arrays X and y.
    
    Parameters:
        X (np.array): Input features array.
        y (np.array): Target array.
        n_samples (int): Number of samples to generate. If None, the same number of samples as X is used.
        random_state (int): Seed for the random number generator for reproducibility.
    
    Returns:
        X_resampled (np.array): Resampled features array.
        y_resampled (np.array): Resampled target array.
    """
    if n_samples is None:
        n_samples = len(X)
    
    rng = np.random.default_rng(random_state)
    indices = rng.choice(len(X), size=n_samples, replace=True)
    X_resampled = X[indices]
    y_resampled = y[indices]
    
    return X_resampled, y_resampled
