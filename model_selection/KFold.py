import numpy as np

class KFold:
    """
    K-Folds cross-validator.

    Parameters
    ----------
    n_splits : int, default=5
        Number of folds. Must be at least 2.
    
    shuffle : bool, default=False
        Whether to shuffle the data before splitting into batches.
    
    random_state : int, RandomState instance or None, default=None
        Controls the randomness of the shuffling.

    Methods
    -------
    split(X, y=None)
        Generate indices to split data into training and test set.
    """

    def __init__(self, n_splits=5, shuffle=False, random_state=None):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def split(self, X, y=None):
        """
        Generate indices to split data into training and test set.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data, where n_samples is the number of samples and n_features is the number of features.
        
        y : array-like, shape (n_samples,)
            The target variable for supervised learning problems.
        """
        num_samples = len(X)
        indices = np.arange(num_samples)
        
        if self.shuffle:
            rng = np.random.RandomState(self.random_state)
            rng.shuffle(indices)

        fold_sizes = np.full(self.n_splits, num_samples // self.n_splits, dtype=int)
        fold_sizes[:num_samples % self.n_splits] += 1
        current = 0
        for fold_size in fold_sizes:
            start, stop = current, current + fold_size
            test_indices = indices[start:stop]
            train_indices = np.concatenate([indices[:start], indices[stop:]])
            yield train_indices, test_indices
            current = stop
