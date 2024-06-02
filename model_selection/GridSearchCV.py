import numpy as np
from itertools import product

class GridSearchCV:
    """
    Exhaustive search over specified parameter values for an estimator.

    Parameters
    ----------
    estimator : estimator object
        The object to use to fit the data.
    
    param_grid : dict
        Dictionary with parameters names (`str`) as keys and lists of parameter settings to try as values.
    
    cv : int, default=5
        Number of folds in cross-validation.
    
    scoring : str, default='accuracy'
        Strategy to evaluate the performance of the cross-validated model on the test set.

    Attributes
    ----------
    best_params_ : dict
        Parameter setting that gave the best results on the hold out data.
    
    best_score_ : float
        Mean cross-validated score of the best_estimator.
    
    results_ : list
        Contains scores and parameter combinations for all parameter combinations in the grid.

    Methods
    -------
    fit(X, y)
        Run fit with all sets of parameters.
    """

    def __init__(self, estimator, param_grid, cv=5, scoring='accuracy'):
        if not isinstance(param_grid, dict):
            raise ValueError("param_grid must be a dictionary.")
        if not isinstance(cv, int) or cv <= 1:
            raise ValueError("cv must be an integer greater than 1.")
        if not hasattr(estimator, 'fit') or not hasattr(estimator, 'predict'):
            raise ValueError("estimator must have 'fit' and 'predict' methods.")
        
        self.estimator = estimator
        self.param_grid = param_grid
        self.cv = cv
        self.scoring = scoring
        self.best_params_ = None
        self.best_score_ = -np.inf
        self.results_ = []

    def fit(self, X, y):
        """
        Run fit with all sets of parameters.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data to fit.
        
        y : array-like of shape (n_samples,)
            The target variable to try to predict in the case of supervised learning.
        """
        if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
            raise TypeError("X and y must be numpy arrays.")
        if X.shape[0] != y.shape[0]:
            raise ValueError("The number of samples in X and y must be equal.")

        for params in self._param_grid_combinations():
            scores = []
            for train_idx, test_idx in self._k_fold(len(y), self.cv):
                self.estimator.set_params(**params)
                self.estimator.fit(X[train_idx], y[train_idx])
                pred = self.estimator.predict(X[test_idx])
                score = self._calculate_score(y[test_idx], pred)
                scores.append(score)

            average_score = np.mean(scores)
            self.results_.append({'params': params, 'score': average_score})
            if average_score > self.best_score_:
                self.best_score_ = average_score
                self.best_params_ = params

        self.results_.sort(key=lambda x: x['score'], reverse=True)
        return self

    def _param_grid_combinations(self):
        keys, values = zip(*self.param_grid.items())
        for v in product(*values):
            yield dict(zip(keys, v))

    def _k_fold(self, n, n_folds):
        indices = np.arange(n)
        np.random.shuffle(indices)
        fold_sizes = np.full(n_folds, n // n_folds, dtype=int)
        fold_sizes[:n % n_folds] += 1
        current = 0
        for fold_size in fold_sizes:
            start, stop = current, current + fold_size
            train_indices = np.concatenate([indices[:start], indices[stop:]])
            test_indices = indices[start:stop]
            yield train_indices, test_indices

    def _calculate_score(self, y_true, y_pred):
        if self.scoring == 'accuracy':
            return np.mean(y_true == y_pred)
        else:
            raise ValueError(f"Unsupported scoring method: {self.scoring}")
