import numpy as np
from supervised_learning.regression.DecisionTreeRegressor import DecisionTreeRegressor
from sklearn.utils import resample
from sklearn.metrics import r2_score
from supervised_learning.BaseEstimator import BaseEstimator

class RandomForestRegressor(BaseEstimator):
    """
    Random Forest Regressor.

    Parameters
    ----------
    n_estimators : int, default=10
        The number of trees in the forest.
    
    min_samples_split : int, default=2
        The minimum number of samples required to split an internal node.
    
    max_depth : int or None, default=None
        The maximum depth of the trees.
    
    oob_score : bool, default=False
        Whether to use out-of-bag samples to estimate the generalization accuracy.

    Methods
    -------
    fit(X, y)
        Fit the random forest regressor to the training data.
    
    predict(X)
        Predict target values for samples in X.
    """

    def __init__(self, n_estimators=10, min_samples_split=2, max_depth=None, oob_score=False):
        if not isinstance(n_estimators, int) or n_estimators <= 0:
            raise ValueError("n_estimators must be a positive integer.")
        if not isinstance(min_samples_split, int) or min_samples_split <= 0:
            raise ValueError("min_samples_split must be a positive integer.")
        if max_depth is not None and (not isinstance(max_depth, int) or max_depth <= 0):
            raise ValueError("max_depth must be a positive integer or None.")
        
        self.n_estimators = n_estimators
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.oob_score = oob_score
        self.oob_score_ = np.nan
        self.trees = []

    def fit(self, X, y):
        """
        Fit the random forest regressor to the training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.
        
        y : array-like of shape (n_samples,)
            The target values.
        """
        if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
            raise TypeError("X and y must be numpy arrays.")
        if X.shape[0] != y.shape[0]:
            raise ValueError("The number of samples in X and y must be equal.")
        
        n_samples = X.shape[0]
        self.trees = []
        oob_predictions = np.zeros(n_samples)
        oob_counts = np.zeros(n_samples)

        for _ in range(self.n_estimators):
            indices = resample(np.arange(n_samples), replace=True)
            oob_indices = np.setdiff1d(np.arange(n_samples), indices, assume_unique=True)
            X_bts, y_bts = X[indices], y[indices]
            tree = DecisionTreeRegressor(min_samples_split=self.min_samples_split, max_depth=self.max_depth)
            tree.fit(X_bts, y_bts)
            self.trees.append(tree)
            
            if self.oob_score and len(oob_indices) > 0:
                y_pred_oob = tree.predict(X[oob_indices])
                oob_predictions[oob_indices] += y_pred_oob
                oob_counts[oob_indices] += 1

        if self.oob_score:
            valid_oob = oob_counts > 0
            oob_predictions[valid_oob] /= oob_counts[valid_oob]
            self.oob_score_ = r2_score(y[valid_oob], oob_predictions[valid_oob])

    def predict(self, X):
        """
        Predict target values for samples in X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        y_pred : array-like of shape (n_samples,)
            The predicted target values.
        """
        if not isinstance(X, np.ndarray):
            raise TypeError("X must be a numpy array.")
        if not self.trees:
            raise RuntimeError("The model has not been fitted yet.")
        
        predictions = np.array([tree.predict(X) for tree in self.trees])
        return np.mean(predictions, axis=0)
