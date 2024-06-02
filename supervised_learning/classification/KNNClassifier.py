from supervised_learning.BaseEstimator import BaseEstimator
import numpy as np
from collections import Counter

class KNNClassifier(BaseEstimator):
    """
    k-Nearest Neighbors classifier.

    Parameters
    ----------
    n_neighbors : int, default=5
        Number of neighbors to use by default for k-neighbors queries.

    Methods
    -------
    fit(X, y)
        Fit the k-NN classifier to the training data.
    
    predict(X)
        Predict class labels for samples in X.
    
    get_params(deep=True)
        Get parameters for this estimator.
    
    set_params(**params)
        Set the parameters of this estimator.
    """

    def __init__(self, n_neighbors=5):
        if not isinstance(n_neighbors, int) or n_neighbors <= 0:
            raise ValueError("n_neighbors must be a positive integer.")
        self.n_neighbors = n_neighbors
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        """
        Fit the k-NN classifier to the training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.
        
        y : array-like of shape (n_samples,)
            The target values (class labels).
        """
        if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
            raise TypeError("X and y must be numpy arrays.")
        if X.shape[0] != y.shape[0]:
            raise ValueError("The number of samples in X and y must be equal.")

        self.X_train = X
        self.y_train = y

    def predict(self, X):
        """
        Predict class labels for samples in X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        y_pred : array-like of shape (n_samples,)
            The predicted class labels.
        """
        if not isinstance(X, np.ndarray):
            raise TypeError("X must be a numpy array.")
        if self.X_train is None or self.y_train is None:
            raise RuntimeError("The model has not been fitted yet.")
        
        predictions = [self._predict_single(x) for x in X]
        return np.array(predictions)

    def _predict_single(self, x):
        # Compute distances from x to all training data
        distances = np.linalg.norm(self.X_train - x, axis=1)
        
        # Find the nearest neighbors
        nearest_indices = np.argsort(distances)[:self.n_neighbors]
        nearest_labels = self.y_train[nearest_indices]
        
        # Majority vote
        most_common = Counter(nearest_labels).most_common(1)
        return most_common[0][0]

    def get_params(self, deep=True):
        """
        Get parameters for this estimator.

        Returns
        -------
        params : dict
            Parameter names mapped to their values.
        """
        return {"n_neighbors": self.n_neighbors}

    def set_params(self, **params):
        """
        Set the parameters of this estimator.

        Parameters
        ----------
        **params : dict
            Estimator parameters.

        Returns
        -------
        self : object
            Estimator instance.
        """
        for param, value in params.items():
            setattr(self, param, value)
        return self
