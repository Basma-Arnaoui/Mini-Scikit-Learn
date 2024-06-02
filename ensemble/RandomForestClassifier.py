import numpy as np
from supervised_learning.BaseEstimator import BaseEstimator
from supervised_learning.classification.DecisionTreeClassifier import DecisionTreeClassifier
from collections import Counter

class RandomForestClassifier(BaseEstimator):
    """
    Random Forest classifier.

    Parameters
    ----------
    n_estimators : int, default=10
        The number of trees in the forest.
    
    max_depth : int, default=2
        The maximum depth of the trees.
    
    min_samples_split : int, default=2
        The minimum number of samples required to split an internal node.
    
    max_features : int or None, default=None
        The number of features to consider when looking for the best split.
    
    random_state : int or None, default=None
        Controls the randomness of the estimator.

    Methods
    -------
    fit(X, y)
        Fit the random forest classifier to the training data.
    
    predict(X)
        Predict class labels for samples in X.
    
    get_params(deep=True)
        Get parameters for this estimator.
    
    set_params(**params)
        Set the parameters of this estimator.
    """

    def __init__(self, n_estimators=10, max_depth=2, min_samples_split=2, max_features=None, random_state=None):
        if not isinstance(n_estimators, int) or n_estimators <= 0:
            raise ValueError("n_estimators must be a positive integer.")
        if not isinstance(max_depth, int) or max_depth <= 0:
            raise ValueError("max_depth must be a positive integer.")
        if not isinstance(min_samples_split, int) or min_samples_split <= 0:
            raise ValueError("min_samples_split must be a positive integer.")
        if max_features is not None and (not isinstance(max_features, int) or max_features <= 0):
            raise ValueError("max_features must be a positive integer or None.")
        
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.random_state = random_state
        self.trees = []

    def fit(self, X, y):
        """
        Fit the random forest classifier to the training data.

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
        
        np.random.seed(self.random_state)
        for _ in range(self.n_estimators):
            tree = DecisionTreeClassifier(max_depth=self.max_depth, min_samples_split=self.min_samples_split)
            X_sample, y_sample = self._bootstrap_samples(X, y)
            y_sample = y_sample.reshape(-1, 1)  # Ensure y has 2 dimensions
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def _bootstrap_samples(self, X, y):
        n_samples = X.shape[0]
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        return X[indices], y[indices]

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
        if not self.trees:
            raise RuntimeError("The model has not been fitted yet.")
        
        predictions = np.array([tree.predict(X) for tree in self.trees])
        return np.array([self._most_common_label(predictions[:, i]) for i in range(X.shape[0])])

    def _most_common_label(self, y):
        counter = Counter(y)
        most_common = counter.most_common(1)[0][0]
        return most_common
    
    def get_params(self, deep=True):
        """
        Get parameters for this estimator.

        Returns
        -------
        params : dict
            Parameter names mapped to their values.
        """
        return {
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "min_samples_split": self.min_samples_split,
            "max_features": self.max_features,
            "random_state": self.random_state
        }

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
