import numpy as np
from sklearn.tree import DecisionTreeRegressor

class GradientBoostingRegressor:
    """
    Gradient Boosting Regressor implementation using DecisionTreeRegressor as the base estimator.

    Parameters
    ----------
    n_estimators : int, default=100
        The number of boosting stages to be run.
    
    learning_rate : float, default=0.1
        Shrinks the contribution of each tree by learning_rate. There is a trade-off between learning_rate and n_estimators.
    
    max_depth : int, default=3
        Maximum depth of the individual regression estimators. The maximum depth limits the number of nodes in the tree.
    
    random_state : int, RandomState instance or None, default=None
        Controls the randomness of the estimator. The features are always randomly permuted at each split.
    
    Attributes
    ----------
    trees : list
        List of fitted trees (weak learners).
    
    initial_prediction : float
        Initial prediction for all instances.

    Methods
    -------
    fit(X, y)
        Build a gradient boosting regressor from the training set (X, y).

    predict(X)
        Predict target values for X.
    """

    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3, random_state=None):
        if not isinstance(n_estimators, int) or n_estimators <= 0:
            raise ValueError("n_estimators must be a positive integer.")
        if not isinstance(learning_rate, float) or learning_rate <= 0:
            raise ValueError("learning_rate must be a positive float.")
        if not isinstance(max_depth, int) or max_depth <= 0:
            raise ValueError("max_depth must be a positive integer.")

        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.random_state = random_state
        self.trees = []
        self.initial_prediction = None

    def fit(self, X, y):
        """
        Build a gradient boosting regressor from the training set (X, y).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.
        
        y : array-like of shape (n_samples,)
            The target values (continuous).

        """
        if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
            raise TypeError("X and y must be numpy arrays.")
        if X.shape[0] != y.shape[0]:
            raise ValueError("The number of samples in X and y must be equal.")

        # Initial prediction (mean)
        self.initial_prediction = np.mean(y)
        F = np.full(y.shape, self.initial_prediction)
        
        for _ in range(self.n_estimators):
            residuals = y - F
            tree = DecisionTreeRegressor(max_depth=self.max_depth, random_state=self.random_state)
            tree.fit(X, residuals)
            predictions = tree.predict(X)
            F += self.learning_rate * predictions
            self.trees.append(tree)

    def predict(self, X):
        """
        Predict target values for X.

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
        
        F = np.full(X.shape[0], self.initial_prediction)
        for tree in self.trees:
            F += self.learning_rate * tree.predict(X)
        return F
