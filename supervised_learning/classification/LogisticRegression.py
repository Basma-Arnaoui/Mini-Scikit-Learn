import numpy as np
from scipy.special import expit
from supervised_learning.BaseEstimator import BaseEstimator
from sklearn.base import ClassifierMixin

class LogisticRegression(BaseEstimator, ClassifierMixin):
    """
    Logistic Regression classifier.

    Parameters
    ----------
    learning_rate : float, default=0.01
        Learning rate for gradient descent.
    
    n_iterations : int, default=1000
        Number of iterations for gradient descent.
    
    tol : float, default=1e-5
        Tolerance for stopping criteria.
    
    regularization : float, default=0.01
        Regularization strength.

    Methods
    -------
    fit(X, y)
        Fit the logistic regression model to the training data.
    
    predict_proba(X)
        Predict class probabilities for samples in X.
    
    predict(X)
        Predict class labels for samples in X.
    
    get_params(deep=True)
        Get parameters for this estimator.
    
    set_params(**params)
        Set the parameters of this estimator.
    """

    def __init__(self, learning_rate=0.01, n_iterations=1000, tol=1e-5, regularization=0.01):
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")
        if n_iterations <= 0:
            raise ValueError("n_iterations must be positive.")
        if tol <= 0:
            raise ValueError("tol must be positive.")
        if regularization < 0:
            raise ValueError("regularization must be non-negative.")
        
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.tol = tol
        self.regularization = regularization
        self.coef_ = None
        self.intercept_ = None

    def _sigmoid(self, z):
        return expit(z)

    def fit(self, X, y):
        """
        Fit the logistic regression model to the training data.

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

        m, n = X.shape
        X_b = np.c_[np.ones((m, 1)), X]  # Adding the intercept term
        self.coef_ = np.zeros(n + 1)

        for iteration in range(self.n_iterations):
            linear_model = X_b.dot(self.coef_)
            predictions = self._sigmoid(linear_model)
            
            # Gradient with L2 regularization
            regularization_term = self.regularization * np.r_[0, self.coef_[1:]]
            gradient = (X_b.T.dot(predictions - y) + regularization_term) / m

            # Update weights
            self.coef_ -= self.learning_rate * gradient

            # Check for convergence
            if np.linalg.norm(gradient) < self.tol:
                break

        self.intercept_ = self.coef_[0]
        self.coef_ = self.coef_[1:]
        return self

    def predict_proba(self, X):
        """
        Predict class probabilities for samples in X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        probabilities : array-like of shape (n_samples,)
            The predicted class probabilities.
        """
        if not isinstance(X, np.ndarray):
            raise TypeError("X must be a numpy array.")
        if self.coef_ is None or self.intercept_ is None:
            raise RuntimeError("The model has not been fitted yet.")
        
        X_b = np.c_[np.ones((X.shape[0], 1)), X]  # Adding the intercept term
        return self._sigmoid(X_b.dot(np.r_[self.intercept_, self.coef_]))

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
        
        return (self.predict_proba(X) >= 0.5).astype(int)

    def get_params(self, deep=True):
        """
        Get parameters for this estimator.

        Returns
        -------
        params : dict
            Parameter names mapped to their values.
        """
        return {
            "learning_rate": self.learning_rate,
            "n_iterations": self.n_iterations,
            "tol": self.tol,
            "regularization": self.regularization
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
