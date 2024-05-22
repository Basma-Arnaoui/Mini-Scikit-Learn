import numpy as np
from scipy.special import expit
from sklearn.base import BaseEstimator, ClassifierMixin

class LogisticRegression(BaseEstimator, ClassifierMixin):
    def __init__(self, learning_rate=0.01, n_iterations=1000, tol=1e-5, regularization=0.01):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.tol = tol
        self.regularization = regularization
        self.coef_ = None
        self.intercept_ = None

    def _sigmoid(self, z):
        return expit(z)

    def fit(self, X, y):
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
        X_b = np.c_[np.ones((X.shape[0], 1)), X]  # Adding the intercept term
        return self._sigmoid(X_b.dot(np.r_[self.intercept_, self.coef_]))

    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)

    def get_params(self, deep=True):
        return {"learning_rate": self.learning_rate, "n_iterations": self.n_iterations, "tol": self.tol, "regularization": self.regularization}

    def set_params(self, **params):
        for param, value in params.items():
            setattr(self, param, value)
        return self
