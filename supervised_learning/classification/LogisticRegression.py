import numpy as np
from scipy.special import expit  # Sigmoid function
from supervised_learning.BaseEstimator import BaseEstimator

class LogisticRegression(BaseEstimator):
    def __init__(self, learning_rate=0.001, n_iterations=5000, tol=0.000001, regularization=0.01):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.tol = tol
        self.regularization = regularization
        self.coef_ = None
        self.intercept_ = None

    def _sigmoid(self, z):
        return expit(z)

    def fit(self, X, y):
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        self.coef_ = np.zeros(X_b.shape[1])

        for iteration in range(self.n_iterations):
            linear_model = X_b.dot(self.coef_)
            predictions = self._sigmoid(linear_model)
            
            # Add regularization term
            regularization_term = self.regularization * np.r_[0, self.coef_[1:]]
            gradient = (X_b.T.dot(predictions - y) + regularization_term) / X_b.shape[0]
            
            prev_coef = np.copy(self.coef_)
            self.coef_ -= self.learning_rate * gradient

            # Check convergence
            if np.all(np.abs(self.coef_ - prev_coef) < self.tol):
                print(f'Converged at iteration {iteration}')
                break

        self.intercept_ = self.coef_[0]
        self.coef_ = self.coef_[1:]

    def predict_proba(self, X):
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        linear_model = X_b.dot(np.r_[self.intercept_, self.coef_])
        return self._sigmoid(linear_model)

    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)
