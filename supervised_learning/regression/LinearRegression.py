from supervised_learning.BaseEstimator import BaseEstimator
import numpy as np
import pandas as pd

class LinearRegression(BaseEstimator):
    def __init__(self):
        self.coef_ = None  # Coefficients (weights) of the linear regression model
        self.intercept_ = None  # Intercept of the linear regression model

    def fit(self, X, y):
        # Convert DataFrame inputs to NumPy arrays
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()
        if isinstance(y, pd.Series):
            y = y.to_numpy()

        # Check if X is 1-dimensional (i.e., a single feature)
        if X.ndim == 1:
            X = X.reshape(-1, 1)  # Reshape X to a 2D array with a single feature column

        # Add a column of ones to X to account for the intercept term
        X_with_intercept = np.column_stack((np.ones(X.shape[0]), X))

        # Calculate the coefficients using the normal equation (closed-form solution)
        coefficients = np.linalg.inv(X_with_intercept.T @ X_with_intercept) @ X_with_intercept.T @ y

        # Store the coefficients
        self.intercept_ = coefficients[0]
        self.coef_ = coefficients[1:]

    def predict(self, X):
        # Convert DataFrame inputs to NumPy arrays
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()

        # Add a column of ones to X to account for the intercept term
        X_with_intercept = np.column_stack((np.ones(X.shape[0]), X))

        # Calculate the predicted values
        predictions = X_with_intercept @ np.concatenate(([self.intercept_], self.coef_))

        return predictions
