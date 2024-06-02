from supervised_learning.BaseEstimator import BaseEstimator
import numpy as np
import pandas as pd

class LinearRegression(BaseEstimator):
    """
    Linear Regression model.

    Methods
    -------
    fit(X, y)
        Fit the linear regression model to the training data.
    
    predict(X)
        Predict target values for samples in X.
    """

    def __init__(self):
        self.coef_ = None  # Coefficients (weights) of the linear regression model
        self.intercept_ = None  # Intercept of the linear regression model

    def fit(self, X, y):
        """
        Fit the linear regression model to the training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features) or DataFrame
            The training input samples.
        
        y : array-like of shape (n_samples,) or Series
            The target values.
        """
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()
        if isinstance(y, pd.Series):
            y = y.to_numpy()

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        if X.shape[0] != y.shape[0]:
            raise ValueError("The number of samples in X and y must be equal.")
        
        X_with_intercept = np.column_stack((np.ones(X.shape[0]), X))
        try:
            coefficients = np.linalg.inv(X_with_intercept.T @ X_with_intercept) @ X_with_intercept.T @ y
        except np.linalg.LinAlgError:
            raise np.linalg.LinAlgError("Matrix inversion failed. This might be due to singular matrix.")

        self.intercept_ = coefficients[0]
        self.coef_ = coefficients[1:]

    def predict(self, X):
        """
        Predict target values for samples in X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features) or DataFrame
            The input samples.

        Returns
        -------
        y_pred : array-like of shape (n_samples,)
            The predicted target values.
        """
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()

        if self.intercept_ is None or self.coef_ is None:
            raise RuntimeError("The model has not been fitted yet.")

        X_with_intercept = np.column_stack((np.ones(X.shape[0]), X))
        predictions = X_with_intercept @ np.concatenate(([self.intercept_], self.coef_))
        return predictions
