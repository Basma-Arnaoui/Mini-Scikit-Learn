import numpy as np
from scipy import stats

class SimpleImputer:
    """
    Imputation transformer for completing missing values.

    Parameters
    ----------
    strategy : str, default='mean'
        The imputation strategy.
        - If "mean", then replace missing values using the mean along each column.
        - If "median", then replace missing values using the median along each column.
        - If "most_frequent", then replace missing values using the most frequent value along each column.

    Methods
    -------
    fit(X)
        Fit the imputer on the data according to the strategy.
    
    transform(X)
        Replace missing values in X using the learned statistics.
    
    fit_transform(X)
        Fit the imputer and transform X in one step.
    """

    def __init__(self, strategy='mean'):
        if strategy not in ['mean', 'median', 'most_frequent']:
            raise ValueError("Strategy must be one of ['mean', 'median', 'most_frequent']")
        self.strategy = strategy
        self.statistics_ = None

    def fit(self, X):
        """
        Fit the imputer on the data according to the strategy.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data used to compute the statistics for imputation.

        Returns
        -------
        self : object
            Fitted imputer.
        """
        if not isinstance(X, np.ndarray):
            raise TypeError("X must be a numpy array.")

        if self.strategy == 'mean':
            self.statistics_ = np.nanmean(X, axis=0)
        elif self.strategy == 'median':
            self.statistics_ = np.nanmedian(X, axis=0)
        elif self.strategy == 'most_frequent':
            self.statistics_ = stats.mode(X, nan_policy='omit').mode[0]
        return self

    def transform(self, X):
        """
        Replace missing values in X using the learned statistics.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data to be imputed.

        Returns
        -------
        X_transformed : array-like of shape (n_samples, n_features)
            The imputed data.
        """
        if not isinstance(X, np.ndarray):
            raise TypeError("X must be a numpy array.")
        if self.statistics_ is None:
            raise RuntimeError("Imputer has not been fitted.")
        
        X_transformed = np.array(X)
        for i in range(X_transformed.shape[1]):
            mask = np.isnan(X_transformed[:, i])
            X_transformed[mask, i] = self.statistics_[i]
        return X_transformed

    def fit_transform(self, X):
        """
        Fit the imputer and transform X in one step.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data to be imputed.

        Returns
        -------
        X_transformed : array-like of shape (n_samples, n_features)
            The imputed data.
        """
        return self.fit(X).transform(X)
