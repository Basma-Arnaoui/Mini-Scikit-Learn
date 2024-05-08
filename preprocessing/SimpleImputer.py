import numpy as np
from scipy import stats

class SimpleImputer:
    def __init__(self, strategy='mean'):
        self.strategy = strategy
        self.statistics_ = None

    def fit(self, X):
        """Fit the imputer on the data according to the strategy."""
        if self.strategy == 'mean':
            self.statistics_ = np.nanmean(X, axis=0)
        elif self.strategy == 'median':
            self.statistics_ = np.nanmedian(X, axis=0)
        elif self.strategy == 'most_frequent':
            self.statistics_ = stats.mode(X, nan_policy='omit').mode[0]
        else:
            raise ValueError("Unsupported strategy")
        return self

    def transform(self, X):
        """Replace missing values in X using the learned statistics."""
        if self.statistics_ is None:
            raise RuntimeError("Imputer has not been fitted")
        
        X_transformed = np.array(X)  # Make a copy of X to avoid changing the original data
        for i in range(X_transformed.shape[1]):
            mask = np.isnan(X_transformed[:, i])
            X_transformed[mask, i] = self.statistics_[i]
        return X_transformed

    def fit_transform(self, X):
        """Fit the imputer and transform X in one step."""
        return self.fit(X).transform(X)
