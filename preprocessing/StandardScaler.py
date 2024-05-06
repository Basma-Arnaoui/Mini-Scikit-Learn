import numpy as np

class StandardScaler:
    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def fit(self, X, y=None):
        """
        Compute the mean and std to be used for later scaling.

        Parameters:
            X : {array-like, sparse matrix} of shape (n_samples, m_features)
                The data used to compute the mean and standard deviation used for later scaling along the features axis.
            y : None
                Ignored.

        Returns:
            self : object
                Fitted scaler.
        """
        X = np.asarray(X)
        # Compute the mean and standard deviation along the features axis
        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0)

        return self

    def transform(self, X):
        """
        Perform standardization by centering and scaling.

        Parameters:
            X : {array-like, sparse matrix} of shape (n_samples, n_features)
                The data used to scale along the features axis.

        Returns:
            X_transformed : {ndarray, sparse matrix} of shape (n_samples, n_features)
                Transformed array.
        """
        # Check if the scaler has been fitted
        if self.mean_ is None or self.std_ is None:
            raise ValueError("Scaler has not been fitted.")

        X = np.asarray(X)
        # Perform standardization by centering and scaling
        X_transformed = (X - self.mean_) / self.std_

        return X_transformed

    def fit_transform(self, X, y=None):
        """
        Fit to data, then transform it.

        Parameters:
            X : array-like of shape (n_samples, n_features)
                Input samples.
            y : array-like of shape (n_samples,) or (n_samples, n_outputs), default=None
                Target values (ignored).

        Returns:
            X_new : ndarray of shape (n_samples, n_features)
                Transformed array.
        """
        self.fit(X)
        return self.transform(X)
