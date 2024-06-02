import numpy as np

class MinMaxScaler:
    """
    MinMaxScaler for scaling features to a given range, usually [0, 1].

    Methods
    -------
    fit(X, y=None)
        Compute the min and max to be used for later scaling.
    
    transform(X)
        Perform normalization by scaling to the range [0, 1].
    
    fit_transform(X, y=None)
        Fit to data, then transform it.
    """

    def __init__(self):
        self.min = None
        self.max = None

    def fit(self, X, y=None):
        """
        Compute the min and max to be used for later scaling.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data used to compute the min and max used for later scaling along the features axis.
        
        y : None
            Ignored.

        Returns
        -------
        self : object
            Fitted scaler.
        """
        if not isinstance(X, np.ndarray):
            raise TypeError("X must be a numpy array.")
        
        self.min = np.min(X, axis=0)
        self.max = np.max(X, axis=0)
        return self

    def transform(self, X):
        """
        Perform normalization by scaling to the range [0, 1].

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data used to scale along the features axis.

        Returns
        -------
        X_transformed : ndarray of shape (n_samples, n_features)
            Transformed array.
        """
        if self.min is None or self.max is None:
            raise ValueError("Scaler has not been fitted.")
        if not isinstance(X, np.ndarray):
            raise TypeError("X must be a numpy array.")
        
        return (X - self.min) / (self.max - self.min)

    def fit_transform(self, X, y=None):
        """
        Fit to data, then transform it.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples.
        
        y : array-like of shape (n_samples,) or (n_samples, n_outputs), default=None
            Target values (ignored).

        Returns
        -------
        X_transformed : ndarray of shape (n_samples, n_features)
            Transformed array.
        """
        self.fit(X)
        return self.transform(X)
