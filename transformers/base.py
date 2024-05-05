class BaseTransformer:
    def transform(self, X):
        """
        Transform the input data X.

        Parameters:
        - X : array-like, shape (n_samples, n_features)
            The input samples.

        Returns:
        - X_transformed : array-like, shape (n_samples, n_features')
            The transformed samples.
        """
        raise NotImplementedError("transform method is not implemented")

    def fit_transform(self, X, y=None):
        """
        Fit to data, then transform it.

        Parameters:
        - X : array-like, shape (n_samples, n_features)
            The input samples.
        - y : array-like, shape (n_samples,), default=None
            Ignored.

        Returns:
        - X_transformed : array-like, shape (n_samples, n_features')
            The transformed samples.
        """
        raise NotImplementedError("fit_transform method is not implemented")
