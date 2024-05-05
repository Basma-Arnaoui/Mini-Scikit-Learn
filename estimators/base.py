class BaseEstimator:
    def fit(self, X, y=None):
        """
        Fit the estimator to the given data.

        Parameters:
        - X : array-like, shape (n_samples, n_features)
            The training input samples.
        - y : array-like, shape (n_samples,), optional
            The target values. For unsupervised learning, y is not required.

        Returns:
        - self : object
            Returns self.
        """
        raise NotImplementedError("fit method is not implemented")

    def get_params(self, deep=True):
        """
        Get parameters for this estimator.

        Parameters:
        - deep : bool, default=True
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.

        Returns:
        - params : dict
            Parameter names mapped to their values.
        """
        raise NotImplementedError("get_params method is not implemented")

    def set_params(self, **params):
        """
        Set the parameters of this estimator.

        Parameters:
        - **params : dict
            Estimator parameters.

        Returns:
        - self : object
            Returns self.
        """
        raise NotImplementedError("set_params method is not implemented")
