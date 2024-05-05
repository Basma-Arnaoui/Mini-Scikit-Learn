class BasePredictor:
    def predict(self, X):
        """
        Predict the target labels for the input data X.

        Parameters:
        - X : array-like, shape (n_samples, n_features)
            The input samples.

        Returns:
        - y_pred : array-like, shape (n_samples,)
            The predicted target labels.
        """
        raise NotImplementedError("predict method is not implemented")

    def score(self, X, y=None):
        """
        Return the mean accuracy on the given test data and labels.

        Parameters:
        - X : array-like, shape (n_samples, n_features)
            Test samples.
        - y : array-like, shape (n_samples,), default=None
            True labels for X.

        Returns:
        - score : float
            Mean accuracy of self.predict(X) with respect to y.
        """
        raise NotImplementedError("score method is not implemented")
