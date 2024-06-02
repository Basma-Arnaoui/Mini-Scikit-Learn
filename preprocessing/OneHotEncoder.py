import numpy as np

class OneHotEncoder:
    """
    Encode categorical integer features as a one-hot numeric array.

    Methods
    -------
    fit(X)
        Fit OneHotEncoder to X data.
    
    transform(X)
        Transform X using one-hot encoding.
    
    fit_transform(X)
        Fit and transform X with one-hot encoding.
    """

    def __init__(self):
        self.classes_ = {}

    def fit(self, X):
        """
        Fit OneHotEncoder to X data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input data.

        Returns
        -------
        self : object
            Fitted encoder.
        """
        for idx in range(X.shape[1]):
            unique = np.unique(X[:, idx])
            self.classes_[idx] = unique
        return self

    def transform(self, X):
        """
        Transform X using one-hot encoding.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input data.

        Returns
        -------
        X_transformed : array-like of shape (n_samples, n_encoded_features)
            The one-hot encoded data.
        """
        output_rows = []
        for row in X:
            encoded_row = []
            for idx, item in enumerate(row):
                template = np.zeros(len(self.classes_[idx]))
                template[np.where(self.classes_[idx] == item)[0][0]] = 1
                encoded_row.extend(template)
            output_rows.append(encoded_row)
        return np.array(output_rows)

    def fit_transform(self, X):
        """
        Fit and transform X with one-hot encoding.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input data.

        Returns
        -------
        X_transformed : array-like of shape (n_samples, n_encoded_features)
            The one-hot encoded data.
        """
        return self.fit(X).transform(X)
