import numpy as np

class LabelEncoder:
    """
    Encode target labels with value between 0 and n_classes-1.

    Methods
    -------
    fit(y)
        Fit label encoder.
    
    transform(y)
        Transform labels to normalized encoding.
    
    fit_transform(y)
        Fit label encoder and return encoded labels.
    
    inverse_transform(y_encoded)
        Transform labels back to original encoding.
    """

    def __init__(self):
        self.classes_ = None
        self.class_to_index_ = {}

    def fit(self, y):
        """
        Fit label encoder.

        Parameters
        ----------
        y : array-like of shape (n_samples,)
            The target values.

        Returns
        -------
        self : object
            Fitted encoder.
        """
        self.classes_ = np.unique(y)
        self.class_to_index_ = {cls: idx for idx, cls in enumerate(self.classes_)}
        return self

    def transform(self, y):
        """
        Transform labels to normalized encoding.

        Parameters
        ----------
        y : array-like of shape (n_samples,)
            The target values.

        Returns
        -------
        y_encoded : array-like of shape (n_samples,)
            The encoded labels.
        """
        if self.classes_ is None:
            raise RuntimeError("LabelEncoder instance needs to be fitted before being used.")
        return np.array([self.class_to_index_[label] for label in y], dtype=int)

    def fit_transform(self, y):
        """
        Fit label encoder and return encoded labels.

        Parameters
        ----------
        y : array-like of shape (n_samples,)
            The target values.

        Returns
        -------
        y_encoded : array-like of shape (n_samples,)
            The encoded labels.
        """
        return self.fit(y).transform(y)

    def inverse_transform(self, y_encoded):
        """
        Transform labels back to original encoding.

        Parameters
        ----------
        y_encoded : array-like of shape (n_samples,)
            The encoded labels.

        Returns
        -------
        y : array-like of shape (n_samples,)
            The original labels.
        """
        index_to_class = {idx: cls for cls, idx in self.class_to_index_.items()}
        return np.array([index_to_class[idx] for idx in y_encoded], dtype=object)
