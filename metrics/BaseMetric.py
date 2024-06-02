from abc import ABC, abstractmethod

class BaseMetric(ABC):
    """
    Base class for all metrics.

    Methods
    -------
    score(y_true, y_pred)
        Calculate the metric score.
    """

    @abstractmethod
    def score(self, y_true, y_pred):
        """
        Calculate the metric score.

        Parameters
        ----------
        y_true : array-like of shape (n_samples,)
            True class labels or values.
        
        y_pred : array-like of shape (n_samples,)
            Predicted class labels or values.

        Returns
        -------
        score : float
            The calculated metric score.
        """
        pass
