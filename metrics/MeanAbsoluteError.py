from .BaseMetric import BaseMetric
import numpy as np  # type: ignore

class MeanAbsoluteError(BaseMetric):
    """
    Mean Absolute Error (MAE) metric for regression tasks.

    Methods
    -------
    score(y_true, y_pred)
        Calculate the mean absolute error.
    """

    def score(self, y_true, y_pred):
        """
        Calculate the mean absolute error.

        Parameters
        ----------
        y_true : array-like of shape (n_samples,)
            True values.
        
        y_pred : array-like of shape (n_samples,)
            Predicted values.

        Returns
        -------
        mae : float
            The mean absolute error of the predictions.
        """
        return np.mean(np.abs(y_true - y_pred))
