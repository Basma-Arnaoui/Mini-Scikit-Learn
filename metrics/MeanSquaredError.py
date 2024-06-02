from .BaseMetric import BaseMetric
import numpy as np  # type: ignore

class MeanSquaredError(BaseMetric):
    """
    Mean Squared Error (MSE) metric for regression tasks.

    Methods
    -------
    score(y_true, y_pred)
        Calculate the mean squared error.
    """

    def score(self, y_true, y_pred):
        """
        Calculate the mean squared error.

        Parameters
        ----------
        y_true : array-like of shape (n_samples,)
            True values.
        
        y_pred : array-like of shape (n_samples,)
            Predicted values.

        Returns
        -------
        mse : float
            The mean squared error of the predictions.
        """
        return np.mean((y_true - y_pred) ** 2)
