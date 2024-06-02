from .MeanSquaredError import MeanSquaredError
import numpy as np  # type: ignore

class RootMeanSquaredError(MeanSquaredError):
    """
    Root Mean Squared Error (RMSE) metric for regression tasks.

    Methods
    -------
    score(y_true, y_pred)
        Calculate the root mean squared error.
    """

    def score(self, y_true, y_pred):
        """
        Calculate the root mean squared error.

        Parameters
        ----------
        y_true : array-like of shape (n_samples,)
            True values.
        
        y_pred : array-like of shape (n_samples,)
            Predicted values.

        Returns
        -------
        rmse : float
            The root mean squared error of the predictions.
        """
        mse = super().score(y_true, y_pred)
        return np.sqrt(mse)
