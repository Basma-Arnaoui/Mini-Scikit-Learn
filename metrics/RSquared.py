from .BaseMetric import BaseMetric
import numpy as np  # type: ignore

class RSquared(BaseMetric):
    """
    R-squared (Coefficient of Determination) metric for regression tasks.

    Methods
    -------
    score(y_true, y_pred)
        Calculate the R-squared score.
    """

    def score(self, y_true, y_pred):
        """
        Calculate the R-squared score.

        Parameters
        ----------
        y_true : array-like of shape (n_samples,)
            True values.
        
        y_pred : array-like of shape (n_samples,)
            Predicted values.

        Returns
        -------
        r_squared : float
            The R-squared score, defined as 1 minus the ratio of the explained variance to the total variance.
        """
        total_variance = np.var(y_true)
        explained_variance = np.var(y_true - y_pred)
        return 1 - explained_variance / total_variance
