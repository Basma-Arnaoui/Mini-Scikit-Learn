from .BaseMetric import BaseMetric
from .Precision import Precision
from .Recall import Recall

class F1Score(BaseMetric):
    """
    F1 Score metric for classification tasks.

    Methods
    -------
    score(y_true, y_pred)
        Calculate the F1 score.
    """

    def score(self, y_true, y_pred):
        """
        Calculate the F1 score.

        Parameters
        ----------
        y_true : array-like of shape (n_samples,)
            True class labels.
        
        y_pred : array-like of shape (n_samples,)
            Predicted class labels.

        Returns
        -------
        f1_score : float
            The F1 score, defined as the harmonic mean of precision and recall.
        """
        precision = Precision().score(y_true, y_pred)
        recall = Recall().score(y_true, y_pred)
        if precision + recall == 0:
            return 0
        return 2 * (precision * recall) / (precision + recall)
