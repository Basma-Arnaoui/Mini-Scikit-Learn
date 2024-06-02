from .BaseMetric import BaseMetric

class Recall(BaseMetric):
    """
    Recall metric for classification tasks.

    Methods
    -------
    score(y_true, y_pred)
        Calculate the recall score.
    """

    def score(self, y_true, y_pred):
        """
        Calculate the recall score.

        Parameters
        ----------
        y_true : array-like of shape (n_samples,)
            True class labels.
        
        y_pred : array-like of shape (n_samples,)
            Predicted class labels.

        Returns
        -------
        recall : float
            The recall score, defined as the ratio of true positive predictions to the total number of actual positive instances.
        """
        true_positive = sum((y_pred == 1) & (y_true == 1))
        actual_positive = sum(y_true == 1)
        return true_positive / actual_positive if actual_positive != 0 else 0
