from .BaseMetric import BaseMetric

class Precision(BaseMetric):
    """
    Precision metric for classification tasks.

    Methods
    -------
    score(y_true, y_pred)
        Calculate the precision score.
    """

    def score(self, y_true, y_pred):
        """
        Calculate the precision score.

        Parameters
        ----------
        y_true : array-like of shape (n_samples,)
            True class labels.
        
        y_pred : array-like of shape (n_samples,)
            Predicted class labels.

        Returns
        -------
        precision : float
            The precision score, defined as the ratio of true positive predictions to the total number of positive predictions.
        """
        true_positive = sum((y_pred == 1) & (y_true == 1))
        predicted_positive = sum(y_pred == 1)
        return true_positive / predicted_positive if predicted_positive != 0 else 0
