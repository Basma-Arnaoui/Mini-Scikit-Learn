from .BaseMetric import BaseMetric

class Accuracy(BaseMetric):
    """
    Accuracy metric for classification tasks.

    Methods
    -------
    score(y_true, y_pred)
        Calculate the accuracy of predictions.
    """

    def score(self, y_true, y_pred):
        """
        Calculate the accuracy of predictions.

        Parameters
        ----------
        y_true : array-like of shape (n_samples,)
            True class labels.
        
        y_pred : array-like of shape (n_samples,)
            Predicted class labels.

        Returns
        -------
        accuracy : float
            The accuracy of the predictions, defined as the ratio of correctly predicted
            samples to the total number of samples.
        """
        correct = sum(y_true == y_pred)
        total = len(y_true)
        return correct / total
