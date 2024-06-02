from .BaseMetric import BaseMetric
import numpy as np  # type: ignore

class ConfusionMatrix(BaseMetric):
    """
    Confusion Matrix metric for classification tasks.

    Methods
    -------
    score(y_true, y_pred)
        Calculate the confusion matrix.
    """

    def score(self, y_true, y_pred):
        """
        Calculate the confusion matrix.

        Parameters
        ----------
        y_true : array-like of shape (n_samples,)
            True class labels.
        
        y_pred : array-like of shape (n_samples,)
            Predicted class labels.

        Returns
        -------
        matrix : ndarray of shape (n_classes, n_classes)
            Confusion matrix where rows represent true classes and columns represent predicted classes.
        """
        unique_classes = np.unique(y_true)
        matrix = np.zeros((len(unique_classes), len(unique_classes)), dtype=int)
        for i, true_class in enumerate(unique_classes):
            for j, pred_class in enumerate(unique_classes):
                matrix[i, j] = np.sum((y_true == true_class) & (y_pred == pred_class))
        return matrix
