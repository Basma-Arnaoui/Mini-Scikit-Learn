from .BaseMetric import BaseMetric
import numpy as np # type: ignore

class ConfusionMatrix(BaseMetric):
    def score(self, y_true, y_pred):
        unique_classes = np.unique(y_true)
        matrix = np.zeros((len(unique_classes), len(unique_classes)), dtype=int)
        for i, true_class in enumerate(unique_classes):
            for j, pred_class in enumerate(unique_classes):
                matrix[i, j] = np.sum((y_true == true_class) & (y_pred == pred_class))
        return matrix
