from .BaseMetric import BaseMetric

class Recall(BaseMetric):
    def score(self, y_true, y_pred):
        true_positive = sum((y_pred == 1) & (y_true == 1))
        actual_positive = sum(y_true == 1)
        return true_positive / actual_positive if actual_positive != 0 else 0
