from .BaseMetric import BaseMetric

class Precision(BaseMetric):
    def score(self, y_true, y_pred):
        true_positive = sum((y_pred == 1) & (y_true == 1))
        predicted_positive = sum(y_pred == 1)
        return true_positive / predicted_positive if predicted_positive != 0 else 0
