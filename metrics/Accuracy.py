from .BaseMetric import BaseMetric

class Accuracy(BaseMetric):
    def score(self, y_true, y_pred):
        """Calculate the accuracy of predictions."""
        correct = sum(y_true == y_pred)
        total = len(y_true)
        return correct / total
