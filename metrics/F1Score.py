from .BaseMetric import BaseMetric
from .Precision import Precision
from .Recall import Recall

class F1Score(BaseMetric):
    def score(self, y_true, y_pred):
        precision = Precision().score(y_true, y_pred)
        recall = Recall().score(y_true, y_pred)
        if precision + recall == 0:
            return 0
        return 2 * (precision * recall) / (precision + recall)
