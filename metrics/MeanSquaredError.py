from .BaseMetric import BaseMetric
import numpy as np # type: ignore

class MeanSquaredError(BaseMetric):
    def score(self, y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)
