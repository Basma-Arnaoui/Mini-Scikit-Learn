from .BaseMetric import BaseMetric
import numpy as np # type: ignore

class MeanAbsoluteError(BaseMetric):
    def score(self, y_true, y_pred):
        return np.mean(np.abs(y_true - y_pred))
