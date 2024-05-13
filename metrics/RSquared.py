from .BaseMetric import BaseMetric
import numpy as np # type: ignore

class RSquared(BaseMetric):
    def score(self, y_true, y_pred):
        total_variance = np.var(y_true)
        explained_variance = np.var(y_true - y_pred)
        return 1 - explained_variance / total_variance
