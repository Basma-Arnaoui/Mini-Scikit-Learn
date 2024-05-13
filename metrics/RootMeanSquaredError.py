from .MeanSquaredError import MeanSquaredError
import numpy as np # type: ignore

class RootMeanSquaredError(MeanSquaredError):
    def score(self, y_true, y_pred):
        mse = super().score(y_true, y_pred)
        return np.sqrt(mse)
