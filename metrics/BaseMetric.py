from abc import ABC, abstractmethod

class BaseMetric(ABC):
    @abstractmethod
    def score(self, y_true, y_pred):
        """Calculate the metric score."""
        pass