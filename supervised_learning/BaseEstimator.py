from abc import ABCMeta, abstractmethod

class BaseEstimator(metaclass=ABCMeta):
    @abstractmethod
    def fit(self, X, y):
        pass

    @abstractmethod
    def predict(self, X):
        pass
