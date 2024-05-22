from abc import ABCMeta, abstractmethod

class BaseEstimator(metaclass=ABCMeta):
    @abstractmethod
    def fit(self, X, y):
        pass

    @abstractmethod
    def predict(self, X):
        pass
    
    def __init__(self, **kwargs):
        self.set_params(**kwargs)

    def get_params(self, deep=True):
        """Get parameters for this estimator."""
        params = {}
        for key in self._get_param_names():
            params[key] = getattr(self, key)
        return params

    def set_params(self, **params):
        """Set the parameters of this estimator."""
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def _get_param_names(self):
        """Get parameter names for the estimator"""
        # Introspection of the constructor arguments to get the parameter names
        import inspect
        class_signature = inspect.signature(self.__init__)
        return [p.name for p in class_signature.parameters.values() if p.name != 'self']
