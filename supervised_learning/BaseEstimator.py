from abc import ABCMeta, abstractmethod

class BaseEstimator(metaclass=ABCMeta):
    """
    Base class for all estimators in the supervised learning module.

    Methods
    -------
    fit(X, y)
        Abstract method to fit the model to the training data.
    
    predict(X)
        Abstract method to predict target values for samples in X.
    
    get_params(deep=True)
        Get parameters for this estimator.
    
    set_params(**params)
        Set the parameters of this estimator.
    """

    @abstractmethod
    def fit(self, X, y):
        pass

    @abstractmethod
    def predict(self, X):
        pass
    
    def __init__(self, **kwargs):
        self.set_params(**kwargs)

    def get_params(self, deep=True):
        """
        Get parameters for this estimator.

        Returns
        -------
        params : dict
            Parameter names mapped to their values.
        """
        params = {}
        for key in self._get_param_names():
            params[key] = getattr(self, key)
        return params

    def set_params(self, **params):
        """
        Set the parameters of this estimator.

        Parameters
        ----------
        **params : dict
            Estimator parameters.

        Returns
        -------
        self : object
            Estimator instance.
        """
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def _get_param_names(self):
        """
        Get parameter names for the estimator.

        Returns
        -------
        param_names : list of str
            List of parameter names.
        """
        import inspect
        class_signature = inspect.signature(self.__init__)
        return [p.name for p in class_signature.parameters.values() if p.name != 'self']
