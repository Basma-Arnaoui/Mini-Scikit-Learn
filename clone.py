from copy import deepcopy

def clone(estimator, safe=True):
    """Constructs a new unfitted estimator with the same parameters."""
    if not hasattr(estimator, 'get_params') or not callable(getattr(estimator, 'get_params')):
        if safe:
            raise TypeError("Cannot clone object '%s' (type %s): "
                            "it does not seem to be a scikit-learn estimator as it does not implement 'get_params'." % 
                            (estimator, type(estimator)))
        return deepcopy(estimator)
    
    klass = estimator.__class__
    new_object = klass(**estimator.get_params())
    return new_object
