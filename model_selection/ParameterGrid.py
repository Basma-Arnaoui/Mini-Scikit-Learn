import itertools

class ParameterGrid:
    """
    Grid of parameters with a discrete number of values for each parameter.

    Parameters
    ----------
    param_grid : dict
        Dictionary with parameters names (`str`) as keys and lists of parameter settings to try as values.

    Methods
    -------
    __iter__()
        Yield the next configuration of parameters.
    """

    def __init__(self, param_grid):
        self.param_grid = param_grid

    def __iter__(self):
        """
        Yield the next configuration of parameters.

        Yields
        ------
        params : dict
            A dictionary containing a configuration of parameter settings.
        """
        keys, values = zip(*sorted(self.param_grid.items()))
        for v in itertools.product(*values):
            params = dict(zip(keys, v))
            yield params
