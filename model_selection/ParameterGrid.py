import itertools

class ParameterGrid:
    def __init__(self, param_grid):
        self.param_grid = param_grid

    def __iter__(self):
        """Yield the next configuration of parameters."""
        keys, values = zip(*sorted(self.param_grid.items()))
        for v in itertools.product(*values):
            params = dict(zip(keys, v))
            yield params
