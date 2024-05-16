import numpy as np
from itertools import product

class GridSearchCV:
    def __init__(self, estimator, param_grid, cv=5, scoring='accuracy'):
        self.estimator = estimator
        self.param_grid = param_grid
        self.cv = cv
        self.scoring = scoring
        self.best_params_ = None
        self.best_score_ = -np.inf
        self.results_ = []  # Initialize here to ensure it's always defined

    def fit(self, X, y):
        for params in self._param_grid_combinations():
            scores = []
            for train_idx, test_idx in self._k_fold(len(y), self.cv):
                self.estimator.set_params(**params)
                self.estimator.fit(X[train_idx], y[train_idx])
                pred = self.estimator.predict(X[test_idx])
                score = self._calculate_score(y[test_idx], pred)
                scores.append(score)

            average_score = np.mean(scores)
            self.results_.append({'params': params, 'score': average_score})
            if average_score > self.best_score_:
                self.best_score_ = average_score
                self.best_params_ = params

        # Sort results after all scores are computed
        self.results_.sort(key=lambda x: x['score'], reverse=True)
        return self

    def _param_grid_combinations(self):
        keys, values = zip(*self.param_grid.items())
        for v in product(*values):
            yield dict(zip(keys, v))

    def _k_fold(self, n, n_folds):
        indices = np.arange(n)
        np.random.shuffle(indices)
        fold_sizes = np.full(n_folds, n // n_folds, dtype=int)
        fold_sizes[:n % n_folds] += 1
        current = 0
        for fold_size in fold_sizes:
            start, stop = current, current + fold_size
            train_indices = np.concatenate([indices[:start], indices[stop:]])
            test_indices = indices[start:stop]
            yield train_indices, test_indices

    def _calculate_score(self, y_true, y_pred):
        if self.scoring == 'accuracy':
            return np.mean(y_true == y_pred)
