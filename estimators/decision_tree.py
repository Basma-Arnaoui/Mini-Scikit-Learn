import numpy as np

from estimators import BaseEstimator
from predictors import BasePredictor

class MyDecisionTreeClassifier(BaseEstimator, BasePredictor):
    def __init__(self, max_depth=None, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None

    def fit(self, X, y):
        self.tree = self._build_tree(X, y, depth=0)
        return self

    def predict(self, X):
        if self.tree is None:
            raise Exception("Estimator is not fitted, call `fit` before `predict`.")
        return np.array([self._predict_tree(x, self.tree) for x in X])

    def _build_tree(self, X, y, depth):
        n_samples, n_features = X.shape
        unique_classes = np.unique(y)

        # Base cases
        if depth == self.max_depth or len(unique_classes) == 1 or n_samples < self.min_samples_split:
            return {'class': max(unique_classes, key=list(y).count)}

        # Find the best split
        best_feature, best_threshold = self._find_best_split(X, y)

        # Split the dataset
        left_indices = X[:, best_feature] < best_threshold
        X_left, y_left = X[left_indices], y[left_indices]
        X_right, y_right = X[~left_indices], y[~left_indices]

        # Recursively build the tree
        left_tree = self._build_tree(X_left, y_left, depth + 1)
        right_tree = self._build_tree(X_right, y_right, depth + 1)

        return {'feature': best_feature, 'threshold': best_threshold,
                'left': left_tree, 'right': right_tree}

    def _find_best_split(self, X, y):
        n_samples, n_features = X.shape
        best_gini = float('inf')
        best_feature, best_threshold = None, None

        for feature_idx in range(n_features):
            feature_values = np.unique(X[:, feature_idx])
            for threshold in feature_values:
                left_indices = X[:, feature_idx] < threshold
                gini = self._calculate_gini(y[left_indices], y[~left_indices])
                if gini < best_gini:
                    best_gini = gini
                    best_feature = feature_idx
                    best_threshold = threshold

        return best_feature, best_threshold

    def _calculate_gini(self, left_labels, right_labels):
        n_left, n_right = len(left_labels), len(right_labels)
        n_total = n_left + n_right
        gini_left = 1 - sum((np.sum(left_labels == c) / n_left) ** 2 for c in np.unique(left_labels))
        gini_right = 1 - sum((np.sum(right_labels == c) / n_right) ** 2 for c in np.unique(right_labels))
        gini = (n_left / n_total) * gini_left + (n_right / n_total) * gini_right
        return gini

    def _predict_tree(self, x, node):
        if 'class' in node:
            return node['class']
        if x[node['feature']] < node['threshold']:
            return self._predict_tree(x, node['left'])
        else:
            return self._predict_tree(x, node['right'])
