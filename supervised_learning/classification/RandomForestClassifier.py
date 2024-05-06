import numpy as np
from supervised_learning.BaseEstimator import BaseEstimator
from supervised_learning.classification.DecisionTreeClassifier import DecisionTreeClassifier
from collections import Counter

class RandomForestClassifier(BaseEstimator):
    def __init__(self, n_estimators=10, max_depth=2, min_samples_split=2, max_features=None, random_state=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.random_state = random_state
        self.trees = []

    def fit(self, X, y):
        np.random.seed(self.random_state)
        for _ in range(self.n_estimators):
            tree = DecisionTreeClassifier(max_depth=self.max_depth, min_samples_split=self.min_samples_split)
            X_sample, y_sample = self._bootstrap_samples(X, y)
            y_sample = y_sample.reshape(-1, 1)  # Ensure y has 2 dimensions
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def _bootstrap_samples(self, X, y):
        n_samples = X.shape[0]
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        return X[indices], y[indices]

    def predict(self, X):
        predictions = np.array([tree.predict(X) for tree in self.trees])
        return np.array([self._most_common_label(predictions[:, i]) for i in range(X.shape[0])])

    def _most_common_label(self, y):
        counter = Counter(y)
        most_common = counter.most_common(1)[0][0]
        return most_common
