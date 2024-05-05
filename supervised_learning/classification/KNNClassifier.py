from supervised_learning.BaseEstimator import BaseEstimator
import numpy as np
from collections import Counter

class KNNClassifier(BaseEstimator):
    def __init__(self, n_neighbors=5):
        self.n_neighbors = n_neighbors
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        predictions = [self._predict_single(x) for x in X]
        return np.array(predictions)

    def _predict_single(self, x):
        # Compute distances from x to all training data
        distances = np.linalg.norm(self.X_train - x, axis=1)
        
        # Find the nearest neighbors
        nearest_indices = np.argsort(distances)[:self.n_neighbors]
        nearest_labels = self.y_train[nearest_indices]
        
        # Majority vote
        most_common = Counter(nearest_labels).most_common(1)
        return most_common[0][0]
