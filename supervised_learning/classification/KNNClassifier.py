from supervised_learning.BaseEstimator import BaseEstimator
import numpy as np
import pandas as pd

class KNNClassifier(BaseEstimator):
    def __init__(self, n_neighbors=5):
        self.n_neighbors = n_neighbors

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        y_pred = []
        for x in X:
            # Calculate Euclidean distances between the current point x and all training points
            distances = np.sqrt(np.sum((self.X_train - x) ** 2, axis=1))
            # Find the indices of the k-nearest neighbors
            nearest_indices = np.argsort(distances)[:self.n_neighbors]
            # Get the corresponding labels of the nearest neighbors
            nearest_labels = self.y_train[nearest_indices]
            # Count the occurrences of each label
            unique_labels, label_counts = np.unique(nearest_labels, return_counts=True)
            # Predict the label with the highest count
            predicted_label = unique_labels[np.argmax(label_counts)]
            y_pred.append(predicted_label)
        return np.array(y_pred)
