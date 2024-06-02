import numpy as np
from sklearn.tree import DecisionTreeRegressor

class GradientBoostingClassifier:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3, random_state=None):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.random_state = random_state
        self.trees = []
        self.initial_prediction = None

    def fit(self, X, y):
        # Convert labels to {-1, 1}
        y = np.where(y == 0, -1, 1)
        # Initial prediction (mean)
        self.initial_prediction = np.log(np.mean(y == 1) / np.mean(y == -1))
        F = np.full(y.shape, self.initial_prediction)
        
        for _ in range(self.n_estimators):
            residuals = y - np.tanh(F)
            tree = DecisionTreeRegressor(max_depth=self.max_depth, random_state=self.random_state)
            tree.fit(X, residuals)
            predictions = tree.predict(X)
            F += self.learning_rate * predictions
            self.trees.append(tree)

    def predict(self, X):
        F = np.full(X.shape[0], self.initial_prediction)
        for tree in self.trees:
            F += self.learning_rate * tree.predict(X)
        return np.where(np.tanh(F) >= 0, 1, 0)

    def predict_proba(self, X):
        F = np.full(X.shape[0], self.initial_prediction)
        for tree in self.trees:
            F += self.learning_rate * tree.predict(X)
        probs = np.tanh(F)
        return np.vstack([1-probs, probs]).T
