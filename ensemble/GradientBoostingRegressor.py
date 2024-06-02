import numpy as np
from sklearn.tree import DecisionTreeRegressor

class GradientBoostingRegressor:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3, random_state=None):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.random_state = random_state
        self.trees = []
        self.initial_prediction = None

    def fit(self, X, y):
        # Initial prediction (mean)
        self.initial_prediction = np.mean(y)
        F = np.full(y.shape, self.initial_prediction)
        
        for _ in range(self.n_estimators):
            residuals = y - F
            tree = DecisionTreeRegressor(max_depth=self.max_depth, random_state=self.random_state)
            tree.fit(X, residuals)
            predictions = tree.predict(X)
            F += self.learning_rate * predictions
            self.trees.append(tree)

    def predict(self, X):
        F = np.full(X.shape[0], self.initial_prediction)
        for tree in self.trees:
            F += self.learning_rate * tree.predict(X)
        return F
