from supervised_learning.classification.DecisionTreeClassifier import DecisionTreeClassifier
import numpy as np

class AdaBoost:
    def __init__(self, n_estimators=50, learning_rate=1.0):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.models = []
        self.model_weights = []

    def fit(self, X, y):
        n_samples, n_features = X.shape
        sample_weights = np.full(n_samples, 1 / n_samples)

        for _ in range(self.n_estimators):
            model = DecisionTreeClassifier(max_depth=1)
            model.fit(X, y)
            model_errors = (model.predict(X) != y).astype(int)
            model_error = np.dot(sample_weights, model_errors)

            if model_error >= 0.5:
                break

            model_weight = self.learning_rate * 0.5 * np.log((1 - model_error) / model_error)
            self.models.append(model)
            self.model_weights.append(model_weight)

            # Update sample weights
            sample_weights *= np.exp(-model_weight * y * model.predict(X))
            sample_weights /= np.sum(sample_weights)

    def predict(self, X):
        model_preds = np.array([model.predict(X) for model in self.models])
        weighted_preds = np.dot(self.model_weights, model_preds)
        return np.sign(weighted_preds)

    def score(self, X, y):
        predictions = self.predict(X)
        return np.mean(predictions == y)
