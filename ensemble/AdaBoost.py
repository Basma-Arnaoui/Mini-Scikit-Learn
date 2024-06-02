from supervised_learning.classification.DecisionTreeClassifier import DecisionTreeClassifier
import numpy as np

class AdaBoost:
    """
    AdaBoost (Adaptive Boosting) implementation using DecisionTreeClassifier as the base estimator.

    Parameters
    ----------
    n_estimators : int, default=50
        The maximum number of estimators at which boosting is terminated. In case of perfect fit, the learning procedure is stopped early.
    
    learning_rate : float, default=1.0
        Weight applied to each classifier at each boosting iteration. A higher learning rate increases the contribution of each classifier.

    Attributes
    ----------
    models : list
        List of fitted models (weak learners).
    
    model_weights : list
        List of weights for each fitted model.

    Methods
    -------
    fit(X, y)
        Build a boosted classifier from the training set (X, y).

    predict(X)
        Predict classes for X.
    
    score(X, y)
        Returns the mean accuracy on the given test data and labels.
    """

    def __init__(self, n_estimators=50, learning_rate=1.0):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.models = []
        self.model_weights = []

    def fit(self, X, y):
        """
        Build a boosted classifier from the training set (X, y).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.
        
        y : array-like of shape (n_samples,)
            The target values (class labels).
        """
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
        """
        Predict classes for X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        y_pred : array-like of shape (n_samples,)
            The predicted classes.
        """
        model_preds = np.array([model.predict(X) for model in self.models])
        weighted_preds = np.dot(self.model_weights, model_preds)
        return np.sign(weighted_preds)

    def score(self, X, y):
        """
        Returns the mean accuracy on the given test data and labels.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The test input samples.
        
        y : array-like of shape (n_samples,)
            The true labels for X.

        Returns
        -------
        score : float
            The mean accuracy of the model on the test data and labels.
        """
        predictions = self.predict(X)
        return np.mean(predictions == y)
