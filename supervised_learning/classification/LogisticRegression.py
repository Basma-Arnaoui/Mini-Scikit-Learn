import numpy as np

class LogisticRegression:
    def __init__(self, learning_rate=0.01, n_iterations=1000, regularization=None, lambda_param=0.1):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.regularization = regularization
        self.lambda_param = lambda_param
        self.weights = None
        self.bias = None

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def _initialize_parameters(self, n_features):
        self.weights = np.zeros(n_features)
        self.bias = 0

    def _compute_cost(self, h, y):
        m = len(y)
        if self.regularization == 'l2':
            reg_term = (self.lambda_param / (2 * m)) * np.sum(np.square(self.weights))
        elif self.regularization == 'l1':
            reg_term = (self.lambda_param / (2 * m)) * np.sum(np.abs(self.weights))
        else:
            reg_term = 0
        return (-1 / m) * np.sum((y * np.log(h)) + ((1 - y) * np.log(1 - h))) + reg_term

    def fit(self, X, y):
        m, n = X.shape
        self._initialize_parameters(n)

        for _ in range(self.n_iterations):
            z = np.dot(X, self.weights) + self.bias
            h = self._sigmoid(z)

            gradient_weights = (1 / m) * np.dot(X.T, (h - y))
            gradient_bias = (1 / m) * np.sum(h - y)

            if self.regularization == 'l2':
                gradient_weights += (self.lambda_param / m) * self.weights
            elif self.regularization == 'l1':
                gradient_weights += (self.lambda_param / m) * np.sign(self.weights)

            self.weights -= self.learning_rate * gradient_weights
            self.bias -= self.learning_rate * gradient_bias

            if _ % 100 == 0:
                cost = self._compute_cost(h, y)
                print(f"Cost after iteration {_}: {cost}")

    def predict_proba(self, X):
        z = np.dot(X, self.weights) + self.bias
        return self._sigmoid(z)

    def predict(self, X, threshold=0.5):
        return np.where(self.predict_proba(X) >= threshold, 1, 0)
