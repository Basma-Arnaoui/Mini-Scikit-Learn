import numpy as np

class MLP:
    """
    Multi-Layer Perceptron (MLP) for classification tasks.

    Parameters
    ----------
    layer_sizes : list
        List containing the number of neurons in each layer, including input and output layers.
    
    learning_rate : float, default=0.1
        The learning rate for weight updates.
    
    epochs : int, default=200
        Number of epochs for training.

    Methods
    -------
    fit(X, y)
        Train the MLP on the given data.
    
    predict(X)
        Predict class labels for samples in X.
    
    score(X, y)
        Returns the mean accuracy on the given test data and labels.
    """

    def __init__(self, layer_sizes, learning_rate=0.1, epochs=200):
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = []
        self.biases = []
        self._init_weights()

    def _init_weights(self):
        for i in range(len(self.layer_sizes) - 1):
            weight = np.random.randn(self.layer_sizes[i], self.layer_sizes[i + 1]) * 0.01
            bias = np.zeros((1, self.layer_sizes[i + 1]))
            self.weights.append(weight)
            self.biases.append(bias)

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def _sigmoid_derivative(self, z):
        return z * (1 - z)

    def fit(self, X, y):
        for epoch in range(self.epochs):
            for x_instance, y_instance in zip(X, y):
                activations = self._forward_pass(x_instance)
                self._backward_pass(y_instance, activations)

    def _forward_pass(self, x_instance):
        activations = [x_instance.reshape(1, -1)]
        for i in range(len(self.weights)):
            z = np.dot(activations[-1], self.weights[i]) + self.biases[i]
            activation = self._sigmoid(z)
            activations.append(activation)
        return activations

    def _backward_pass(self, y_instance, activations):
        labels = np.zeros((1, self.layer_sizes[-1]))
        labels[0, y_instance] = 1
        error = activations[-1] - labels
        for i in reversed(range(len(self.weights))):
            delta = error * self._sigmoid_derivative(activations[i + 1])
            error = np.dot(delta, self.weights[i].T)
            self.weights[i] -= self.learning_rate * np.dot(activations[i].T, delta)
            self.biases[i] -= self.learning_rate * np.sum(delta, axis=0, keepdims=True)

    def predict(self, X):
        results = []
        for x_instance in X:
            activations = self._forward_pass(x_instance)
            result = np.argmax(activations[-1])
            results.append(result)
        return np.array(results)

    def score(self, X, y):
        predictions = self.predict(X)
        return np.mean(predictions == y)
