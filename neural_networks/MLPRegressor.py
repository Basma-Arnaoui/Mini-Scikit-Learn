import numpy as np

class MLPRegressor:
    def __init__(self, hidden_layer_sizes=(100,), activation='relu', learning_rate=0.001, max_iter=200, tol=1e-4):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol
        self.weights = []
        self.biases = []
        
    def _initialize_weights(self, layer_sizes):
        for i in range(len(layer_sizes) - 1):
            limit = np.sqrt(6 / (layer_sizes[i] + layer_sizes[i+1]))
            weight = np.random.uniform(-limit, limit, (layer_sizes[i], layer_sizes[i+1]))
            bias = np.zeros((1, layer_sizes[i+1]))
            self.weights.append(weight)
            self.biases.append(bias)
    
    def _forward_pass(self, X):
        activations = [X]
        for weight, bias in zip(self.weights, self.biases):
            net_input = np.dot(activations[-1], weight) + bias
            if self.activation == 'relu':
                activation = np.maximum(0, net_input)
            elif self.activation == 'tanh':
                activation = np.tanh(net_input)
            elif self.activation == 'sigmoid':
                activation = 1 / (1 + np.exp(-net_input))
            else:
                activation = net_input  # linear activation
            activations.append(activation)
        return activations
    
    def _compute_loss(self, y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)
    
    def _backward_pass(self, activations, y):
        grad_weights = [np.zeros_like(w) for w in self.weights]
        grad_biases = [np.zeros_like(b) for b in self.biases]
        error = activations[-1] - y
        
        for i in reversed(range(len(self.weights))):
            if self.activation == 'relu':
                grad_activation = activations[i+1] > 0
            elif self.activation == 'tanh':
                grad_activation = 1 - np.square(activations[i+1])
            elif self.activation == 'sigmoid':
                grad_activation = activations[i+1] * (1 - activations[i+1])
            else:
                grad_activation = 1  # derivative of linear is 1

            error *= grad_activation
            grad_biases[i] = np.mean(error, axis=0, keepdims=True)
            grad_weights[i] = np.dot(activations[i].T, error) / len(activations[i])

            error = np.dot(error, self.weights[i].T)
        
        return grad_weights, grad_biases
    
    def fit(self, X, y):
        layer_sizes = [X.shape[1]] + list(self.hidden_layer_sizes) + [y.shape[1]]
        self._initialize_weights(layer_sizes)
        
        for _ in range(self.max_iter):
            activations = self._forward_pass(X)
            loss = self._compute_loss(y, activations[-1])
            grad_weights, grad_biases = self._backward_pass(activations, y)
            
            for i in range(len(self.weights)):
                self.weights[i] -= self.learning_rate * grad_weights[i]
                self.biases[i] -= self.learning_rate * grad_biases[i]
            
            if loss < self.tol:
                break

    def predict(self, X):
        activations = self._forward_pass(X)
        return activations[-1]
