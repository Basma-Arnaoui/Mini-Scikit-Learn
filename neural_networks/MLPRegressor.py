import numpy as np

class MLPRegressor:
    """
    Multi-Layer Perceptron (MLP) for regression tasks.

    Parameters
    ----------
    hidden_layer_sizes : tuple, default=(100,)
        The ith element represents the number of neurons in the ith hidden layer.
    
    activation : {'relu', 'tanh', 'sigmoid', 'linear'}, default='relu'
        Activation function for the hidden layers.
    
    learning_rate : float, default=0.001
        The learning rate for weight updates.
    
    max_iter : int, default=200
        Maximum number of iterations.
    
    tol : float, default=1e-4
        Tolerance for the optimization. Training stops when the loss is below this threshold.

    Methods
    -------
    fit(X, y)
        Train the MLPRegressor on the given data.
    
    predict(X)
        Predict target values for samples in X.
    """

    def __init__(self, hidden_layer_sizes=(100,), activation='relu', learning_rate=0.001, max_iter=200, tol=1e-4):
        if not isinstance(hidden_layer_sizes, tuple) or not all(isinstance(i, int) for i in hidden_layer_sizes):
            raise ValueError("hidden_layer_sizes must be a tuple of integers.")
        if activation not in {'relu', 'tanh', 'sigmoid', 'linear'}:
            raise ValueError("activation must be one of {'relu', 'tanh', 'sigmoid', 'linear'}.")
        if not isinstance(learning_rate, float) or learning_rate <= 0:
            raise ValueError("learning_rate must be a positive float.")
        if not isinstance(max_iter, int) or max_iter <= 0:
            raise ValueError("max_iter must be a positive integer.")
        if not isinstance(tol, float) or tol <= 0:
            raise ValueError("tol must be a positive float.")

        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol
        self.weights = []
        self.biases = []

    def _initialize_weights(self, layer_sizes):
        """
        Initialize weights and biases for the neural network.

        Parameters
        ----------
        layer_sizes : list
            List containing the number of neurons in each layer, including input and output layers.
        """
        for i in range(len(layer_sizes) - 1):
            limit = np.sqrt(6 / (layer_sizes[i] + layer_sizes[i+1]))
            weight = np.random.uniform(-limit, limit, (layer_sizes[i], layer_sizes[i+1]))
            bias = np.zeros((1, layer_sizes[i+1]))
            self.weights.append(weight)
            self.biases.append(bias)

    def _forward_pass(self, X):
        """
        Perform a forward pass through the network.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input data.

        Returns
        -------
        activations : list
            List of activations for each layer.
        """
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
        """
        Compute the loss (mean squared error).

        Parameters
        ----------
        y_true : array-like of shape (n_samples,)
            True target values.
        
        y_pred : array-like of shape (n_samples,)
            Predicted target values.

        Returns
        -------
        loss : float
            The mean squared error of the predictions.
        """
        return np.mean((y_true - y_pred) ** 2)

    def _backward_pass(self, activations, y):
        """
        Perform a backward pass through the network (backpropagation).

        Parameters
        ----------
        activations : list
            List of activations for each layer.
        
        y : array-like of shape (n_samples,)
            True target values.

        Returns
        -------
        grad_weights : list
            Gradients for the weights.
        
        grad_biases : list
            Gradients for the biases.
        """
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
        """
        Train the MLPRegressor on the given data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.
        
        y : array-like of shape (n_samples,)
            The target values.
        """
        if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
            raise TypeError("X and y must be numpy arrays.")
        if X.shape[0] != y.shape[0]:
            raise ValueError("The number of samples in X and y must be equal.")

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
        """
        Predict target values for samples in X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        y_pred : array-like of shape (n_samples,)
            The predicted target values.
        """
        if not isinstance(X, np.ndarray):
            raise TypeError("X must be a numpy array.")
        
        activations = self._forward_pass(X)
        return activations[-1]
