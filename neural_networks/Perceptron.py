import numpy as np

class Perceptron:
    """
    Perceptron classifier.

    Parameters
    ----------
    learning_rate : float, default=0.01
        Learning rate (between 0.0 and 1.0).
    
    n_iters : int, default=1000
        The number of passes over the training dataset.

    Methods
    -------
    fit(X, y)
        Fit the Perceptron model to the training data.
    
    predict(X)
        Predict class labels for samples in X.
    """

    def __init__(self, learning_rate=0.01, n_iters=1000):
        if not isinstance(learning_rate, float) or learning_rate <= 0:
            raise ValueError("learning_rate must be a positive float.")
        if not isinstance(n_iters, int) or n_iters <= 0:
            raise ValueError("n_iters must be a positive integer.")

        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.activation_func = self._unit_step_func
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        """
        Fit the Perceptron model to the training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.
        
        y : array-like of shape (n_samples,)
            The target values (class labels).
        """
        if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
            raise TypeError("X and y must be numpy arrays.")
        if X.shape[0] != y.shape[0]:
            raise ValueError("The number of samples in X and y must be equal.")
        
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        y_ = np.array([1 if i > 0 else 0 for i in y])

        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                linear_output = np.dot(x_i, self.weights) + self.bias
                y_predicted = self.activation_func(linear_output)
                update = self.learning_rate * (y_[idx] - y_predicted)
                self.weights += update * x_i
                self.bias += update

    def predict(self, X):
        """
        Predict class labels for samples in X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        y_pred : array-like of shape (n_samples,)
            The predicted class labels.
        """
        if not isinstance(X, np.ndarray):
            raise TypeError("X must be a numpy array.")
        
        linear_output = np.dot(X, self.weights) + self.bias
        y_predicted = self.activation_func(linear_output)
        return y_predicted

    def _unit_step_func(self, x):
        """
        Unit step activation function.

        Parameters
        ----------
        x : array-like
            The input data.

        Returns
        -------
        result : array-like
            The output of the unit step function.
        """
        return np.where(x >= 0, 1, 0)
