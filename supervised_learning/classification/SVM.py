import numpy as np

class SVM:
    """
    Support Vector Machine (SVM) classifier using stochastic gradient descent.

    Parameters
    ----------
    learning_rate : float, default=0.001
        Learning rate for weight updates.
    
    lambda_param : float, default=0.01
        Regularization parameter.
    
    n_iters : int, default=1000
        Number of iterations for training.

    Methods
    -------
    fit(X, y)
        Fit the SVM classifier to the training data.
    
    predict(X)
        Predict class labels for samples in X.
    """

    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000):
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")
        if lambda_param <= 0:
            raise ValueError("lambda_param must be positive.")
        if n_iters <= 0:
            raise ValueError("n_iters must be positive.")
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.w = None
        self.b = None

    def fit(self, X, y):
        if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
            raise TypeError("X and y must be numpy arrays.")
        if X.shape[0] != y.shape[0]:
            raise ValueError("The number of samples in X and y must be equal.")
        
        n_samples, n_features = X.shape
        y_ = np.where(y <= 0, -1, 1)
        
        self.w = np.zeros(n_features)
        self.b = 0

        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                condition = y_[idx] * (np.dot(x_i, self.w) - self.b) >= 1
                if condition:
                    self.w -= self.lr * (2 * self.lambda_param * self.w)
                else:
                    self.w -= self.lr * (2 * self.lambda_param * self.w - np.dot(x_i, y_[idx]))
                    self.b -= self.lr * y_[idx]

    def predict(self, X):
        if not isinstance(X, np.ndarray):
            raise TypeError("X must be a numpy array.")
        if X.shape[1] != self.w.shape[0]:
            raise ValueError("The number of features in X must match the number of features in the training data.")
        
        linear_output = np.dot(X, self.w) - self.b
        return np.sign(linear_output)
