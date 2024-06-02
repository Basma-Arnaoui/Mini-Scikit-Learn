import numpy as np
from supervised_learning.BaseEstimator import BaseEstimator

class Node:
    """
    Class representing a node in the decision tree.
    """
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, var_red=None, value=None):
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.var_red = var_red
        self.value = value

class DecisionTreeRegressor(BaseEstimator):
    """
    Decision Tree regressor.

    Parameters
    ----------
    min_samples_split : int, default=2
        The minimum number of samples required to split an internal node.
    
    max_depth : int or None, default=None
        The maximum depth of the tree.

    Methods
    -------
    fit(X, Y)
        Fit the decision tree regressor to the training data.
    
    predict(X)
        Predict target values for samples in X.
    """

    def __init__(self, min_samples_split=2, max_depth=None):
        if not isinstance(min_samples_split, int) or min_samples_split <= 0:
            raise ValueError("min_samples_split must be a positive integer.")
        if max_depth is not None and (not isinstance(max_depth, int) or max_depth <= 0):
            raise ValueError("max_depth must be a positive integer or None.")
        
        self.root = None
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth

    def _build_tree(self, dataset, depth=0):
        X, Y = dataset[:, :-1], dataset[:, -1]
        num_samples, num_features = np.shape(X)

        if num_samples >= self.min_samples_split and (self.max_depth is None or depth <= self.max_depth):
            best_split = self._get_best_split(dataset, num_samples, num_features)
            if best_split.get("var_red", 0) > 0:
                left_subtree = self._build_tree(best_split["dataset_left"], depth + 1)
                right_subtree = self._build_tree(best_split["dataset_right"], depth + 1)
                return Node(best_split["feature_index"], best_split["threshold"], left_subtree, right_subtree, best_split["var_red"])

        leaf_value = self._calculate_leaf_value(Y)
        return Node(value=leaf_value)

    def _get_best_split(self, dataset, num_samples, num_features):
        best_split = {"var_red": 0}
        max_var_red = -float("inf")

        for feature_index in range(num_features):
            feature_values = dataset[:, feature_index]
            possible_thresholds = np.unique(feature_values)
            for threshold in possible_thresholds:
                dataset_left, dataset_right = self._split(dataset, feature_index, threshold)
                if len(dataset_left) > 0 and len(dataset_right) > 0:
                    y, left_y, right_y = dataset[:, -1], dataset_left[:, -1], dataset_right[:, -1]
                    curr_var_red = self._variance_reduction(y, left_y, right_y)
                    if curr_var_red > max_var_red:
                        best_split = {
                            "feature_index": feature_index,
                            "threshold": threshold,
                            "dataset_left": dataset_left,
                            "dataset_right": dataset_right,
                            "var_red": curr_var_red
                        }
                        max_var_red = curr_var_red
        return best_split

    def _split(self, dataset, feature_index, threshold):
        dataset_left = dataset[dataset[:, feature_index] <= threshold]
        dataset_right = dataset[dataset[:, feature_index] > threshold]
        return dataset_left, dataset_right

    def _variance_reduction(self, parent, l_child, r_child):
        weight_l = len(l_child) / len(parent)
        weight_r = len(r_child) / len(parent)
        return np.var(parent) - (weight_l * np.var(l_child) + weight_r * np.var(r_child))

    def _calculate_leaf_value(self, Y):
        return np.mean(Y)

    def fit(self, X, Y):
        """
        Fit the decision tree regressor to the training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.
        
        Y : array-like of shape (n_samples,)
            The target values.
        """
        if not isinstance(X, np.ndarray) or not isinstance(Y, np.ndarray):
            raise TypeError("X and Y must be numpy arrays.")
        if X.shape[0] != Y.shape[0]:
            raise ValueError("The number of samples in X and Y must be equal.")

        dataset = np.concatenate((X, Y.reshape(-1, 1)), axis=1)
        self.root = self._build_tree(dataset)

    def _make_prediction(self, x, tree):
        if tree.value is not None:
            return tree.value
        feature_val = x[tree.feature_index]
        if feature_val <= tree.threshold:
            return self._make_prediction(x, tree.left)
        else:
            return self._make_prediction(x, tree.right)

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
        if self.root is None:
            raise RuntimeError("The model has not been fitted yet.")
        
        return np.array([self._make_prediction(x, self.root) for x in X])
