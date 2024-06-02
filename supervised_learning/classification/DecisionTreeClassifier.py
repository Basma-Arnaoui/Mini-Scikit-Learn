import numpy as np
from supervised_learning.BaseEstimator import BaseEstimator

class Node:
    """
    Class representing a node in the decision tree.
    """
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, info_gain=None, value=None):
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.info_gain = info_gain
        self.value = value

class DecisionTreeClassifier(BaseEstimator):
    """
    Decision Tree classifier.

    Parameters
    ----------
    min_samples_split : int, default=2
        The minimum number of samples required to split an internal node.
    
    max_depth : int, default=2
        The maximum depth of the tree.

    Methods
    -------
    fit(X, Y)
        Fit the decision tree classifier to the training data.
    
    predict(X)
        Predict class labels for samples in X.
    
    get_params(deep=True)
        Get parameters for this estimator.
    
    set_params(**params)
        Set the parameters of this estimator.
    """

    def __init__(self, min_samples_split=2, max_depth=2):
        self.root = None
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth

    def build_tree(self, dataset, curr_depth=0):
        X, Y = dataset[:, :-1], dataset[:, -1]
        num_samples, num_features = np.shape(X)

        if num_samples >= self.min_samples_split and curr_depth <= self.max_depth:
            best_split = self.get_best_split(dataset, num_samples, num_features)
            if best_split.get("info_gain", 0) > 0:
                left_subtree = self.build_tree(best_split["dataset_left"], curr_depth + 1)
                right_subtree = self.build_tree(best_split["dataset_right"], curr_depth + 1)
                return Node(best_split["feature_index"], best_split["threshold"], left_subtree, right_subtree, best_split["info_gain"])

        leaf_value = self.calculate_leaf_value(Y)
        return Node(value=leaf_value)

    def get_best_split(self, dataset, num_samples, num_features):
        best_split = {"info_gain": 0}
        max_info_gain = -float("inf")

        for feature_index in range(num_features):
            feature_values = dataset[:, feature_index]
            possible_thresholds = np.unique(feature_values)
            for threshold in possible_thresholds:
                dataset_left, dataset_right = self.split(dataset, feature_index, threshold)
                if len(dataset_left) > 0 and len(dataset_right) > 0:
                    y, left_y, right_y = dataset[:, -1], dataset_left[:, -1], dataset_right[:, -1]
                    curr_info_gain = self.information_gain(y, left_y, right_y)
                    if curr_info_gain > max_info_gain:
                        best_split = {
                            "feature_index": feature_index,
                            "threshold": threshold,
                            "dataset_left": dataset_left,
                            "dataset_right": dataset_right,
                            "info_gain": curr_info_gain
                        }
                        max_info_gain = curr_info_gain
        return best_split

    def split(self, dataset, feature_index, threshold):
        dataset_left = np.array([row for row in dataset if row[feature_index] <= threshold])
        dataset_right = np.array([row for row in dataset if row[feature_index] > threshold])
        return dataset_left, dataset_right

    def information_gain(self, parent, l_child, r_child):
        weight_l = len(l_child) / len(parent)
        weight_r = len(r_child) / len(parent)
        return self.gini_index(parent) - (weight_l * self.gini_index(l_child) + weight_r * self.gini_index(r_child))

    def gini_index(self, y):
        class_labels = np.unique(y)
        gini = 1 - sum([(len(y[y == cls]) / len(y)) ** 2 for cls in class_labels])
        return gini

    def calculate_leaf_value(self, Y):
        Y = list(Y)
        return max(Y, key=Y.count)

    def fit(self, X, Y):
        """
        Fit the decision tree classifier to the training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.
        
        Y : array-like of shape (n_samples,)
            The target values (class labels).
        """
        Y = Y.reshape(-1, 1)  # Reshape Y to be 2-dimensional
        dataset = np.concatenate((X, Y), axis=1)
        self.root = self.build_tree(dataset)

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
        return np.array([self.make_prediction(x, self.root) for x in X])

    def make_prediction(self, x, tree):
        if tree.value is not None:
            return tree.value
        feature_val = x[tree.feature_index]
        if feature_val <= tree.threshold:
            return self.make_prediction(x, tree.left)
        else:
            return self.make_prediction(x, tree.right)

    def get_params(self, deep=True):
        """
        Get parameters for this estimator.

        Returns
        -------
        params : dict
            Parameter names mapped to their values.
        """
        return {"min_samples_split": self.min_samples_split, "max_depth": self.max_depth}

    def set_params(self, **params):
        """
        Set the parameters of this estimator.

        Parameters
        ----------
        **params : dict
            Estimator parameters.

        Returns
        -------
        self : object
            Estimator instance.
        """
        for param, value in params.items():
            setattr(self, param, value)
        return self
