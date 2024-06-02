import numpy as np
from sklearn.base import ClassifierMixin
from supervised_learning.BaseEstimator import BaseEstimator

class VotingClassifier(BaseEstimator, ClassifierMixin):
    """
    Voting Classifier implementation that allows for both hard and soft voting.

    Parameters
    ----------
    estimators : list of (str, estimator) tuples
        List of named base estimators. Each element of the list is a tuple containing
        the name and the estimator instance.
    
    voting : {'hard', 'soft'}, default='hard'
        If 'hard', uses predicted class labels for majority rule voting.
        If 'soft', predicts the class label based on the argmax of the sums of the predicted probabilities.

    Attributes
    ----------
    models_ : list
        List of fitted base estimators.

    Methods
    -------
    fit(X, y)
        Fit the voting classifier to the training data.
    
    predict(X)
        Predict classes for X.
    """

    def __init__(self, estimators, voting='hard'):
        self.estimators = estimators
        self.voting = voting
    
    def fit(self, X, y):
        """
        Fit the voting classifier to the training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.
        
        y : array-like of shape (n_samples,)
            The target values (class labels).

        Returns
        -------
        self : object
            Fitted estimator.
        """
        self.models_ = [clf.fit(X, y) for _, clf in self.estimators]
        return self
    
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
        if self.voting == 'hard':
            predictions = np.asarray([clf.predict(X) for clf in self.models_]).T
            maj_vote = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=1, arr=predictions)
            return maj_vote
        else:
            predictions = np.asarray([clf.predict_proba(X) for clf in self.models_])
            avg_proba = np.average(predictions, axis=0)
            return np.argmax(avg_proba, axis=1)
