import numpy as np
from sklearn.base import  ClassifierMixin
from supervised_learning.BaseEstimator import BaseEstimator

class VotingClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, estimators, voting='hard'):
        self.estimators = estimators
        self.voting = voting
    
    def fit(self, X, y):
        self.models_ = [clf.fit(X, y) for _, clf in self.estimators]
        return self
    
    def predict(self, X):
        if self.voting == 'hard':
            predictions = np.asarray([clf.predict(X) for clf in self.models_]).T
            maj_vote = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=1, arr=predictions)
            return maj_vote
        else:
            predictions = np.asarray([clf.predict_proba(X) for clf in self.models_])
            avg_proba = np.average(predictions, axis=0)
            return np.argmax(avg_proba, axis=1)
