from clone import clone
from model_selection.KFold import KFold
import numpy as np
from supervised_learning.classification.LogisticRegression import LogisticRegression

class StackingClassifier:
    """
    Stacking Classifier implementation using base models and a meta-model for ensemble learning.

    Parameters
    ----------
    base_models : list
        List of base models to be trained on the dataset.
    
    meta_model : estimator
        The meta-model to be trained on the predictions of the base models.
    
    n_folds : int, default=5
        Number of folds to be used for cross-validation.
    
    random_state : int, RandomState instance or None, default=None
        Controls the randomness of the cross-validation splitting and shuffling.

    Attributes
    ----------
    base_models_ : list
        List of lists containing the fitted base models.
    
    Methods
    -------
    fit(X, y)
        Fit the stacking classifier to the training data.
    
    predict(X)
        Predict classes for X.
    """

    def __init__(self, base_models, meta_model, n_folds=5, random_state=None):
        self.base_models = base_models
        self.meta_model = meta_model
        self.n_folds = n_folds
        self.random_state = random_state
        self.base_models_ = []

    def fit(self, X, y):
        """
        Fit the stacking classifier to the training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.
        
        y : array-like of shape (n_samples,)
            The target values (class labels).
        """
        kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
        out_of_fold_predictions = np.zeros((X.shape[0], len(self.base_models)))

        for i, model in enumerate(self.base_models):
            self.base_models_.append([])
            for train_index, holdout_index in kfold.split(X):
                instance = clone(model)
                self.base_models_[i].append(instance)
                instance.fit(X[train_index], y[train_index])
                y_pred = instance.predict(X[holdout_index])
                out_of_fold_predictions[holdout_index, i] = y_pred

        # Fit the meta-model using the out-of-fold predictions as features
        self.meta_model.fit(out_of_fold_predictions, y)

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
        meta_features = np.column_stack([
            np.column_stack([model.predict(X) for model in single_model]).mean(axis=1)
            for single_model in self.base_models_
        ])
        return self.meta_model.predict(meta_features)
