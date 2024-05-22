from clone import clone
from model_selection.KFold import KFold
import numpy as np
from supervised_learning.classification.LogisticRegression import LogisticRegression
class StackingClassifier:
    def __init__(self, base_models, meta_model, n_folds=5, random_state=None):
        self.base_models = base_models
        self.meta_model = meta_model
        self.n_folds = n_folds
        self.random_state = random_state
        self.base_models_ = []

    def fit(self, X, y):

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
        meta_features = np.column_stack([
            np.column_stack([model.predict(X) for model in single_model]).mean(axis=1)
            for single_model in self.base_models_
        ])
        return self.meta_model.predict(meta_features)
