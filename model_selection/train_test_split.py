import numpy as np

def train_test_split(X, y, test_size=0.25, train_size=None, random_state=None, shuffle=True, stratify=None):
    """
    Split arrays or matrices into random train and test subsets with optional stratification.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        The input data to split.
    
    y : array-like of shape (n_samples,)
        The target variable to split.
    
    test_size : float or int, default=0.25
        If float, should be between 0.0 and 1.0 and represent the proportion of the dataset to include in the test split.
        If int, represents the absolute number of test samples.

    train_size : float or int, default=None
        If float, should be between 0.0 and 1.0 and represent the proportion of the dataset to include in the train split.
        If int, represents the absolute number of train samples.
        If None, the value is automatically set to the complement of the test size.

    random_state : int or None, default=None
        Controls the shuffling applied to the data before applying the split.

    shuffle : bool, default=True
        Whether or not to shuffle the data before splitting.

    stratify : array-like or None, default=None
        If not None, data is split in a stratified fashion, using this as the class labels.

    Returns
    -------
    X_train : array-like of shape (n_train_samples, n_features)
        The training input samples.
    
    X_test : array-like of shape (n_test_samples, n_features)
        The testing input samples.
    
    y_train : array-like of shape (n_train_samples,)
        The training target values.
    
    y_test : array-like of shape (n_test_samples,)
        The testing target values.
    """
    np.random.seed(random_state)
    if stratify is not None:
        if y.shape[0] != stratify.shape[0]:
            raise ValueError("Stratify array should be the same length as y")

        classes, y_indices = np.unique(stratify, return_inverse=True)
        train_indices, test_indices = [], []

        for cls in classes:
            cls_indices = np.where(y_indices == cls)[0]
            if shuffle:
                np.random.shuffle(cls_indices)
            
            cls_train_size = int(np.floor(train_size * len(cls_indices))) if train_size is not None else len(cls_indices) - int(np.ceil(test_size * len(cls_indices)))
            cls_test_size = len(cls_indices) - cls_train_size

            train_indices.extend(cls_indices[:cls_train_size])
            test_indices.extend(cls_indices[cls_train_size:cls_train_size + cls_test_size])

        train_indices, test_indices = np.array(train_indices), np.array(test_indices)
    else:
        num_samples = X.shape[0]
        indices = np.arange(num_samples)
        if shuffle:
            np.random.shuffle(indices)

        if train_size is not None:
            if isinstance(train_size, float):
                train_size = int(train_size * num_samples)
            test_size = num_samples - train_size
        elif isinstance(test_size, float):
            test_size = int(test_size * num_samples)
            train_size = num_samples - test_size
        else:
            train_size = num_samples - test_size

        train_indices = indices[:train_size]
        test_indices = indices[train_size:train_size + test_size]

    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]

    return X_train, X_test, y_train, y_test
