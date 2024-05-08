import numpy as np

def train_test_split(X, y, test_size=0.25, train_size=None, random_state=None, shuffle=True, stratify=None):
    """Split arrays or matrices into random train and test subsets with optional stratification."""
    np.random.seed(random_state)
    if stratify is not None:
        if y.shape[0] != stratify.shape[0]:
            raise ValueError("Stratify array should be the same length as y")

        # Get unique classes and the indices for them
        classes, y_indices = np.unique(stratify, return_inverse=True)
        train_indices = []
        test_indices = []

        for cls in classes:
            cls_indices = np.where(y_indices == cls)[0]
            if shuffle:
                np.random.shuffle(cls_indices)
            
            cls_train_size = int(np.floor(train_size * len(cls_indices))) if train_size is not None else len(cls_indices) - int(np.ceil(test_size * len(cls_indices)))
            cls_test_size = len(cls_indices) - cls_train_size

            cls_train_indices = cls_indices[:cls_train_size]
            cls_test_indices = cls_indices[cls_train_size:cls_train_size + cls_test_size]

            train_indices.extend(cls_train_indices)
            test_indices.extend(cls_test_indices)

        train_indices = np.array(train_indices)
        test_indices = np.array(test_indices)
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

    X_train = X[train_indices]
    y_train = y[train_indices]
    X_test = X[test_indices]
    y_test = y[test_indices]

    return X_train, X_test, y_train, y_test
