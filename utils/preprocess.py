from numpy import ndarray
import numpy as np

def std(X: ndarray) -> ndarray:
    mean = np.mean(X, axis=0).reshape(-1, X.shape[1])
    max = X.max(axis=0).reshape(-1, X.shape[1])
    min = X.min(axis=0).reshape(-1, X.shape[1])
    return (X - mean) / (max - min)


def split_train_test(train_data: ndarray, train_label: ndarray, ratio=.2, rand_state=None):
    if rand_state:
        np.random.seed(rand_state)
    perm = np.random.permutation(len(train_data))
    test_size = int(len(train_data) * ratio)
    return train_data[perm[test_size:], :], train_label[perm[test_size:], :], train_data[perm[:test_size], :], train_label[perm[:test_size], :]

def shuffle(X: ndarray, y: ndarray, rand_state=None):
    if rand_state:
        np.random.seed(rand_state)
    perm = np.random.permutation(len(X))
    return X[perm, :], y[perm, :]
