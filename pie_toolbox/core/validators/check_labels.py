import numpy as np


def check_labels(labels: np.ndarray, only_two_labels: bool = False):
    assert isinstance(labels, np.ndarray), "Labels must be a numpy array"
    assert labels.shape[0] > 1, "Labels array must have more than one value"
    assert np.unique(
        labels).size >= 2, "Labels array must have at least two unique values"
    if only_two_labels:
        assert np.unique(
            labels).size == 2, "Labels array must have exactly two unique values"
