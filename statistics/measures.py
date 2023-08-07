"""A module for assessing the similarities between masks.

"""

import numpy as np
import numpy.typing as npt


def _tp(arr, actual):
    """Returns the number of True Positives between two 1-D arrays of binary
    classifications.

    Args:
        arr:
            A 1-D boolean of measured binary classifications.
        actual:
            A 1-D boolean array of ground-truth binary classifications.

    Examples:
        >>> arr = np.array([1, 1, 0, 1, 0, 1], bool)
        >>> actual = np.array([1, 0, 1, 1, 0, 1], bool)
        >>> _tp(arr, actual)
        3

    Returns:
        An integer count of true positive classifications.
    """

    x = set(np.flatnonzero(arr))
    y = set(np.flatnonzero(actual))
    return len(x.intersection(y))


def _tn(arr, actual):
    """Returns the number of True Negatives between two 1-D arrays of binary
    classifications.

    Args:
        arr:
            A 1-D boolean of measured binary classifications.
        actual:
            A 1-D boolean array of ground-truth binary classifications.

    Examples:
        >>> arr = np.array([1, 1, 0, 1, 0, 1], bool)
        >>> actual = np.array([1, 0, 1, 1, 0, 1], bool)
        >>> _tn(arr, actual)
        1

    Returns:
        An integer count of true negative classifications.
    """

    x = set(np.flatnonzero(~arr))
    y = set(np.flatnonzero(~actual))
    return len(x.intersection(y))


def _fp(arr, actual):
    """Returns the number of False Positives between two 1-D arrays of binary
    classifications.

    Args:
        arr:
            A 1-D boolean of measured binary classifications.
        actual:
            A 1-D boolean array of ground-truth binary classifications.

    Examples:
        >>> arr = np.array([1, 1, 0, 1, 0, 1], bool)
        >>> actual = np.array([1, 0, 1, 1, 0, 1], bool)
        >>> _fp(arr, actual)
        1

    Returns:
        An integer count of false positive classifications.
    """

    x = arr.astype(int)
    y = actual.astype(int)
    return len(np.where((x-y) > 0)[0])


def _fn(arr, actual):
    """Returns the number of False Negatives between two 1-D arrays of binary
    classifications.

    Args:
        arr:
            A 1-D boolean of measured binary classifications.
        actual:
            A 1-D boolean array of ground-truth binary classifications.

    Examples:
        >>> arr = np.array([1, 1, 0, 1, 0, 1], bool)
        >>> actual = np.array([1, 0, 1, 1, 0, 1], bool)
        >>> _fn(arr, actual)
        1

    Returns:
        An integer count of false negative classifications.
    """

    x = arr.astype(int)
    y = actual.astype(int)
    return len(np.where((y-x) > 0)[0])


def accuracy(arr: npt.NDArray, actual: npt.NDArray):
    """Returns the accuracy between two 1-D boolean classification arrays.

    The accuracy is the number of correct classifications divided by all 
    classifications (TP + TN) / (TP + TN + FP + FN).

    Args:
        arr:
            A 1-D boolean of measured binary classifications.
        actual:
            A 1-D boolean array of ground-truth binary classifications.

    Returns:
        A float accuracy value.
    """

    tp = _tp(arr, actual)
    tn = _tn(arr, actual)
    fp = _fp(arr, actual)
    fn = _fn(arr, actual)

    return (tp + tn) / (tp + tn + fp + fn)


def sensitivity(arr: npt.NDArray, actual: npt.NDArray):
    """Returns the sensitivity between two 1-D boolean classification arrays.

    The sensitivity is "how many did I catch out of all the ones I want to find"

    Args:
        arr:
            A 1-D boolean of measured binary classifications.
        actual:
            A 1-D boolean array of ground-truth binary classifications.

    Returns:
        A float sensitivity value.
    """

    tp = _tp(arr, actual)
    fn = _fn(arr, actual)

    return tp / (tp + fn)


def specificity(arr: npt.NDArray, actual: npt.NDArray):
    """Returns the specificity between two 1-D boolean classification arrays.

    The sensitivity is "how many did I ignore out of all the ones I want to
    ignore"

    Args:
        arr:
            A 1-D boolean of measured binary classifications.
        actual:
            A 1-D boolean array of ground-truth binary classifications.

    Returns:
        A float sensitivity value.
    """

    tn = _tn(arr, actual)
    fp = _fp(arr, actual)

    return tn / (tn + fp)


def precision(arr: npt.NDArray, actual: npt.NDArray):
    """Returns the precision between two 1-D boolean classification arrays.

    The precision is "how many did I correctly catch out of all the ones caught."

    Args:
        arr:
            A 1-D boolean of measured binary classifications.
        actual:
            A 1-D boolean array of ground-truth binary classifications.

    Returns:
        A float sensitivity value.
    """

    tp = _tp(arr, actual)
    fp = _fp(arr, actual)

    return tp / (tp + fp)





