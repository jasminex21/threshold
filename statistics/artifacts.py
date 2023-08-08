"""A script to compute the number and duration of artifacts for STXBP1 and UBE3A
mice in both sleep and wake states."""

import functools
import numpy as np


def events(*masks, logic=np.logical_and):
    """Returns the start, stop indices of True element runs in the element-wise
    combination of 1-D boolean masks.

    Args:
        *masks:
            A sequence of 1-D boolean arrays of the same length to combine.
        logic:
            A function that returns the element-wise combination of masks.

    Examples:
    >>> import numpy as np
    >>> mask_a = np.array([1, 1, 0, 1, 1, 1 ,1, 0, 1], dtype=bool)
    >>> mask_b = np.array([1, 1, 0, 0, 0, 1, 1, 1, 1], dtype=bool)
    >>> np.logical_and(mask_a, mask_b)
    array([ True,  True, False, False, False,  True,  True, False,  True])
    >>> events(mask_a, mask_b)
    array([[0, 2],
           [5, 7],
           [8, 9]])

    Returns:
        A 2-D array with start stop indices of each run along the last axis.

    Notes:
        The start, stop indices of each event follow python slicing conventions
        where start is inclusive and stop is exclusive.

    Raises:
        A value error is issued if the length of all masks are not equal.
    """

    mask = functools.reduce(logic, masks)
    
    if mask[-1]:
        mask = np.append(mask, False)

    # prepend 0 for right-hand side of each diff change
    endpoints = np.diff(mask, prepend=0)
    return np.where(endpoints)[0].reshape(-1, 2)


def count(*masks, logic=np.logical_and):
    """Counts the number of True events in the element-wise combination of all
    masks.

    Args:
        *masks:
            A sequence of 1-D boolean arrays of the same length to combine.
        logic:
            A function that returns the element-wise combination of masks.

    Returns:
        The number of True events in the combined mask.
    """

    return events(*masks, logic=logic).shape[0]


def durations(*masks, logic=np.logical_and):
    """Returns the duration of each True element run in the element-wise
    combination of all 1-D boolean masks in masks.

    Args:
        *masks:
            A sequence of 1-D boolean arrays of the same length to combine.
        logic:
            A function that returns the element-wise combination of masks.

    Returns:
        The duration of each True run in the logically combined mask of masks.
    """

    runs = events(*mask, logic=logic)
    return np.diff(runs, axis=-1)
