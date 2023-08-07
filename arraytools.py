from collections import deque

import numpy as np
import numpy.typing as npt


def aggregate1d(arr: npt.NDArray[np.bool_], radius: float):
    """Returns start, stop indices for runs of True elements in a 1D
    boolean array that are within radius distance of one another.

    Args:
        arr: 
            A 1-D boolean whose True runs are to be aggregated.
        radius:
            The maximum index distance between two True elements in arr for 
            below which the elements are combined into the same run.

    Examples:
        >>> x = np.array([0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1], dtype=bool)
        >>> y = aggregate1d(x, radius=3)
        >>> y
        array([[ 1,  5],
               [ 8, 12]])

    Returns:
        A 2D array of start, stop indices for each True run in arr.
    """

    if arr.ndim > 1:
        raise ValueError('arr argument must be 1D')

    locs = np.flatnonzero(arr)

    # if no True events add an empty start, stop = 0, 0
    if not np.any(locs):
        return np.array([0, 0])

    result = []
    deck = deque(locs)
    sublist = [deck.popleft()]
    while deck:
        item = deck.popleft()
        if abs(item - sublist[-1]) <= radius:
            sublist.append(item)
        else:
            result.append(sublist)
            sublist = [item]
 
    # use else to place last sublist when deck is empty
    else: #pylint: disable=useless-else-on-loop
        result.append(sublist)

    # + 1 for consistency with slicing
    return np.array([(r[0], r[-1] + 1) for r in result], dtype=int)

