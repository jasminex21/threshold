"""A collection of features that characterize a PSD."""

import numpy as np



def psd_results(psd_dict, geno, condition='awake'):
    """generator returning all psds for all states containing state=awake"""

    return {state_tuple: [psd_tuples on per animal]}

def mse(a, b, axis=-1):
    """Returns the mean squared error between ndarrays a and b along axis.

    Args:
        a:
            An ndarray whose samples lie along axis.
        b:
            Another ndarray whose samples also lie along axis and whose shape
            must equal the shape of x.
        axis:
            The axis over which the MSE is to be computed.
    
    Returns:
        An ndarray with one less dim than a or b.
    """

    return np.sum((a - b)**2, axis=axis)


def max_height(x, y, axis=-1):
    """Returns the x location at which each slice in y is maximal along axis.

    Args:
        x:
            A 1-D array of locations that must match the shape of y along axis.
        y:
            An ndarray whose max height is to be measured along axis.
        axis:
            The sample axis along which the heights are to be measured.

    Returns:
        An array of x-locations with one-less dimension than y.
    """

   indices = np.argmax(y, axis=axis) 
   return x[indices]


def peaks(x, **kwargs):
    """Wrapper around scipy's find peaks for a 2-D array x.

    Args:
        x:
            A 2-D array whose peaks along axis=1 are to be estimated.
        **kwargs:
            Any valid kwarg for scipy's find_peaks function.

    Return:
        A list of find_peak tuple results.
    """

    return [find_peaks(signal, **kwargs) for signal in x]

