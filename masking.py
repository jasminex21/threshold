import csv
import functools
from itertools import zip_longest
from typing import List, Optional, Union
from pathlib import Path

import numpy as np
import numpy.typing as npt
from openseize import producer
from openseize.core.producer import Producer
from openseize.file_io import annotations

from threshold import core


def threshold(pro: Producer,
              nstds: float,
              chunksize: Optional[int]=None,
) -> npt.NDArray:
    """Returns a 1-D boolean array the same length as the producer where
    normalized voltage values exceeding nstds are set to False.

    Args:
        pro:
            A producer of ndarrays of data to threshold.
        nstds:
            A multiple of the data's standard deviation for determining extreme
            values.
        chunksize:
            The number of samples the producer will produce during iteration.
            This value determines the local mean and standard deviation for
            thresholding.

    Examples:
        >>> from openseize.file_io import edf
        >>> from openseize import producer
        >>> import numpy as np
        >>> # make a random array with 50 spikes in each row
        >>> rng = np.random.default_rng(0)
        >>> x = rng.normal(loc=0, scale=1.0, size=(4,1000))
        >>> locs = rng.choice(np.arange(1000), size=(4,50), replace=False)
        >>> for row, loc_ls in enumerate(locs):
        ...     x[row, loc_ls] = 10
        >>> # make a producer from spiked data
        >>> pro = producer(x, chunksize=100, axis=-1)
        >>> masked_pro = threshold(pro, nstds=2)
        >>> print(masked_pro.shape[-1] == pro.shape[-1] - len(np.unique(locs)))
        True

    Returns:
        A 1-D boolean mask the same length as producer along axis.
    """

    pro.chunksize = chunksize if chunksize else pro.chunksize
    axis = pro.axis

    results = []
    for idx, arr in enumerate(pro):

        mu = np.mean(arr, axis=axis, keepdims=True)
        std = np.std(arr, axis=axis, keepdims=True)

        arr -= mu
        arr /= std

        # if exceeds in one channel we say exceeds in all
        _, cols = np.where(np.abs(arr) > nstds)
        cols += idx * pro.chunksize
        results.append(np.unique(cols))

    indices = np.concatenate(results)
    mask = np.ones(pro.shape[pro.axis], dtype=bool)
    mask[indices] = False
    return mask

def state(path, labels, fs, winsize, include=True, **kwargs):
    """Returns a boolean mask from a spindle sleep score text file.

    Args:
        path:
            A path to a spindle file.
        labels:
            A list of labels to include or exclude in the mask depending on the
            include argument
        fs:
            The sampling rate of the data this mask will be applied to.
        winsize:
            The Spindle window size in secs.
        include:
            A boolean indicating if labels should be set to True and all others
            False or False and all others True. This option allows for inclusion
            or exclusion of labels from a mask.
        **kwargs:
            Any valid kwarg for the read_spindle core function.

    Returns:
        A 1-D boolean array for masking data at a given sample rate.
    """

    # reads the states from a spindle formatted text file.
    with open(path) as infile:
        reader = csv.reader(infile)
        states = [row[1] for row in reader]

    # build an negate mask
    mask = np.array([state in labels for state in states], dtype=bool)
    if not include:
        mask = ~mask
    
    # interpolate mask from window units to sample units
    mask = np.atleast_2d(mask)
    result = np.repeat(mask, fs * winsize, axis=0)
    return result.flatten(order='F')

def artifact(path, size, labels, fs, between=[None, None], include=False, 
             **kwargs):
    """Returns a boolean mask from a Pinnacle annotations file.

    Args:
        path:
            Pinnacle file path containing annotations.
        size:
            The size of the mask to return.
        labels:
            The artifact labels to be designated False in returned mask.
        fs:
            The sampling rate of the data acquisition.
        relative_to:
            An annotation label from which all other annotations times are
            relative to. If None, the annotations are relative to the start of
            the recording.
        start:
            The line number of the file at path at which annotation reading
            should begin. Defaults to line number 6.
        kwargs:
            Any valid kwarg for core.read_pinnacle function.

    Returns:
        A 1-D boolean mask where indices whose label is in labels are False
        and all non-label indices are True.
    """

    headers = kwargs.pop('start', 6)
    with annotations.Pinnacle(path, start=headers, **kwargs) as reader:
        annotes = reader.read()
    
    # get the first and last annote to return annotes between
    a, b = between
    first = next(ann for ann in annotes if ann.label == a) if a else annotes[0]
    last = next(ann for ann in annotes if ann.label == b) if b else annotes[-1]
    
    # filter the annotes by the given labels & the between labels
    annotes = [ann for ann in annotes if ann.label in labels]
    annotes = [ann for ann in annotes if first.time <= ann.time <= last.time]
    # adjust the times of the annotes relative to the first & last annotes
    for annote in annotes:
        annote.time -= first.time

    return annotations.as_mask(annotes, size, fs, include=include)
    
def _between_gen(reader, start, stop, chunksize, axis):
    """A generating function returns a generator of ndarrays of samples between
    start and stop.

    Args:
        reader:
            An openseize reader instance.
        start:
            The index at which production of values from reader begins.
        stop:
            The index at which production of values from reader ends.
        chunksize:
            The number of samples to return along axis of each yield ndarray.
        axis:
            The axis along which samples will be produced.
    
    Returns:
        A generator of ndarrays of chunksize shape along axis between start and
        stop.

    Notes:
        This is a protected module-level function that is not intended for
        external calling. Its placement at module level versus nesting is to
        support concurrent processing.
    """

    starts = np.arange(start, stop, chunksize)
    for a, b in zip_longest(starts, starts[1:], fillvalue=stop):
        yield reader.read(a,b)

def between_pro(reader, start, stop, chunksize, axis=-1):
    """Returns a producer from a reader instance that produces values between
    start and stop.

    Args:
        reader:
            An openseize reader instance.
        start:
            The index at which production of values from reader begins.
        stop:
            The index at which production of values from reader ends.
        axis:
            The axis along which production should occur. Defaults to last axis.

    Examples:
        >>> from openseize.demos import paths
        >>> from openseize.file_io import edf
        >>> path = paths.locate('recording_001.edf')
        >>> reader = edf.Reader(path)
        >>> reader.channels = [0,1,2]
        >>> # read samples 10 to 1155 in chunks of 100
        >>> pro = between_pro(reader, 10, 1155, chunksize=100)
        >>> np.allclose(pro.to_array(), reader.read(10, 1155))
        True

    Returns:
        A producer instance producing samples between start and stop along axis.
    """

    # build a partial freezing all arg of _between_gen
    gen_func = functools.partial(_between_gen, reader, start, stop, chunksize, 
                                 axis)
    # compute the shape of the new producer
    shape = list(reader.shape)
    shape[axis] = stop - start

    return producer(gen_func, chunksize, axis, shape=shape)
    

class Multimask:
    """A factory that combines multiple 1-D boolean masks into a single boolean
    mask using any logical callable.

    Attributes:
        logical:
            A callable that applies element-wise logic to combine each appended
            mask into a single 1-D boolean mask.
        mask:
            A 1-D boolean array composed of the logical combination of each
            appended mask to this multimask.
    
    Examples:
        >>> arrs = [[1,0,1,0], [1,1,1,0], [1,0,1,1]]
        >>> masker = Multimask()
        >>> # set the protected _mask attr directly
        >>> masker._masks = arrs
        >>> masker.mask
        array([True, False, True, False])
    """

    def __init__(self, logical=np.logical_and):
        """Initialize this multimask.

        Args:
            logical:
                Any callable that takes multiple mask and returns a single mask.
                Defaults to numpy's logical_and.
        """

        self._masks = []

    def append(self, func, *args, **kwargs):
        """Constructs a mask from a function & stores to this Multimask."""

        self._mask.append(func(*args, **kwargs))

    @property
    def mask(self):
        """Returns the combined mask of all appended masks in this Multimask."""

        return functools.reduce(self.logical, self._mask)


if __name__ == '__main__':

    spath = ('/home/matt/python/nri/scripting/threshold/sandbox/data/'
    'CW0DI2_P097_KO_92_30_3dayEEG_2020-05-07_09_54_11_sleep_states.csv')

    mask = state_mask(path, labels=['r', 'n'], fs=250, winsize=4)
