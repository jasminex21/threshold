import csv
import functools
from itertools import zip_longest
from typing import List, Optional, Union
from pathlib import Path
import warnings

import numpy as np
import numpy.typing as npt
from openseize import producer
from openseize.core.producer import Producer
from openseize.file_io import annotations
from openseize.core.mixins import ViewInstance

from threshold import arraytools


def threshold(pro: Producer,
              nstds: List[float],
              chunksize: Optional[int]=None, radius: Optional[int] = None,
) -> List[npt.NDArray]:
    """Returns a list of 1-D boolean arrays, one per std in nstds, that denote
    samples from the producer whose normalized voltage values exceed that std.
    Theses samples are marked as False.

    Args:
        pro:
            A producer of ndarrays of data to threshold.
        nstds:
            A list of multiples of the data's standard deviation for determining 
            extreme values.
        chunksize:
            The number of samples the producer will produce during iteration.
            This value determines the local mean and standard deviation for
            thresholding.
        radius:
            The distance between two False samples indices below which all
            intermediate samples will be filled with False values in the mask.

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
        >>> # make a producer from spiked data and build masks
        >>> pro = producer(x, chunksize=100, axis=-1)
        >>> masks = threshold(pro, nstds=[2])
        >>> mask = masks[0]
        >>> # are the mask indices that are False equal to the locs provided?
        >>> set(np.where(~mask)[0]) == set(locs.flatten())
        True
        
    Returns:
        A list of 1-D boolean mask each having the same length as producer along 
        axis.
    """

    pro.chunksize = chunksize if chunksize else pro.chunksize
    axis = pro.axis

    masks = [np.ones(pro.shape[pro.axis], dtype=bool) for std in nstds]
    for idx, arr in enumerate(pro):

        mu = np.mean(arr, axis=axis, keepdims=True)
        std = np.std(arr, axis=axis, keepdims=True)
        
        # if a chunk is constant then arr - mu is all zeros
        if not np.any(std):
            std = np.ones_like(std)

        arr -= mu
        arr /= std

        for sigma, mask in zip(nstds, masks):
            _, cols = np.where(np.abs(arr) > sigma)
            cols = np.unique(cols + idx * pro.chunksize)
            mask[cols] = False
    
    # merge False values within radius samples of each other
    if radius:

        for mask in masks:
            for epoch in arraytools.aggregate1d(~mask, radius):
                mask[slice(*epoch)] = False

    return masks
    
def state(path, labels, fs, winsize, include=True):
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
        between:
            The start and stop annotation labels between which artifacts are
            used in the resulting mask.
        kwargs:
            Any valid kwarg for openseize's Pinnacle reader.

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


class MetaMask(ViewInstance):
    """A class that combines 1-D boolean submask with element-wise logic
    & tracks submask metadata.
    
    Attributes:
        submasks:
            A sequence of 1-D boolean arrays that are to be combined into
            a single mask.
        names:
            String names of each boolean array mask in submasks.
        logical:
            A callable that accepts 2 1-D boolean arrays and returns an
            element-wise logical combination.
        **kwargs:
            Any metadata that should be associated with this MetaMask instance.
            Examples include; genotype data, paths to submasks etc.
    """

    def __init__(self, submasks, names, logical=np.logical_and, **kwargs):
        """Initialize this MetaMask."""

        self.submasks = dict(zip(names, submasks))
        self.logical = logical
        self.__dict__.update(**kwargs)

    @property
    def mask(self):
        """Return the element-wise combination of all submasks in MetaMask."""

        # if masks are of unequal length, truncate masks to equal length
        submasks = list(self.submasks.values())
        lengths = np.array([len(m) for m in submasks])
        if any(lengths-min(lengths)):
            msg = (f'length of mask do not equal {lengths[0]} != {lengths[1]}.'
            ' Truncating mask!')
            warnings.warn(msg)
            submasks = [mask[:min(lengths)] for mask in submasks]

        return functools.reduce(self.logical, submasks)

if __name__ == '__main__':

    from openseize.file_io import edf
    from openseize import producer
    import numpy as np
   
    """
    # make a random array with 50 spikes in each row
    rng = np.random.default_rng(0)
    x = rng.normal(loc=0, scale=1.0, size=(4,1000))
    locs = rng.choice(np.arange(1000), size=(4,50), replace=False)
    for row, loc_ls in enumerate(locs):
        x[row, loc_ls] = 10
    # make a producer from spiked data
    pro = producer(x, chunksize=100, axis=-1)
    mask = threshold(pro, nstds=[2])[0]
    """

    rng = np.random.default_rng(0)
    mask_a = rng.choice([True, False], size=1000)
    mask_b = rng.choice([True, False], size=1000)

    metamask = MetaMask([mask_a, mask_b], ['a', 'b'])
    print(len(metamask.mask))
    

