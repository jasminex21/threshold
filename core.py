"""This module has functions for computing the Power Spectral Density of
produced data that has been filtered to remove extreme events.

Functions:
    threshold:
        A function that returns indices of a producer that exceed a float
        multiple of the local standard deviation.
"""

import csv
from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np
import numpy.typing as npt
from openseize import producer
from openseize.core.producer import Producer
from openseize.file_io.annotations import Pinnacle
from openseize.spectra import metrics, plotting


def threshold(pro: Producer,
              nstds: float,
              chunksize: Optional[int]=None, axis:
              int=-1
) -> Producer:
    """Returns a producer where normalized samples exceeding nstds has been
    removed along axis.

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
        axis:
            The sample axis of the producer.

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
        An openseize producer instance.
    """

    pro.chunksize = chunksize if chunksize else pro.chunksize

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
    return producer(pro, pro.chunksize, pro.axis, mask=mask)

def read_pinnacle(path: Union[str, Path],
                  labels: Sequence[str],
                  relative_to: Optional[str] = None,
                  **kwargs):
    """Reads pinnacle annotations from file located at path with labels.

    Args:
        path:
            The pinnacle fmt. annotation file to read.
        labels:
            The annotation labels to return.
        relative_to:
            An annotation from which the times of all other annotations are
            relative to.
        kwargs:
            Any valid kwarg for the Pinnacle reader initializer.
    """

    all_labels = labels + [relative_to] if relative_to else labels
    with Pinnacle(path, **kwargs) as reader:
        annotes = reader.read(all_labels)

    if relative_to:
        idx = [idx for idx, ann in enumerate(annotes)
                if ann.label == relative_to][0]
        relative_annote = annotes.pop(idx)
        init_time = relative_annote.time
    else:
        init_time = 0

    for annote in annotes:
        annote.time = annote.time - init_time
    return annotes


def read_spindle(path, col=1):
    """Reads all the states in a spindle file.

    Args:
        path:
            The path to a spindle csv file.
    
    Returns:
        A list of states one per row in spindle file.
    """

    with open(path) as infile:
        reader = csv.reader(infile)
        return [row[col] for row in reader]


def plot(cnt: int,
        freqs: npt.NDArray[np.float64],
        estimates: npt.NDArray[np.float64],
        plt_axes: npt.NDArray,
        label: str,
        norm: bool = True,
        **kwargs):
    """Plots the Power Spectral Density by channel

    Args:
        cnt:
            The number of windows used to estimate the PSD
        freqs:
            A 1-D array of frequencies at which the PSD is estimated
        estimates:
            A 2-D array of estimates for the PSD, one per channel
        plt_axes:
            Matplotlib axes array upon which the PSD is plotted
        label:
            String label to be placed in the plot legend
        norm:
            Boolean determining whether to normalize the PSD by the total power
            between start and stop indices
        kwargs:
            Any valid keyword argument for openseize power_norm.

    Returns:
        A matplotlib axes instance.
    """

    estimates = metrics.power_norm(estimates, freqs, **kwargs) if norm else estimates
    cints = metrics.confidence_interval(estimates, cnt)

    for ch_idx, psd in enumerate(estimates):
        ci_upper, ci_lower = cints[ch_idx]
        plt_axes[ch_idx].plot(freqs, psd, label=label)
        plotting.banded(freqs, ci_upper, ci_lower, plt_axes[ch_idx])
        plt_axes[ch_idx].set_title(f'Channel {ch_idx}')

    plt_axes[0].set_xlabel('Frequency (Hz)', fontsize = 16)
    plt_axes[0].set_ylabel(r'PSD ($\mu V^2 / Hz$)', fontsize = 16)
    plt_axes[-1].legend()

    return plt_axes
