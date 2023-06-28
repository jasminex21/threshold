"""This module has functions for computing the Power Spectral Density of
produced data that has been filtered to remove extreme events.

Functions:
    threshold: A function that returns indices of a producer that exceed a float
    multiple of the local standard deviation.
"""

from typing import Optional

import numpy as np
import numpy.typing as npt
from openseize import producer
from openseize.core.producer import Producer
from openseize.spectra import estimators, metrics, plotting


def threshold(pro: Producer,
              nstds: float,
              chunksize: Optional[int]=None, axis:
              int=-1
) -> npt.NDArray:
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

def psd_estimator(pro: Producer, fs: float, **kwargs):
    """ """

    return estimators.psd(pro, fs, **kwargs)

def channel_plot(cnt, freqs, estimates, plt_axes, label, norm=True, **kwargs):
    """ """

    estimates = metrics.power_norm(estimates, **kwargs) if norm else estimates
    ci_upper, ci_lower = metrics.confidence_interval(estimates, cnt)

    for ch_idx, psd in enumerate(estimates):
        plt_axes[ch_idx].plot(freqs, psd, label=label)
        plotting.banded(freqs, ci_upper, ci_lower, plt_axes[ch_idx])
        plt_axes[ch_idx].set_title(f'Channel {ch_idx}')

    plt_axes[0].set_xlabel('Frequency (Hz)', fontsize = 16)
    plt_axes[0].set_ylabel(r'PSD ($\mu V^2 / Hz$)', fontsize = 16)

    return plt_axes
            








if __name__ == '__main__':


    from openseize import demos
    from openseize.file_io import edf
    path = demos.paths.locate('recording_001.edf')
    reader = edf.Reader(path)
    reader.channels = [0,1,2]
    pro = producer(reader, chunksize=1e6, axis=-1)

    #results = threshold(pro, 2) # 2 stds is 95% of data
    indices = np.arange(100)
    mpro = masker(pro, indices)
