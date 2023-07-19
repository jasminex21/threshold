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
