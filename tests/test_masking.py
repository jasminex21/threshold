"""A module for testing thresholds masking procedures

Typical usage example:
    !pytest test_masking.py::<TEST NAME>
"""

import csv
import tempfile
from itertools import permutations
from pathlib import Path

import pytest
import numpy as np
from pytest_lazyfixture import lazy_fixture
from openseize import producer
from openseize.core.arraytools import slice_along_axis
from openseize.demos import paths
from openseize.file_io import edf
from openseize.file_io.annotations import Pinnacle

from threshold import masking


@pytest.fixture(scope='module')
def rng():
    """Returns a numpy default_rng for building reproducible random arrays."""

    seed = 0
    return np.random.default_rng(seed)


def test_threshold():
    """Test that threshold mask detects the correct number of spikes for each
    std in nstds."""

    seed = 0
    rng = np.random.default_rng(seed)

    sig_len = 10000
    x = rng.normal(loc=0, scale=1, size=(4, sig_len))
    locs = rng.choice(np.arange(sig_len), size=(4, 20), replace=False)

    print('locs', locs)

    sigma_locs = [locs[:, ::3], locs[:, 1::3], locs[:, 2::3]]
    cnts = [len(l.flatten()) for l in sigma_locs]

    
    
    print('sigma_locs', sigma_locs)
    print('cnts', cnts)
    amplitudes = [4, 5, 6]
    for locs, amplitude in zip(sigma_locs, amplitudes):
        for row, spike_idxs in enumerate(locs):
            x[row, spike_idxs] = amplitude

    pro = producer(x, chunksize=1000, axis=-1)
    masks = masking.threshold(pro, nstds=amplitudes)
    print([np.count_nonzero(m) for m in masks])

    """
    for mask, cnt in zip(masks, [sum(cnts), sum(cnts[1:]), cnts[-1]]):
        assert np.count_nonzero(mask) == 10000 - cnt
    """

    return x, masks, sigma_locs




def test_between_pro(rng):
    """Validate that between_pro generates the correct ndarrays for openseizes
    sample data."""
    
    path = paths.locate('recording_001.edf')
    reader = edf.Reader(path)
    
    # build 100 producers from random starts and stops
    starts = rng.integers(int(15e6), size=100)
    stops = starts + rng.integers(10, int(2e6), size=100)
    for start, stop in zip(starts, stops):

        pro = masking.between_pro(reader, start, stop, reader.channels,
                                  chunksize=(stop-start)//5)
        
        # verify produced matches reader's read samples
        assert np.allclose(pro.to_array(), reader.read(start, stop))


def make_spindle(states):
    """Constructs a sample spindle file to test with."""

    tempdir = tempfile.mkdtemp(prefix='spindle_')
    path = Path(tempdir).joinpath('test_spindle.csv')
    
    with open(path, 'w') as outfile:
        
        writer = csv.writer(outfile)
        for idx, state in enumerate(states):
            writer.writerow([idx, state])

    return path


def test_state_mask():
    """Validates that a mask made from a spindle file has True values in the
    correct indices."""

    states = ['w', 'w', 'r', 'w', 'n']
    path = make_spindle(states)

    fs = 2
    winsize = 1
    
    # build a state mask for 'w' state and compare 
    probe = masking.state_mask(path, labels=['w'], fs=fs, winsize=winsize)
    expected = np.zeros(len(states) * fs * winsize)
    expected[0:4] = True
    expected[6:8] = True
    assert np.allclose(probe, expected)
    


if __name__ == '__main__':

    import matplotlib.pyplot as plt
    plt.ion()

    x, masks, sigma_locs = test_threshold()

    plt.plot(x[0])
    plt.scatter(sigma_locs[0], 4 * np.ones_like(sigma_locs[0]), 
                label='4-Sigma events')
    plt.legend()
    plt.show()

