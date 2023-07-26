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


def test_threshold(rng):
    """Test that threshold mask detects the correct number of spikes for each
    std in nstds."""

    # TODO add a second channel
    csize = 1000
    stds = range(3,7)
    seq = np.concatenate([rng.normal(scale=s, size=csize) for s in stds])
    seq = np.atleast_2d(seq)

    x = np.reshape(seq, (len(stds), csize))
    x -= np.mean(x, axis=-1, keepdims=True)
    normed = x / np.std(x, axis=-1, keepdims=True)
    extremes = []
    for sigma in stds:
        cols = np.where(np.abs(normed.flatten()) > sigma)[0]
        extremes.append(cols)

    pro = producer(seq, chunksize=csize, axis=-1)
    masks = masking.threshold(pro, stds)
    for mask, extrema in zip(masks, extremes):
        _, cols = np.where(np.atleast_2d(~mask))
        assert np.allclose(cols, extrema)


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

