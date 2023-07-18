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

from threshold import core, masking


@pytest.fixture(scope='module')
def rng():
    """Returns a numpy default_rng for building reproducible random arrays."""

    seed = 0
    return np.random.default_rng(seed)


@pytest.fixture(scope='module')
def random1D(rng):
    """Returns a random 1D array using rng."""

    return rng.random((234098,))


@pytest.fixture(scope='module', params=permutations(range(2)))
def random2D(rng, request):
    """Yields random 2D arrays varying the sample axis across all possible axes
    permutations."""

    axes = request.param
    yield np.transpose(rng.random((197333, 6)), axes=axes)


@pytest.fixture(scope='module', params=permutations(range(3)))
def random3D(rng, request):
    """Returns random 3D arrays varying the sample axis across all possible axes
    permutations."""

    axes = request.param
    yield np.transpose(rng.random((100012, 6, 3)), axes=axes)


@pytest.fixture(scope='module', params=permutations(range(4)))
def random4D(rng, request):
    """Returns random 4D arrays varying the sample axis across all possible axes
    permutations."""

    axes = request.param
    yield np.transpose(rng.random((100012, 2, 3, 3)), axes=axes)


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
    




    

