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


# use lazy fixtures to pass parameterized fixtures into test
@pytest.mark.parametrize('arr', 
    [
        lazy_fixture('random1D'), 
        lazy_fixture('random2D'), 
        lazy_fixture('random3D'),
        lazy_fixture('random4D'),
    ]
)
def test_produce_between_dims(arr):
    """Test that produce_between produces the correct samples for producers with
    1-4 dimensions."""

    axis = np.argmax(arr.shape)
    pro = producer(arr, axis=axis, chunksize=10223)

    start, stop = 33, 1096
    pro_bw = core.produce_between(pro, start, stop)

    probe = pro_bw.to_array()
    actual = slice_along_axis(arr, start, stop, axis=axis)
    assert np.allclose(probe, actual)


def test_produce_between_indices(rng):
    """Test that produce beteween produces correct values for 100 start and
    stops on a random 2D array."""

    # generate random array with samples on axis=-1 and build producer
    arr = rng.random((4, 231098))
    pro = producer(arr, axis=-1, chunksize=10223)
    nsamples = pro.shape[-1]
    
    # select 100 random starts and stops
    starts = rng.choice(np.arange(nsamples), size=100, replace=False, axis=-1)
    stops = np.minimum(starts + 90, nsamples)
    
    for start, stop in zip(starts, stops):

        pro_bw = core.produce_between(pro, start, stop) 
        
        probe = pro_bw.to_array()
        actual = arr[..., start:stop]
        assert np.allclose(probe, actual)


def test_masked_between():
    """Validate that a between producer with mask produces the correct
    values."""

    # this test will be finished once masked pro in openseize is optimized
    pass


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
    




    

