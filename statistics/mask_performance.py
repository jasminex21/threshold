"""A module for assessing the similarities between masks.

"""

import numpy as np
import numpy.typing as npt
import time
import pickle
from functools import partial
from pathlib import Path
from multiprocessing import Pool

from openseize.file_io import annotations, path_utils
from threshold.psds.script import _masks
from threshold.tools import concurrency


def _tp(arr, actual):
    """Returns the number of True Positives between two 1-D arrays of binary
    classifications.

    Args:
        arr:
            A 1-D boolean of measured binary classifications.
        actual:
            A 1-D boolean array of ground-truth binary classifications.

    Examples:
        >>> arr = np.array([1, 1, 0, 1, 0, 1], bool)
        >>> actual = np.array([1, 0, 1, 1, 0, 1], bool)
        >>> _tp(arr, actual)
        3

    Returns:
        An integer count of true positive classifications.
    """

    x = set(np.flatnonzero(arr))
    y = set(np.flatnonzero(actual))
    return len(x.intersection(y))


def _tn(arr, actual):
    """Returns the number of True Negatives between two 1-D arrays of binary
    classifications.

    Args:
        arr:
            A 1-D boolean of measured binary classifications.
        actual:
            A 1-D boolean array of ground-truth binary classifications.

    Examples:
        >>> arr = np.array([1, 1, 0, 1, 0, 1], bool)
        >>> actual = np.array([1, 0, 1, 1, 0, 1], bool)
        >>> _tn(arr, actual)
        1

    Returns:
        An integer count of true negative classifications.
    """

    x = set(np.flatnonzero(~arr))
    y = set(np.flatnonzero(~actual))
    return len(x.intersection(y))


def _fp(arr, actual):
    """Returns the number of False Positives between two 1-D arrays of binary
    classifications.

    Args:
        arr:
            A 1-D boolean of measured binary classifications.
        actual:
            A 1-D boolean array of ground-truth binary classifications.

    Examples:
        >>> arr = np.array([1, 1, 0, 1, 0, 1], bool)
        >>> actual = np.array([1, 0, 1, 1, 0, 1], bool)
        >>> _fp(arr, actual)
        1

    Returns:
        An integer count of false positive classifications.
    """

    x = arr.astype(int)
    y = actual.astype(int)
    return len(np.where((x-y) > 0)[0])


def _fn(arr, actual):
    """Returns the number of False Negatives between two 1-D arrays of binary
    classifications.

    Args:
        arr:
            A 1-D boolean of measured binary classifications.
        actual:
            A 1-D boolean array of ground-truth binary classifications.

    Examples:
        >>> arr = np.array([1, 1, 0, 1, 0, 1], bool)
        >>> actual = np.array([1, 0, 1, 1, 0, 1], bool)
        >>> _fn(arr, actual)
        1

    Returns:
        An integer count of false negative classifications.
    """

    x = arr.astype(int)
    y = actual.astype(int)
    return len(np.where((y-x) > 0)[0])


def accuracy(arr: npt.NDArray, actual: npt.NDArray):
    """Returns the accuracy between two 1-D boolean classification arrays.

    The accuracy is the number of correct classifications divided by all 
    classifications (TP + TN) / (TP + TN + FP + FN).

    Args:
        arr:
            A 1-D boolean of measured binary classifications.
        actual:
            A 1-D boolean array of ground-truth binary classifications.

    Returns:
        A float accuracy value.
    """

    tp = _tp(arr, actual)
    tn = _tn(arr, actual)
    fp = _fp(arr, actual)
    fn = _fn(arr, actual)

    return (tp + tn) / (tp + tn + fp + fn)


def sensitivity(arr: npt.NDArray, actual: npt.NDArray):
    """Returns the sensitivity between two 1-D boolean classification arrays.

    The sensitivity is "how many did I catch out of all the ones I want to find"

    Args:
        arr:
            A 1-D boolean of measured binary classifications.
        actual:
            A 1-D boolean array of ground-truth binary classifications.

    Returns:
        A float sensitivity value.
    """

    tp = _tp(arr, actual)
    fn = _fn(arr, actual)

    return tp / (tp + fn)


def specificity(arr: npt.NDArray, actual: npt.NDArray):
    """Returns the specificity between two 1-D boolean classification arrays.

    The sensitivity is "how many did I ignore out of all the ones I want to
    ignore"

    Args:
        arr:
            A 1-D boolean of measured binary classifications.
        actual:
            A 1-D boolean array of ground-truth binary classifications.

    Returns:
        A float sensitivity value.
    """

    tn = _tn(arr, actual)
    fp = _fp(arr, actual)

    return tn / (tn + fp)


def precision(arr: npt.NDArray, actual: npt.NDArray):
    """Returns the precision between two 1-D boolean classification arrays.

    The precision is "how many did I correctly catch out of all the ones caught."

    Args:
        arr:
            A 1-D boolean of measured binary classifications.
        actual:
            A 1-D boolean array of ground-truth binary classifications.

    Returns:
        A float sensitivity value.
    """

    tp = _tp(arr, actual)
    fp = _fp(arr, actual)

    return tp / (tp + fp)


def _build(epath, apath, spath, radius, nstds):
    """Builds all metamasks for a single eeg file.

    Args:
        epath:
            Path to an eeg file
        apath:
            Path to annotation file associated with eeg at epath
        spath:
            Path to a spindle state file associated with eeg at epath.
        radius:
            The max distance between two masked samples that will be filled with
            False.
    """

    results = {}
    name = epath.stem.split('_')[0]
    results[name] = _masks(epath, apath, spath, radius, nstds)

    print(f'Mask estimated for file {name} complete')

    return results


def build_masks(dirpaths, save_path, radius, nstds, ncores=None):
    """Builds a 2-el list of dicts containing metamasks for each file in dirpath
    of dirpaths.

    Args:
        epath:
            Path to an eeg file
        apath:
            Path to annotation file associated with eeg at epath
        spath:
            Path to a spindle state file associated with eeg at epath.
        radius:
            The max distance between two masked samples that will be filled with
            False.
 """

    t0 = time.perf_counter()
    results = [{} for _ in range(len(dirpaths))]
    for result, dirpath in zip(results, dirpaths):

        epaths = list(Path(dirpath).glob('*.edf'))
        apaths = list(Path(dirpath).glob('*.txt'))
        spaths = list(Path(dirpath).glob('*.csv'))

        # use regex matching to match on animal names
        a = path_utils.re_match(epaths, apaths, r'\w+_')
        b = path_utils.re_match(epaths, spaths, r'\w+_')
        paths = []
        for (epath, apath), (epath, spath) in zip(a, b):
            paths.append((epath, apath, spath))

        workers = concurrency.set_cores(ncores, len(paths))

        # fix stds with partial
        f = partial(_build, radius=radius, nstds=nstds)
        with Pool(workers) as pool:
            processed = pool.starmap(f, paths)

        [result.update(dic) for dic in processed]

    # save data
    with open(Path(save_path).joinpath('masks.pkl'), 'wb') as outfile:
        pickle.dump(results, outfile)

    print(f'processed {len(paths)} files in {time.perf_counter() - t0} s')

    return results



if __name__ == '__main__':


    dirpaths = ['/media/matt/Zeus/jasmine/stxbp1/',
                '/media/matt/Zeus/jasmine/ube3a/']
    save_path = '/media/matt/Zeus/jasmine/results/'

    """
    base_path = Path('/media/matt/Zeus/jasmine/stxbp1/')
    name = 'CW0DA1_P096_KO_15_53_3dayEEG_2020-04-13_08_58_30'
    epath = base_path.joinpath(name + '.edf')
    apath = base_path.joinpath(name + '.txt')
    spath = base_path.joinpath(name + '_sleep_states.csv')

    results = _build(epath, apath, spath, radius=125, nstds=[5,6])
    """
    
    build_masks(dirpaths, save_path, radius=None, nstds=[4,5,6])
