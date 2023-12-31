"""A collection of features that characterize a PSD."""

import pickle
import numpy as np
from collections import defaultdict

from scipy.signal import find_peaks
from scipy.integrate import simpson
from openseize.core.arraytools import slice_along_axis
from openseize.spectra.metrics import power_norm


def fetch_psds(data, geno, condition, genos=['ube3a', 'stxbp1']):
    """Returns the names, counts, frequencies & PSDs of animals with matching
    genotype & condition from a pickled list of dicts stored at path.

    Args:
        data:
            A list of dicts one per geno in genos where each dict has structure:
            {animal_i: {(condition_tuple)_j: (psd_tuple)_ij}
            where i runs over animals in this geno and condition runs over possible
            detection condition tuples (eg. [(awake,), (awake, annotated), ...]
        geno:
            A string name of a genotype to return psd results from.
        condition:
            A 2-tuple condition key to return cnts, freqs and psds for.
        genos:
            The order of the genotypes of the pickled list stored at path.

    Returns:
        names:
            A list of animal names in the same order as cnts, freqs & psds
        cnts:
            A 1-D array of counts, one per animal of that matches genotype & state
        freqs:
            A 1-D array of psd frequencies. Note assumption is that all psds
            were computed at the SAME frequencies.
        psds:
            A 3-D array of shape animals x channels x frequencies.
    """

    names, cnts, freqs, psds = [], [], [], []
    geno_idx = genos.index(geno)
    for animal, state_dict in data[geno_idx].items():
        for state_tuple, (cnt, freqs, psd) in state_dict.items():
            if condition == state_tuple:
                names.append(animal)
                cnts.append(cnt)
                psds.append(psd)

    cnts = np.array(cnts)
    psds = np.stack(psds)
    return names, cnts, freqs, psds


def filter_conditions(data, filterby='awake'):
    """Returns a list of conditions that contain the filterby state string.

    Args:
        data:
            A list of dicts one per geno in genos where each dict has structure:
            {animal_i: {(state_tuple)_j: (psd_tuple)_ij}
            where i runs over animals in this geno and condition runs over possible
            detection condition tuples (eg. [(awake,), (awake, annotated), ...]
        filterby:
            A string that must be a part of data's condition tuple keys in order
            to be returned.

    Returns:
        A list of condition tuples.
    """

    state_dict = list(data[0].values())[0]
    return [tup for tup in state_dict if filterby in tup]


def peak_counts(psds, **kwargs):
    """Returns the number of peaks feature from 3-D PSD data.

    Args:
        psds:
            An animals x channels x frequencies array of PSD values.

    Returns:
        A 2-D array of shape chs x animals containing peak counts.
    """

    results = np.zeros((psds.shape[1], psds.shape[0]))
    for mouse_idx, arr in enumerate(psds):
        for ch_idx, signal in enumerate(arr):

            peaks, _ = find_peaks(signal, **kwargs)
            results[ch_idx, mouse_idx] = len(peaks)
    
    return results


def max_height(psds, **kwargs):
    """Returns the height of the largest prominence peak feature in psds for
    each animal and channel.

    Args:
        psds:
            An animals x channels x frequencies array of PSD values.

    Returns:
        A 2-D array of shape chs x animals containing heights.
    """

    results = np.zeros((psds.shape[1], psds.shape[0]))
    for mouse_idx, arr in enumerate(psds):
        for ch_idx, signal in enumerate(arr):

            locs, info = find_peaks(signal, **kwargs)
            idx = np.argmax(info['prominences'])
            freq_idx = locs[idx]
            results[ch_idx, mouse_idx] = psds[mouse_idx, ch_idx, freq_idx]
    
    return results


def max_prominence(psds, **kwargs):
    """Returns the prominence of the largest prominence peak feature in psds 
    for each animal and channel.

    Args:
        psds:
            An animals x channels x frequencies array of PSD values.

    Returns:
        A 2-D array of shape chs x animals containing max peak prominences.
    """

    results = np.zeros((psds.shape[1], psds.shape[0]))
    for mouse_idx, arr in enumerate(psds):
        for ch_idx, signal in enumerate(arr):

            _, info = find_peaks(signal, **kwargs)
            results[ch_idx, mouse_idx] = np.max(info['prominences'])
    
    return results


def max_width(psds, **kwargs):
    """Returns the width of the largest prominence peak feature in psds for each
    animal and channel.

    Args:
        psds:
            An animals x channels x frequencies array of PSD values.

    Returns:
        A 2-D array of shape chs x animals containing max peak widths.
    """

    results = np.zeros((psds.shape[1], psds.shape[0]))
    for mouse_idx, arr in enumerate(psds):
        for ch_idx, signal in enumerate(arr):

            _, info = find_peaks(signal, **kwargs)
            idx = np.argmax(info['prominences'])
            results[ch_idx, mouse_idx] = info['widths'][idx]
    
    return results


def max_location(psds, **kwargs):
    """Returns the index of the max prominence peak feature in psds for each
    animal and channel.

    Args:
        psds:
            An animals x channels x frequencies array of PSD values.

    Returns:
        A 2-D array of shape chs x animals containing frequencies of max
        prominent peaks.
    """

    results = np.zeros((psds.shape[1], psds.shape[0]))
    for mouse_idx, arr in enumerate(psds):
        for ch_idx, signal in enumerate(arr):

            locs, info = find_peaks(signal, **kwargs)
            idx = np.argmax(info['prominences'])
            results[ch_idx, mouse_idx] = locs[idx]
    
    return results


def area(psds, **kwargs):
    """Returns the area below each PSD in psds.

    Args:
        psds:
            An animals x channels x frequencies array of PSD values.

    Returns:
        A 2-D array of shape chs x animals containing areas.

    Note:
        The spacing of integration points is 1 meaning the frequency resolution
        is ignored. This is fine since similarity is a comparison and the actual
        area units are not used.
    """

    x = simpson(psds, axis=-1)
    # all features return chs x animals
    return x.T


def featurize_psds(psds, freqs, features, start, stop, **kwargs):
    """Transforms a 3-D PSD array into a 3-D feature array for frequencies
    between start and stop.

    Args:
        psds:
            An animals x channels x frequencies array of PSD values.
        freqs:
            A 1-D array of frequencies for which PSDs were measured.
        *features:
            A list of frozen callables whose only free variable is psds.
        start:
            The start frequency over which to measure features. Must be an
            element of freqs.
        stop:
            The stop frequency over which to measure features. Must be an
            element of freqs.

    Returns:
        A len(features) x len(channels) x len(animals) feature array.
    """

    frequencies = list(freqs)
    a, b = [frequencies.index(f) for f in (start, stop)]
    x = slice_along_axis(psds, a, b, axis=-1)
    
    result = []
    for feature in features:
        result.append(feature(x))
    
    return np.stack(result)


def featurize(path, geno, state, features, genos=['ube3a', 'stxbp1'], start=0,
              stop=40):
    """Featurizes all the PSDs pickled at path for a given genotype and
    containing state in the condition.

    Args:
        path:
            String or Path instance to a pickle containing a list of PSD dicts
            with the following structures:
            {animal_i: {(state_tuple)_j: (psd_tuple)_ij}
            where i runs over animals in this geno and condition runs over possible
            detection condition tuples (eg. [(awake,), (awake, annotated), ...]
        geno:
            The genotype for which similarities will be determined, Must be one
            of genos.
        features:
            A list of callables that accept only a 3-D PSD ndarray with shape
            animals x channels x frequencies.
        genos:
            The order of the genotypes in the pickled list stored at path.
        start & stop:
            The start to stop frequencies over which features will be measured. 
            Must be an actual frequency in the pickled PSD data.
    
    Returns:
        A 2-el tuple with the first element being a dict keyed on conditions
        with an ndarray of features for values. The shape of this array will be
        len(features) x len(channels) x len(animals). The tuples second element
        is a list of the string names of the animals in the same order as the
        last axis of the PSD ndarrays.
    """

    with open(path, 'rb') as infile:
        data = pickle.load(infile)
    
    result = {}
    for condition in filter_conditions(data, filterby=state):
        # fetch the PSDs for each condition in filtered conditions
        # PSDs.shape = (animals, channels, frequencies)
        names, cnts, freqs, psds = fetch_psds(data, geno, condition, genos=genos)

        # featurize PSDs and store
        # featurized.shape = (features, channels, animals)
        result[condition] = featurize_psds(psds, freqs, features, start=start,
                                          stop=stop)

    return result, names


def similarity(featurized, relative_to=('awake', 'annote')):
    """Returns a dict of cosine similarities one per condition in featurized
    relative to supplied condition.

    Args:
        featurized:
            A dict containing condition keys and correpsonding 3-D feature
            arrays of shape (features, channels, animals) (see featurize)
        relative_to:
            A tuple indicating which vectors in featurized each featurized
            vector should be compared against for cosine similarity.

    Returns:
        A dict with the same condition keys as featurized and correponding 1-D
        array of cosine similarities, the average of the cosine similarities for
        each channel in featurized.
    """


    comparator = featurized[relative_to]

    # use matrix multiplication to compute all dot products at once:
    # diag(A x B) / sqrt(sum(A**2, feature_axis) * sum(B**2, feature_axis)) 
    # Jasmine seek to understand how numpy does high-d matrix multiplication

    # move comparators animal axis to last -> (animals, features, chs)
    y = np.moveaxis(comparator, -1, 0)
    #print(f'y shape is: {y.shape}')
    
    result = {}
    for condition, arr in featurized.items():

        # swap the featurized fea
        x = np.swapaxes(arr, 0, -1)
        #print(f'x shape is: {x.shape}')

        prod = np.diagonal(x @ y, axis1=-1, axis2=-2)
        # need to normalize
        z = prod / np.sqrt(np.sum(x**2, axis=-1) * np.sum(y**2, axis=1))
        
        # store avg similarity across channels
        result[condition] = np.mean(z, axis=-1)

    return result


    


if __name__ == '__main__':

    from functools import partial

    path = '/home/matt/python/nri/scripting/threshold/data/psds.pkl'
   
    features = []
    
    # These two contribute a lot to the similarity and may be ignored
    #features.append(partial(area))
    #features.append(partial(max_height, prominence=2, width=0))
    
    features.append(partial(peak_counts, prominence=4, width=0))
    features.append(partial(max_prominence, prominence=4, width=0))
    features.append(partial(max_width, prominence=4, width=0))
    features.append(partial(max_location, prominence=4, width=0))
    

    featurized, names = featurize(path, 'ube3a', 'awake', features)

    m = similarity(featurized)


