"""This script computes the power spectral density for a collection of eeg files
and their associated annotation and spindel state files. Specifically it
constructs PSDs for each of the following conditions for each file:

    awake
    sleep
    awake + manually annotated
    sleep + manually annotated
    awake + threshold = 4 standard deviation
    awake + threshold = 5
    awake + threshold = 6
    sleep + theshold = 4
    sleep + theshold = 5
    sleep + theshold = 6

Before PSD computation, each file is preprocessed by notch filter 60 Hz and
downsampling from 5 KHz to 250 Hz.

Functions marked with an underscore are not intended for external calls.    
"""

import pickle
import time
from functools import partial
from itertools import zip_longest
from pathlib import Path
from multiprocessing import Pool

from openseize import producer
from openseize.file_io import annotations, edf, path_utils
from openseize.filtering.iir import Notch
from openseize.resampling import resampling
from openseize.spectra import estimators

from threshold.tools import concurrency
from threshold import masking


def _preprocess(path, annotes, start, stop, fs, M, chunksize=30e5, axis=-1):
    """Returns a producer producing chunksize samples along axis that has been
    notch filtered at 60 Hz and downsampled from fs to fs/M."""

    reader = edf.Reader(path)
    reader.channels = [0, 1, 2]
    a, b = [int(ann.time * fs) for ann in annotes if ann.label in [start, stop]]
    pro = masking.between_pro(reader, a, b, chunksize, axis)
    
    # Notch filter the producer
    notch = Notch(fstop=60, width=6, fs=fs)
    result = notch(pro, chunksize, axis, dephase=False)

    # downsample the producer
    result = resampling.downsample(result, M, fs, chunksize, axis)
    return result


def _masks(epath, apath, spath, nstds=[3, 4,5,6], verbose=False):
    """Returns a list of Metamask instances for the following conditions:

    awake
    sleep
    awake + manually annotated
    sleep + manually annotated
    awake + threshold = 4 standard deviation
    awake + threshold = 5
    awake + threshold = 6
    sleep + theshold = 4
    sleep + theshold = 5
    sleep + theshold = 6
    
    Args:
        epath:
            Path to an eeg file
        apath:
            Path to annotation file associated with eeg at epath
        spath:
            Path to a spindle state file associated with eeg at epath.
        verbose:
            If True prints timing information for mask construction.
    """

    t0 = time.perf_counter()

    # read annotations file to list of annote instances.
    with annotations.Pinnacle(apath, start=6) as reader:
        annotes = reader.read()
   
    # proprocess file
    pro = _preprocess(epath, annotes, 'Start', 'Stop', fs=5000, M=20)

    # threshold preprocessed by computing a boolean for each std in nstds
    thresholds = masking.threshold(pro, nstds, chunksize=1.5e4)

    # construct manually annotated boolean
    annote = masking.artifact(apath, size=pro.shape[-1],
                              labels=['Artifact', 
                                      'Artifact ',
                                      'water_drinking',
                                      'water_drinking '],
                              fs=250, between=['Start', 'Stop'])
    
    # build awake and sleep booleans
    awake = masking.state(spath, ['w'], fs=250, winsize=4)
    sleep = masking.state(spath, ['r', 'n'], fs=250, winsize=4)

    metamasks = []
    # build awake and threshold metamasks
    for a, b, std in zip_longest([awake], thresholds, nstds, fillvalue=awake):
        names = ['awake', f'threshold = {std}']
        metamasks.append(masking.MetaMask([a, b],  names))

    # build sleep and threshold metamasks
    for a, b, std in zip_longest([sleep], thresholds, nstds, fillvalue=sleep):
        names = ['sleep', f'threshold = {std}']
        metamasks.append(masking.MetaMask([a, b],  names))

    # build awake and sleep with manual artifacts removed
    for a, b, names in [(awake, annote, ['awake', 'annote']),
                        (sleep, annote, ['sleep', 'annote'])]:
        metamasks.append(masking.MetaMask([a, b], names))

    # build awake and sleep without artifact removal
    for mask, name in zip([awake, sleep], ['awake', 'sleep']):
        dummy = np.ones(pro.shape[-1])
        metamasks.append(masking.MetaMask([mask, dummy], [name, '']))
        
    if verbose:
        print(f'build masks completed in {time.perf_counter() - t0} s')

    return metamasks


def process_file(epath, apath, spath, nstds, verbose=False):
    """Returns a nested dictionary with animal name as the outer key and
    a condition tuple as the inner key storing openseize psd tuple results."""

    t0 = time.perf_counter()
    metamasks = _masks(epath, apath, spath, nstds)

    # open annotes
    with annotations.Pinnacle(apath, start=6) as reader:
        annotes = reader.read()
   
    # proprocess file
    pro = _preprocess(epath, annotes, start='Start', stop='Stop',
                     fs=5000, M=20)

    masked_pros = [producer(pro, chunksize=30e5, axis=-1, mask=m.mask) 
                   for m in  metamasks]

    psd_tups = [estimators.psd(mpro, fs=250) for mpro in masked_pros] 

    results = [[tuple(m.submasks.keys()), psd_tup] for m, psd_tup in 
                zip(metamasks, psd_tups)]

    name = epath.stem.split('_')[0]
    conds = [tuple(m.submasks.keys()) for m in metamasks]
    results = {name: dict(zip(conds, psd_tups))}

    print(f'PSD estimation for file {name} complete')

    return results


def process_files(dirpaths, save_path, nstds, ncores=None):
    """Processes all eeg, annotation and associated state files for each dir in
    dirpaths.

    This function would 
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
        f = partial(process_file, nstds=[4,5,6])
        with Pool(workers) as pool:
            processed = pool.starmap(f, paths)

        [result.update(dic) for dic in processed]

    # save data
    with open(Path(save_path).joinpath('psds.pkl'), 'wb') as outfile:
        pickle.dump(results, outfile)

    print(f'processed {len(paths)} files in {time.perf_counter() - t0} s')

    return results


if __name__ == '__main__':

    """
    base_path = Path('/media/matt/Zeus/jasmine/stxbp1/')
    name = 'CW0DA1_P096_KO_15_53_3dayEEG_2020-04-13_08_58_30'
    epath = base_path.joinpath(name + '.edf')
    apath = base_path.joinpath(name + '.txt')
    spath = base_path.joinpath(name + '_sleep_states.csv')

    #results = process_single(epath, apath, spath, nstds=[4,5,6])

    #metamasks = _masks(epath, apath, spath, verbose=True)

    #psd_dict = process_file(epath, apath, spath, nstds=[4,5,6], verbose=True)
    """
    
    dirpaths = ['/media/matt/Zeus/jasmine/stxbp1/',
                '/media/matt/Zeus/jasmine/ube3a/']
    save_path = '/media/matt/Zeus/jasmine/results'

    psds = process_files(dirpaths, save_path=save_path, nstds=[3, 4,5,6])



    


    

