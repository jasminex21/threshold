"""This script computes & stores the Power Spectral Densities of a collection
of files using different voltage thresholds for artifact removal.

The primary question this project aims to address is how strongly do artifacts
impact long-duration (24 hour) estimates of the PSD and is there an optimal
local volatge threshold to remove these artifacts?

"""

from itertools import zip_longest

from openseize import producer
from openseize.file_io import edf
from openseize.filtering.iir import Notch
from openseize.resampling import resampling
from openseize.file_io import annotations, edf

import time

from threshold import masking

# PLAN
# for an edf, annotation and corresponding spindle file
# 1. create a between_pro
# 2. Notch filter the pro
# 3. Downsample the pro
#
# 4. Create masks the combination of (awake, artifact) and (sleep, artifact)
# 5. Mask the downsampled pro with (awake, artifact) mask & (sleep, artifact)
#    mask
# 6. create thresholded producers in awake and sleep states on per threshold
# 
# 7. take downsampled pro and mask with manual artifacts -> man_pro
# 8. split man_pro into awake (zx) and sleep (zy) producers
# 9. Finally compute psd estimates of each producer and store
#


def preprocess(path, annotes, start, stop, fs, M, chunksize=30e5, axis=-1):
    """Notch filters and downsamples data produced from a reader between start
    and stop annotations.


    # Fix if start or stop is None
    """

    reader = edf.Reader(path)
    reader.channels = [0, 1, 2]
    a, b = [ann.time * fs for ann in annotes if ann.label in [start, stop]]
    pro = masking.between_pro(reader, a, b, chunksize, axis)
    
    # Notch filter the producer
    notch = Notch(fstop=60, width=6, fs=fs)
    result = notch(pro, chunksize, axis, dephase=False)

    # downsample the producer
    result = resampling.downsample(result, M, fs, chunksize, axis)
    return result


def build_masks(epath, apath, spath, nstds=[4,5,6]):
    """ """

    # use of this function before a psd func will mean data is transversed 2X


    t0 = time.perf_counter()
    # open annotes
    with annotations.Pinnacle(apath, start=6) as reader:
        annotes = reader.read()
   
    # proprocess file
    pro = preprocess(epath, annotes, start='Start', stop='Stop',
                     fs=5000, M=20)

    # threshold masks
    thresholds = masking.threshold(pro, nstds, chunksize=1.5e4)

    
    # manual artifact mask
    annote = masking.artifact(apath, size=pro.shape[-1],
                        labels=[ 'Artifact', 'Artifact ', 'water', 'water '],
                        fs=250, between=['Start', 'Stop'])
    
    # build state masks
    awake = masking.state(spath, ['w'], fs=250, winsize=4)
    sleep = masking.state(spath, ['r', 'n'], fs=250, winsize=4)

    metamasks = []
    # build awake and threshold masks
    for a, b, std in zip_longest([awake], thresholds, nstds, fillvalue=awake):
        names = ['awake', f'threshold = {std}']
        results.append(masking.MetaMask([a, b],  names))

    # build sleep and threshold masks
    for a, b, std in zip_longest([sleep], thresholds, nstds, fillvalue=sleep):
        names = ['sleep', f'threshold = {std}']
        results.append(masking.MetaMask([a, b],  names))

    # build awake and annotated as well as sleep and annotated
    for a, b, names in [(awake, annote, ['awake', 'annote']),
                        (sleep, annote, ['sleep', 'annote'])]:
        results.append(masking.MetaMask([a, b], names))

    # build awake and sleep with artifacts in place
    for mask, name in zip([awake, sleep], ['awake', 'sleep']):
        results.append(masking.MetaMask([mask], [name]))
        

    print(f'build masks completed in {time.perf_counter() - t0} s')

    return results

if __name__ == '__main__':

    from pathlib import Path

    base_path = Path('/media/matt/Zeus/jasmine/stxbp1/')
    name = 'CW0DA1_P096_KO_15_53_3dayEEG_2020-04-13_08_58_30'
    epath = base_path.joinpath(name + '.edf')
    apath = base_path.joinpath(name + '.txt')
    spath = base_path.joinpath(name + '_sleep_states.csv')

    metamasks = build_masks(epath, apath, spath)

    

