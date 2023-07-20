"""Script for computing Power Spectral Density estimations from data that has
been thresholded to remove statistically outlying values."""

import time
import matplotlib.pyplot as plt
from openseize.demos import paths
from openseize.file_io import edf, annotations
from openseize import producer
from openseize.filtering import iir
from openseize.resampling import resampling
from openseize.spectra.estimators import psd
import numpy.typing as npt
import numpy as np
from openseize.core.producer import Producer
from openseize import demos
from openseize.file_io import dialogs

from core import read_pinnacle, threshold, plot, produce_between

SAVEDIR = '/home/matt/python/nri/scripting/threshold/sandbox/'

PATHS = dialogs.matching(r'\w+', kind='askopenfilenames')

#FIXME path ordering in tuples not guaranteed by dialog

for ann_path, eeg_path in PATHS:
    t0 = time.perf_counter()
    ANNOTES = read_pinnacle(ann_path, labels=['Artifact'], 
                            relative_to='Heet Start', start=6)
    ANN_PTS = read_pinnacle(ann_path, labels = ["Heet Start", "Heet Stop"], 
                              start = 6)
    START, STOP = [int(pt.time*5000) for pt in ANN_PTS]

    READER = edf.Reader(eeg_path)
    READER.channels = [0, 1, 2]
    
    PRO = producer(READER, chunksize=20e6, axis=-1)

    # FIXME REPLACE THIS WITH Reader.read and build producer
    BW_PRO = produce_between(PRO, START, STOP)

    # Build a notch filter and apply to producer
    NOTCH = iir.Notch(fstop = 60, width = 8, fs = 5000)
    NOTCHED = NOTCH(BW_PRO, axis = -1, chunksize = 20e6, dephase = False)

    # Downsampling the notched data
    DOWNED = resampling.downsample(NOTCHED, M = 20, fs = 5000, chunksize = 20e6)

    AMASK = annotations.as_mask(ANNOTES, size = DOWNED.shape[-1], fs = 250, include = False)
    APRO = producer(DOWNED, chunksize = 20e6, axis = -1, mask = AMASK)

    NSTDS = [1, 1.5, 2, 3]
    MASKED = [threshold(DOWNED, nstd) for nstd in NSTDS]

    fig, axarr = plt.subplots(1, 3, figsize=(30, 10))
    LABELS = [f'Std: {std}' for std in NSTDS] + ['No threshold', "Artifacts marked manually"]

    PROS = MASKED + [DOWNED, APRO]
    for label, pro in zip(LABELS, PROS):

        cnt, freqs, estimates = psd(pro, fs=250)
        plot(cnt, freqs, estimates, axarr, label)

    print(f'File {eeg_path.stem} took {time.perf_counter() - t0} secs')
    plt.setp(axarr, xlim=(0, 100))
    plt.savefig(SAVEPATH.join(eeg_path.stem))

