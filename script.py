"""Script for computing Power Spectral Density estimations from data that has
been thresholded to remove statistically outlying values."""

import matplotlib.pyplot as plt
from openseize.demos import paths
from openseize.file_io import edf
from openseize import producer
from openseize.filtering import iir
from openseize.resampling import resampling
from openseize.spectra.estimators import psd

from threshold.core import threshold, plot

PATH = paths.locate('recording_001.edf')
READER = edf.Reader(PATH)
READER.channels = [0,1,2]
PRO = producer(READER, chunksize=20e6, axis=-1)

# Build a notch filter and apply to producer
NOTCH = iir.Notch(fstop = 60, width = 8, fs = 5000)
NOTCHED = NOTCH(PRO, axis = -1, chunksize = 5e6, dephase = False)

# Downsampling the notched data
DOWNED = resampling.downsample(NOTCHED, M = 20, fs = 5000, chunksize = 5e6)

NSTDS = [1, 1.5, 2, 3]
MASKED = [threshold(DOWNED, nstd) for nstd in NSTDS]

fig, axarr = plt.subplots(1, 3, figsize=(30, 10))
LABELS = [f'Std: {std}' for std in NSTDS] + ['No threshold']

PROS = MASKED + [DOWNED]
for label, pro in zip(LABELS, PROS):

    cnt, freqs, estimates = psd(pro, fs=250)
    plot(cnt, freqs, estimates, axarr, label)

plt.setp(axarr, xlim=(0, 100))
plt.show()





