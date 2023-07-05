"""Script for computing Power Spectral Density estimations from data that has
been thresholded to remove statistically outlying values."""

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

from core import threshold, plot

"""
APATH = "/home/jasmine/python/nri/threshold/sandbox/data/CW0DI2_P097_KO_92_30_3dayEEG_2020-05-07_09_54_11.txt"
"""

APATH = ("/home/matt/python/nri/scripting/threshold/sandbox/data"
         "/CW0DI2_P097_KO_92_30_3dayEEG_2020-05-07_09_54_11.txt")
with annotations.Pinnacle(APATH, start = 6) as areader:
    ANNOTES = areader.read()

# Some wonky stuff to fix
start_time = ANNOTES[1].time
for ann in ANNOTES:
    ann.time = ann.time - start_time

# defining start and stop indices for the reader
start_indx = [ann.time for ann in ANNOTES if ann.label == "Heet Start"][0] * 5000
stop_indx = [ann.time for ann in ANNOTES if ann.label == "Heet Stop"][0] * 5000
# annotes should only contain the artifacts
ANNOTES = [ann for ann in ANNOTES if ann.label == "Artifact"] 

"""
PATH = "/home/jasmine/python/nri/threshold/sandbox/data/CW0DI2_P097_KO_92_30_3dayEEG_2020-05-07_09_54_11.edf"
"""

PATH = ("/home/matt/python/nri/scripting/threshold/sandbox/data"
        "/CW0DI2_P097_KO_92_30_3dayEEG_2020-05-07_09_54_11.edf")
READER = edf.Reader(PATH)
READER.channels = [0, 1, 2]
with READER as infile: 
    x = infile.read(start = int(start_indx), stop = int(stop_indx))

PRO = producer(x, chunksize=20e6, axis=-1)

# Build a notch filter and apply to producer
NOTCH = iir.Notch(fstop = 60, width = 8, fs = 5000)
NOTCHED = NOTCH(PRO, axis = -1, chunksize = 20e6, dephase = False)

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

plt.setp(axarr, xlim=(0, 100))
plt.show()

