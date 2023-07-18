"""This script computes & stores the Power Spectral Densities of a collection
of files using different voltage thresholds for artifact removal.

The primary question this project aims to address is how strongly do artifacts
impact long-duration (24 hour) estimates of the PSD and is there an optimal
local volatge threshold to remove these artifacts?

"""

from openseize import producer
from openseize.file_io import edf
from openseize.filtering.iir import Notch
from openseize.resampling import resampling
from openseize.file_io import annotations

from threshold import core
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


def preprocess(reader, annotes, start, stop, fs, M, chunksize=30e6, axis=-1):
    """Notch filters and downsamples data produced from a reader between start
    and stop annotations.


    # Fix if start or stop is None
    """

    a, b = [ann.time * fs for ann in annotes if ann.label in [start, stop]]
    pro = masking.between_pro(reader, a, b, chunksize, axis)
    
    # Notch filter the producer
    notch = Notch(fstop=60, width=6, fs=fs)
    result = notch(pro, chunksize, axis, dephase=False)

    # downsample the producer
    result = resampling.downsample(result, M, fs, chunksize, axis)
    return result



