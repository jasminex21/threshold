"""A module to assess how well the local threshold model classifies sample
indices as artifact or non-artifact in the awake state."""

import numpy as np

from threshold import masking

def process_file(
