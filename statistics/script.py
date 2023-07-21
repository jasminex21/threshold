"""A script to construct statistical plots of the counts and durations of
artifacts for STXBP1 and UBE3A mice.
"""

from collections import defaultdict
import numpy as np

from openseize.file_io.dialogs import matching
from threshold import masking
from threshold import statistics


# 0. select annotation and corresponding spindle files
#
# Function()
# FOR EACH PAIR:
# 1. extract the genotype
# 2. Construct an annotation mask (be sure to use the new between arg)
# 3. Construct 2 state mask: an awake mask and a sleep 
# 4. Supply the awake and annotation mask to count and durations and store to
#    a dict under the key {genotype_name: [{'awake': cnts, 'sleep': cnts}]}
# 5. make another dict that stores durations instead of cnts
#
# Function()
# 6. use these two dicts to make boxplots

def select(regex=r'\w+', kind='askopenfilenames', 
           filetypes=[("Annotation File", "*.txt"), 
                      ("Spindle File", "*.csv")], **kwargs):
    # FIXME I need documenting
    """ """

    title = 'Please select STXBP1 files'
    stxbp1_paths = matching(regex, kind, filetypes, title=title, **kwargs)

    title = 'Please select UBE3A files'
    ube3a_paths = matching(regex, kind, filetypes, title=title, **kwargs)

    return {'stxbp1': stxbp1_paths, 'ube3a': ube3a_paths}


def build_dict(geno_paths):
    """ """

    result = {}
    for genotype, paired_paths in geno_paths.items():
        geno_result = defaultdict(list)
        for ann_path, spin_path in paired_paths:

            awake_mask = masking.state(spin_path, labels=['w'], winsize=4,
                                       include=True)
            sleep_mask = masking.state(spin_path, labels=['r', 'n'], 
                                       winsize=4, include=True)

            ann_mask = masking.artifact(ann_path, len(awake_mask), 
                                        labels=['Artifact', 'water', 
                                                'Artifact ', 'water '],
                                        fs=5000, between=['Start', 'Stop'],
                                        include=True)
            
            x = statistics.artifacts.events(awake_mask, ann_mask)
            y = statistics.artifacts.events(sleep_mask, ann_mask)
            geno_result['awake'].append(x)
            geno_result['sleep'].append(y)

        result[genotype] = geno_result
            
    return result

def counts(events_dict):
    """ """

    result = defaultdict(list)
    for genotype, state_dict in events_dict.items():
        for state, ls in state_dict:
            result[genotype][state] = [arr.shape[0] for each arr in ls]

    return result



