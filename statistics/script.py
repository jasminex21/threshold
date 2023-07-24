#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 21 15:51:17 2023

@author: jasmine
"""

"""A script to construct statistical plots of the counts and durations of
artifacts for STXBP1 and UBE3A mice.
"""

from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt

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
    stxbp1_paths = matching(regex, kind, filetypes=filetypes, **kwargs)

    title = 'Please select UBE3A files'
    ube3a_paths = matching(regex, kind, filetypes=filetypes, **kwargs)

    return {'stxbp1': stxbp1_paths, 'ube3a': ube3a_paths}



def build_masks(ann_paths, spindle_paths, genotype): # -> list of metamasks
    """ """
    
    # jasmine to document and add types and type check and lint
    result = []
    for apath, spath in zip(ann_paths, spindle_paths):

        mouse_id = str(apath.stem).split('_')[0]
        
        awake = masking.state(spath, labels=['w'], fs=5000, winsize=4)
        sleep = masking.state(spath, labels=['r', 'n'], fs=5000, winsize=4)
        annote = masking.artifact(apath, len(awake), 
                            labels=['Artifact', 'water_drinking', 'Artifact ','water_drinking '],
                            fs=5000, between=['Start', 'Stop'], include=True)
            
        metamask_0 = masking.MetaMask([awake, annote], ['awake', 'annote'],
                                    genotype=genotype, path=(spath, apath),
                                    mouse_id=mouse_id)

        metamask_1 = masking.MetaMask([sleep, annote], ['sleep', 'annote'],
                                    genotype=genotype, path=(spath, apath),
                                    mouse_id=mouse_id)

        result.extend([metamask_0, metamask_1])
    return result

# not sure if entirely functional as intended 
def counts(metamasks):
    
    art_counts = []
    
    for mmask in metamasks: 
        count = masking.events(mmask.mask).shape[0]
        art_counts.append(count)
        
    return art_counts
    
def durations(metamasks): 
    
    art_durations = []
    
    for mmask in metamasks: 
        dur = sum(np.diff(masking.events(mmask.mask))) / 5000
        art_durations.append(dur)
        
    return art_durations
    

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

def counts_old(events_dict):
    """ """
    
    # dict instead of list
    result = defaultdict(dict)
    for genotype, state_dict in events_dict.items():
        # state_dict.items()
        for state, ls in state_dict.items(): 
            result[genotype][state] = [arr.shape[0] for arr in ls]

    return result


def durations_old(events_dict): 
    
    result = defaultdict(dict)
    for genotype, state_dict in events_dict.items(): 
        for state, ls in state_dict.items(): 
            result[genotype][state] = [sum(np.diff(arr)).item() / 5000 for arr in ls]
            
    return result

def create_boxplots(data_dict, title):
    fig, axarr = plt.subplots(nrows=1, ncols=1, figsize=(8, 10))
    
    plot_labels = ["STXBP1 awake", "Ube3a awake", "STXBP1 asleep", "Ube3a asleep"]
    
    data = [data_dict["stxbp1"]["awake"], data_dict["ube3a"]["awake"], 
            data_dict["stxbp1"]["sleep"], data_dict["ube3a"]["sleep"]]
    
    bplot = axarr.boxplot(data, vert=True, patch_artist=True, labels=plot_labels)
    axarr.set_title(title)
    
    # colors
    colors = ["darkolivegreen", "darkcyan"]
    for patch, color in zip(bplot['boxes'], colors * 2):
        patch.set_facecolor(color)


if __name__ == '__main__':

    paths = select()
    
    stxbp1_annpaths, stxbp1_spaths = list(zip(*paths["stxbp1"]))
    ube3a_annpaths, ube3a_spaths = list(zip(*paths["ube3a"]))
    
