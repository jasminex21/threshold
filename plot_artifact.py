#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 12 15:59:15 2023

@author: jasmine
"""

import matplotlib.pyplot as plt
from matplotlib import collections as mc
import numpy as np

from openseize import producer
from openseize.file_io import edf, dialogs

from core import read_pinnacle


# PATHS = dialogs.matching(r'\w+', kind='askopenfilenames', 
#                          initialdir="/home/jasmine/python/nri/threshold/sandbox/data",
#                          filetypes=[("Annotation File", "*.txt"), ("EEG Data", "*.edf")])
# PATH = PATHS[0][1]
# APATH = PATHS[0][0]
# ANNOTES = read_pinnacle(APATH, labels=['Artifact'], 
#                         relative_to='Heet Start', start=6)

# ANN_PTS = read_pinnacle(APATH, labels = ["Heet Start", "Heet Stop"], 
#                         start = 6)

PATHS = dialogs.matching(r'\w+', kind='askopenfilenames', 
                         initialdir="/media/jasmine/Data_A/jasmine",
                         filetypes=[("Annotation File", "*.txt"), ("EEG Data", "*.edf")])
PATH = PATHS[0][1]
APATH = PATHS[0][0]
ANNOTES = read_pinnacle(APATH, labels=['Artifact', "water_drinking"], 
                        relative_to='Start', start=6)

ANN_PTS = read_pinnacle(APATH, labels = ["Start", "Stop"], 
                        start = 6)

print(f"{len(ANNOTES)} artifacts available")
START, STOP = [int(pt.time*5000) for pt in ANN_PTS]

READER = edf.Reader(PATH)
READER.channels = [0, 1, 2]
with READER as infile: 
    x = infile.read(start=START, stop=STOP)

PRO = producer(x, chunksize=1e6, axis=-1)

def plot_artifact(pro, annotes, nstds, artifact_num, fs, axis=-1, chunksize=None): 
    
    pro.chunksize = chunksize if chunksize else pro.chunksize
    
    artifact = annotes[artifact_num]
    
    if pro.chunksize < (artifact.duration * fs): 
        raise ValueError(f"chunksize must be >= sample artifact duration ({int(artifact.duration * fs)})")
        
    art_indx = int(artifact.time * fs)
    art_midpoint = art_indx + int(0.5 * artifact.duration * fs)
    
    # plot extends 1/2 chunksize to either side of artifact 
    start = max(art_midpoint - (0.5 * pro.chunksize), 0)
    stop = min(art_midpoint + (0.5 * pro.chunksize), pro.shape[pro.axis])
    
    # to store coordinates of threshold lines 
    ulines = [[] for nstd in nstds]
    llines = [[] for nstd in nstds]
    
    # creating masked producer that contains only data from start to stop
    mask = np.zeros(pro.shape[pro.axis], dtype=bool)
    mask[int(start):int(stop)] = True
    contents = producer(pro, chunksize=pro.chunksize, 
                        axis=pro.axis, mask=mask).to_array()
    
    # means and standard deviations of each channel
    mu = np.mean(contents, axis=axis, keepdims=True)
    std = np.std(contents, axis=axis, keepdims=True)
    
    # computing thresholds and line coordinates for each nstd
    for indx, nstd in enumerate(nstds): 
        
        uthresholds = mu + (nstd * std)
        lthresholds = mu - (nstd * std)
        
        std_ulines = [[], [], []]
        std_llines = [[], [], []]

        # line coordinates format per channel per nstd: [(x1, y1), (x2, y2)]
        for nchannel in range(3): 
            std_ulines[nchannel] = [(0, uthresholds[nchannel][0]), 
                            (stop - start, uthresholds[nchannel][0])]
            std_llines[nchannel] = [(0, lthresholds[nchannel][0]), 
                            (stop - start, lthresholds[nchannel][0])]
                           
        ulines[indx] = std_ulines
        llines[indx] = std_llines
        
    fig, ax = plt.subplots(3, 1, figsize=(30, 20))
    colors = "bgcmykw"
    
    # plotting by channel
    for channel, content in enumerate(contents):
        ax[channel].plot(content)
        
        # plotting threshold line for each nstd
        for nstd_indx in range(len(nstds)): 
            ulc = mc.LineCollection([ulines[nstd_indx][channel]], 
                                    colors=colors[nstd_indx], linewidths=1.5, 
                                    label=f"Std {nstds[nstd_indx]}")
            ax[channel].add_collection(ulc)
            llc = mc.LineCollection([llines[nstd_indx][channel]], 
                                    colors=colors[nstd_indx], linewidths=1.5)
            ax[channel].add_collection(llc)
            
        # red dot at the location of the artifact (midpoint of duration)
        ax[channel].plot(art_midpoint - start, 
                         np.mean(content), "ro") 
        
        # red line spanning the duration of artifact
        duration_line = [[(art_indx - start, np.mean(content)), 
                          (art_indx + (artifact.duration * fs) - start, 
                           np.mean(content))]]
        dlc = mc.LineCollection(duration_line, colors = "r")
        ax[channel].add_collection(dlc)
        
        ax[channel].set_title(f"Channel {channel}")
    
    ax[0].legend(fontsize=10, loc="upper right")
    plt.setp(ax, xticks=plt.xticks()[0][1:-1], 
             xticklabels=plt.xticks()[0][1:-1] + start, ylim=(-1500, 1500))
    plt.show()
        
# 0-indexing for artifact number, I can change it though    
plot_artifact(PRO, ANNOTES, [3], 10, 5000, chunksize=500000)
                   
               




