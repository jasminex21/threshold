import numpy as np
import matplotlib.pyplot as plt

from openseize.spectra.metrics import confidence_interval
from openseize.spectra.plotting import banded

def plot_psds(data, genotype,  name, normalize=False):
    """ """
    
    genotypes = ['stxbp1', 'ube3a']
    geno_idx = genotypes.index(genotype)
    datum = data[geno_idx][name]

    awake_data = {key: value for key, value in datum.items() if 'awake' in key} 
    sleep_data = {key: value for key, value in datum.items() if 'sleep' in key} 
    
    fig, axarr = plt.subplots(2, 3, figsize=(14, 6))
    fig.suptitle(f'Genotype: {genotype}, Animal: {name}')
    for state_idx, state_data in enumerate([awake_data, sleep_data]):
        for label, (cnt, freqs, psd) in state_data.items():
            for ch_idx, ch_psd in enumerate(psd):
                axarr[state_idx, ch_idx].plot(freqs, ch_psd, label=label)
                
                ci = confidence_interval(ch_psd, cnt, alpha=0.05)
                lowers, uppers = list(zip(*ci))
                banded(freqs, lowers, uppers, ax=axarr[state_idx, ch_idx])

        axarr[state_idx, -1].legend(loc='upper right')

    for ax in axarr[0,:]:
        ax.get_xaxis().set_ticks([])

    for ax in axarr.flatten():
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

    [axarr[0, idx].title(f'Ch {idx}') for idx in range(3)]
    axarr[1, 0].set_xlabel('Frequency (Hz)', fontsize=14)
    axarr[1,0].set_ylabel(r'$V^2 /\ Hz$', fontsize=14)
    
    
    return fig, axarr

if __name__ == '__main__':

    fig, axarr = plot_psds(data, 'stxbp1', 'cw01')
    plt.show()
