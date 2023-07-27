<h1 align="center">
    <img src="https://github.com/jasminex21/threshold/blob/main/logos/readme_logo.png" 
    style="width:600px;height:auto;"/>
</h1>

# Introduction

The sources of artifacts in electroencephalogram (EEG) data is highly varied.
Body motions, eye movements, breathing, electrical interference and more induce
small voltage changes at the electrode contacts used to measure the EEG. Despite
their ubiquitous presence, it is unclear what types of signal processing
analyses are degraded by the presence of artifacts. Clearly, spike counting
analyses would be impacted but what about averaging processes like Welch's
estimation of the Power Spectral Density?
 
This project examines the impact of artifacts on the PSD measured in two mouse
genotypes with very different numbers of artifacts generated per hour. This
readme walks you through the process of estimating the PSDs with and without
artifacts. Additionally, we compare a local threshold model for artifact
detection against human annotated artifacts.

