### Simulations performed for SKA science chapters
# Inputs: courtesy of Evan Keane


# METHOD

#1: sim_SKA_configs.py
This program reads in the inputs from Evan Keane.
They contain fields of view, and sensitivities,
when considering tied beams from the innermost N
antennas (mid) or stations (low).

The program produces out files in Configs/, which saves
redshift and dm distributionbs, thresholds, observing times
(because it's easier to scale TOBS than FOV, i.e. this is
a proxy for the tied array beam size, which reduces FOV
as more stations are included), total number of observed FRBs,
and calculated detection thresholds.

#2: make_zdists.py
This program reads in the configs from above.
It then chooses the one with the best N (highest number of FRBs
detected in total), and proceeds to calculate the zDM distribution
for 100 iterations of the Hoffmann et al paper.

