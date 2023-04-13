
Functions to perform repeating FRB calculations fit to CHIME data


OTHER
Inside Papers/CHIMErepetition

tied_beam_sim.py
    - Simulates the paths of objects over CHIME beams
    - Saves output to "DATA" sub-directory
    - Run this in both "Formed" and "Tied" modes to generate output in
        TiedBeamSimulation and FormedBeamSimulation

plot_paths.py
    - plots output of tied beam sim
    - generates  histograms of CHIME events
    - Writes to "Nbounds[M]" where M is the number of dec bins



##############    sort_CHIME_FRBs.py   ##################
Script sorts CHIME FRBs into declination ranges


############ fit_repeating_distributions.py ##################
- This iterates over Rmin, Rmax, and gamma (3D cube) to calculate likelihoods for the number and DM distribution of observed repeaters.

- The most recent iteration first fixes Rgamma and Rmax, then optimises for Rmin (only good down to Rmin of 1e-6)

############# analyse repeat dists #################
This script exists to plot the results of fit_repeating_distributions.py

################ make_bestfit_plots.py #################
This performs detailed analysis for a small subset of cases. It takes
the output of fit_repeating_distributions, finds the best-fitting
parameters, and produces plots against DM, declination etc.


