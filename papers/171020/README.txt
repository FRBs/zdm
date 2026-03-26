################ ABOUT ###############
This directory contains files that were used to produce the
z-priors for the paper:
https://ui.adsabs.harvard.edu/abs/2023PASA...40...29L/abstract

These calculations were done before zDM was put on github, and
effectively lived in a local "branch".

These files will almost certainly not run as-is, and no effort
has been made to allow them to run on any modern version of 
zDM.

They are simply included for posterity, and scientific
reproducability.


######## Steps ###########

171020.py
    - This file produces a prior on p(z) given observed
        properties of FRB20171020A, saves it to 
        z_priors_bestfit.npy and zvalues_for_priors.npy
        (likely in the "Data" folder)

use_z_priors.py
    - This step adds priors on redshift according to estimates of p(z|DM),
        contained in z_priors_bestfit.npy and zvalues_for_priors.npy.
    - It reads in all FRBs with 'reasonable' localisations, contained in
        'second_cut.csv', and spits out a file with this prior at the end
    - The output should be added to "zpriors_added.csv"
    - It does NOT account for the greater number of galaxies at large z.
        In other words, this is NOT a prior on a particular galaxy at a
        particular z, but a prior on all galaxies in a given z bin
        
paper_mag_prior.py
    - This functon adds the magnitude-based priors from Driver et al
        as per the PATH methodology.
    - It reads in "zpriors_added.csv", and spits out an array with
        the magnitude from Driver et al at the end
        This gets printed to the screen. Save it as a "Data/z_mag_priors.csv"

mag_prior.py
    - This reads in data from all the above and other sources,
        and prints out a table for insertion into the paper



####### Other Scripts #####

match_galaxies.py
    - Used to match galaxies from Liz's paper with galaxies I found from
        the catalogue.

plot_prior.py
    - This is used to produce to prior p(z) figure for the paper
    - Outputs paper_plot_pz_bestfit_only.pdf
