
Find "clancy_testing" under the 171020 paper page. Use that. Ignore the rest below.


Look in
/Users/cjames/CRAFT/Localisation/FromPravir/RecentResults
for Pravir's original data

Here, read in everything using "mag_prior.py"

Need to check confidence limits, they are inconsistent


"mod" files have been modified by me for easy reading

Using R-MAG_CANDIDATES.csv for r-band magnitudes
cand_positional_likelihood for likelihoods EXCEPY they seem to be 0.000...



######## Steps ###########

171020.py
    - This file produces a prior on p(z), saves it to 
        z_priors_bestfit.npy and zvalues_for_priors.npy

use_z_priors.py
    - This step adds priors on redshift according to estimates of p(z|DM),
        contained in z_priors_bestfit.npy and zvalues_for_priors.npy.
    - It reads in all FRBs with 'reasonable' localisations, contained in
        'second_cut.csv', and spits out a file with this prior at the end
    - The output should be added to "zpriors_added.csv"
    - It does NOT account for the greater number of galaxies at large z.
        
paper_mag_prior.py
    - This functon adds the magnitude-based priors from Driver et al
        as per the PATH methodology.
    - It reads in "zpriors_added.csv", and spits out an array with
        the magnitude from Driver et al at the end

mag_prior.py
    - This reads in the file modR-MAG_CANDIDATES.csv, and reads in
        the positional likelihoods from 'mod_cand_pos_likelihood.csv'



####### Other Scripts #####

match_galaxies.py
    - Used to match galaxies from Liz's paper with galaxies I found from
        the catalogue.

plot_prior.py
    - This is used to produce to prior p(z) figure for the paper
    - Outputs paper_plot_pz_bestfit_only.pdf
