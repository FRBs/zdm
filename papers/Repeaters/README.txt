#######################################################
######################## Overview #####################
#######################################################

This directory contains all information for plot used for the paper on
modelling repeating FRBs in zDM and its to CHIME data.

Below, each python file is described. All should be run with
python3 [filename].py
There are several "batches" of programs which should be run in order.
Each workflow is described below in basic detail. They are:
- Modelling CHIME: scripts to model CHIME in the zDM code
- 

As well as standard zDM dependencies, a copy of the CHIME
beam model package "CFBM", available at
https://github.com/chime-frb-open-data/chime-frb-beam-model
will be required. This may not be compatible with zDM, and
could require a separate Python environment. However,
the script that requires it, tied_beam_sim.py, does not
depent on zDM and can be used independently.

Obsolete, but potentially useful, files are contained in "Obsolete"

#######################################################
#################### MODELLING CHIME ##################
#######################################################

# The following scripts are set up to help model CHIME
# These must be run prior to the main processing sequence
# Some by-hand corrections will also be required.

################## dm_bias_fit.py  ####################

This produces a fit to the DM bias from CHIME Cat1.
Output is placed in "DMBiasFit/". The main figure
produced is "chime_bias_fit.pdf.

The key output of this has already been copied
to "survey.py" in /zdm/ to model the response of
the CHIME experiment.

This script also produces Figure 4 of the paper.

################## tied_beam_sim.py ###################

This script simulates the full CHIME beam using the CHIME
"cfbm" package (might not be compatible with zdm; I had
to use a separate python environment to run it).

It is designed to produce a 3D array in RA, DEC, FREQ
that gives the CHIME beam sensitivity as a function of position.
The outputs are stored in "TiedBeamSimulation".

Most of the dimensions of these arrays are hard coded, 
but could in principle be flexible.

This script takes typically O~few hours to run (from memory).

The "main" routine is run twice: once in "standard" mode,
to calculate the full beamshape; and once accounting for
tied beams only, for comparison with CHIME's published exposure
function.

################# plot_paths.py #####################

This script, named "plot paths" after its original function
of plotting the output of tied_beam_sim.py, reads in the
output from that script, and produces a beamshape for input
into the zdm simulation.

The most important routine is "make_beamfiles", which produces
directories "Nbounds6" and "Nbounds30" that contain CHIME
beam data when dividing into 6 and 30 declination bins
respectively.

It also generates Survey files of the form
"Surveys/CHIME_decbin_"+str(i)+"_of_"+str(Nbounds)+".dat"
where i runs from 0 to Nbounds-1 inclusive. However,
these do NOT contain FRBs or the correct FRB number
- this must be added by hand later (sorry!).

"Main" should be run with Nbounds=30 and Nbounds=6.

This routine also produces an effective exposure for each
of these declination bins, which should be added to the
CHIME survey files (by hand, sorry).

It also plots Figure 3 in the paper, which compares the CHIME beam
model to results published in CHIME Catalog 1.

######### sort_chime_frbs.py ##########

Reads in CHIME FRB data, and declination bins from
an "Nbounds[6/30]" directory.

Sorts it by declination bins, and extracts relevant info
for use in zDM.

Outputs to screen text tables of FRB data to be copied
into CHIME survey files. Very sorry, this isn't automated.
What needs to be copied are the "FRB" lines, and the total
number of FRBs.



#######################################################
##################### FITTING CHIME ###################
#######################################################

The following is the main processing sequence used to
produce "results".

########## fit_repeating_distributions.py #############
This script iterates over Rmax, and gamma (2D grid) to 
calculate likelihoods used in Section 6 of the paper.

The steps are:

- Load the CHIME surveys and generate grids for each
    declination bins.

- Find Rstar, the value at which Rmin=Rmax=Rstar produces
    the correct number of repeating CHIME FRBs relative to
    the number of single FRBs.

- For each Rmax, Rgamma, find Rmin, which produces the
    correct number of repeating CHIME FRBs. Note RMin
    can only be fit down to ~10^-9.

- Generate the repeating zDM grids, and calculate
    statistics for goodness-of-fit.
    
- Save results as a npz file in a directory named
    "Rfitting39_F", where F is the fraction of single
    CHIME FRBs attributes to repeaters (default 1.0).

The script by default does this only for F=1.0. However,
it should be looped over F=0.1... 1.0. Expect O~30
minutes per iteration.

The '39' refers to the default value of 10^39 erg at which
the repetition rate R is normalised.


################# add_mc_calculations.py ###############

Reads in output of fit_repeating_distributions.py, generates MC samples,
and evaluates the likelihood of CHIME observations against them.

It outputs "mc" files in each "Rfitting" directory, e.g.
Rfitting39_1.0/mc_FC391.0converge_set_0_output.npz

Typical run time is 1-2 hours per Rfitting iteration.

Note that some computational efficiency could be gained by
incorporating this into fit_repeating_distributions.py, and
this separation is largely for historical reasons.


################# Analyse repeat dists #################
This script exists to plot the results of add_mc_calculations.py,
and thus requires 'mc' files to be present.
(e.g. Rfitting39_1.0/mc_FC391.0converge_set_0_output.npz)

It produces a large set of plots in each "Rfitting" directory.

It also adds a line to the file "peak_likelihood.txt" to allow
comparisons between different values of FC.

This script produces Figures 7, 9, 11, 13, and 14 of the paper.

################ make_bestfit_plots.py #################

This performs detailed analysis for a small subset of cases.
Using "analyse repeat dists", four reasonable cases (a,b,c,d)
are identified, and their parameters Rgamma and Rmax are
entered by hand. It then reads in the best-fit value of Rmin
from fit_repeating_distributions.py, simulates these four cases,
and makes detailed plots for each.

Output is in BestfitCalculations.

This produces Figures 8, 10, 12, 16, and 17 of the paper.


#################### plot pFrep.py #####################
This script plots the data contained in "peak_likelihood.txt".
It expects "analyse_repeat_dists" to have been run in a loop
from either 0.1 to 1 or 1 to 0.1, thus creating an ordered
dataset in peak_likelihood.txt.

It produces the plot "Ptot_FC.pdf" in the base directory.
This is Figure 15 of the paper.




#######################################################
################## SEQUENCE 3: RSTAR ##################
#######################################################

################### find_rstar_of_f.py ################

This script generates a plot of Rstar as a function of the
repeater fraction F. It places outputs in the "Rstar" 
directory, one for each parameter set used.

################### plot_rstar_of_f.py ################

This script produces a plot of rstar for each data set.
It relies on the script "find_rstar_of_f.py" being run.

This generates Figure 6 of the paper.


#######################################################
############### SEQUENCE 4: z Simulation ##############
#######################################################

This sequence simulates observations assuming that
repeater redshifts are known.

########### evaluate_mc_calculation.py ################

This script performs two functions.

Firstly, for a fixed value of Rmax, Rgamma, Rmin, it will
generate a large number of Monte Carlo FRBs, and save
these in the file "mc.npy".

It will then iterate over the grid of Rmax, Rgamma, Rmin
generated by "add_mc_calculation" found in
Rfitting39_1.0/mc_FC391.0converge_set_0_output.npz
and evaluate the likelihood for each repetition set.
These are saved in the 'Rfitting39_1.0' directory as
'MCevaluation_set_0_all_lls.npy'

Finally, it plots the results of these evaluations
in Posteriors/, producing Figure 18 of the paper.

############# plot_single_MC_example.py ###############

This script is not used for the paper, but it's a
useful diagnostic. It takes parameters Rmin, Rmax, and
loads the output of "evaluate_mc_calculation.py"
(the file "mc.npy"), and generate plots in the directory
"TEST" of the Monte Carlo FRBs. It can be used to observe
where any given parameter set does or does not compare
with simulations


#######################################################
############### OTHER IMPORTANT SCRIPTS ###############
#######################################################

############ example_repeater_plots.py ################
This script serves two functions.

Firstly, it generates prediction plots of different
instruments for different timescales and parameter sets.

It places the output in the directory "Predictions/".

This produces Figure 19 of the paper.

Secondly, it produces an example plot using ASKAP to show
the effect of increasing pointing time in a given direction.
This gets saved to "ExamplePlots/", and used for
Figure 1 and Figure 2. Note that Figure 2 will be different
for each and every iteraton, since it is a Monte Carlo.

This script can be thought of as being independent from
the rest of the work, however the parameter sets for the
"predictions" are based on allowed regions derived from
the main processing sequence.

################# fit_single_bursts.py #################

This script compared predictions for the zDM distribution
of CHIME single bursts from different FRB population
parameters.

It iterates through all 14 parameter sets returned by
states.py (best-fit from Shin et al, and best-fit plus
12 90% CI sets from James et al) and models the 
predicted DM distribution of single bursts using two
predictions from repeating FRBs.

This produces Figure 5 from the paper. Also, KS
values used for Table 2 are also output. 


################## combine_followup.py ################

This script reads in data from FRB follow-up observations:
https://ui.adsabs.harvard.edu/abs/2020ApJ...895L..22J/abstract
contained in FollowUpData and the output of
fit_repeating_distributions.py and evaluates the likelihood
(simple linear interpolation in log space) from the former at
Rmin, Rmax, Rgamma points from the latter.

This produces Figure 21 of the paper, and is output in
FollowUpData.

############## toy_weibull.cpp and plot_weibull.py ########

This code lives in the directory "Weibull".

Firstly, the c++ code should be compiled with e.g.
g++ toy_weibull.cpp -o tw.exe -std=c++17

Next, run it with ./tw.exe . It will simulate the detection
probability of Weibull-distributed FRBs by CHIME.

Copy the output from the screen to the file "output.dat".

Then run "python3 plot_weibull.py" to plot the output, and
genrate Figure 20 from the paper.

Note that none of this depends on the zDM or other
external libraries.

############# test_decbin_division.py ############

This script compares predictions for the declination-
dependence of single and repeater CHIME FRBs when using
two different numbers of declination bins.

#######################################################
#################### LIBRARIES ########################
#######################################################

These two scripts contain common functions.

####################### states.py #####################

This script defines different parameter sets, and returns
corresponding states for input to zDM modelling.

It relies on the file "newE_planck_extremes.dat" being present,
which is based on the results of
https://ui.adsabs.harvard.edu/abs/2022MNRAS.516.4862J/abstract
and includes parameter sets consistent with the best fit at
90% confiendece, but with individual parameters pushed to
their extremes.

These best-fit parameters have been updated after the results of
https://ui.adsabs.harvard.edu/abs/2022arXiv221004680R/abstract
which indicate a higher value of Emax. However, these maximum
energies have been inreased uniformly by about 0.2, while
clearly low values of Emax are simply ruled out. We nonetheless
deem it better than ignoring the new Emax value.

####################### utilities.py ##################

This script defines useful functions for loading CHIME
FRBs and returning standard lists, e.g. lists of
repeaters, single bursts, histograms of DM and declination
etc. It is based on the data from CHIME_FRBs/chimefrbcat1.csv
which is taken from the online CHIME catalogue.


########### add_cprime.py #########

This script reads in the Monte Carlo file, and calculates
cprime (the total number of repeating FRBs) for that
particular state. It then adds these values of Cprime
to the output Monte Carlo file. It has only been used
as an intermediate, temporary file, since originally
the MC file did not include this information.

That functionality should be restored, hence this file is
outdated.
