These files relate to the CRACO paper modelling, and for effective
ICS observations for CRAFT repetition modelling.

They process logs from CRACO observation files (in "Logs/"), and generate necessary inputs to zDM

The scripts are currently set up to do this for the 28ms, 13.8 ms and 3.4ms survey data.

Please regenerate the data by running through in the following order:


#1: get_configs
This routine reads in the log file, updates it with derived data, and re-saves it under Logs. It then looks up unique observing configurations, and prints them to screen

The current logfile is craco_84500_survey_db.cs

Produces "Logs/configs_???.csv" "Logs/itsamp_???.csv", where
??? is 1,2,4,8,16,64 are the time durations, in units of 1.728ms,
of CRACO integrations

#2: sim_configs
The routine loops over all previously identified configs (above), and generates
CRACO beams for them. Note these beams do not care about integration time


#3: weight_configs.py

MANUAL #1: copy zdm/data/BeamData/CRACO_900_bins.npy to BeamHistograms/craco_histogram_bins.npy

This loads in beam histograms, and weights them according to weights derived in configs.dat
It then generates final beam patters for CRACO 900 and CRACO 1300 MHz observations

MANUAL: Copy these to the CRAFT beam directory zdm/data/BeamData/, giving appropriate names

This also outputs the total effective beam sensitivity (\int B^-1.5 d \Omega).
Note this is relative - still TBD what the correct normalisation should be

#4: plot_beams
This simply plots the previously generated beams. It also generates numerous plots of the individual components, and a plot including the primary beamshape only.


#5: make_dm_response
Calculates a DM mask, which represents the different limitations of maximum DM over the survey. This should be copied to zdm/data/Efficiencies/

The units of the mask are in effective deg^2 hrs.


#6 gen_diagnostics.py
This generates plots of mean frequency etc etc, and averaged weightings
factors over the entire survey

It also generates FRB surveys.

#8 plot_[900 or 1300]_alternatives.py
Calculates total rates, and plots zDM curves, for various alternative configurations of CRACO, in order to evalaute the effect of various inefficiencies or future improvements

Or, create the ICS-like options, where FRBs have identical DM and width response to ICS observations.

It also creates .npy files which save the width and dm efficiency info


####### Now it's time to actually make the CRACO surveys!!! ############


#9: re-run gen_diagnostics.py, which will now take in the width and dm efficiency, and use the printed tables now
with these factors estimated

#10 plot_ASKAP_CRACO.py
This script plots the rate of FRB detections for these different surveys

Copy down the numbers into the rate table in the appendix




