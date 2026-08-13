These files relate to the CRACO paper modelling, and for effective
ICS observations for CRAFT repetition modelling.

They process logs from CRACO observation files (in "Logs/"), and generate necessary inputs to zDM

The scripts are currently set up to do this for both the 13.8 ms and 3.4ms survey data.

Please regenerate the data by running through in the following order:


#1: get_configs
This routine reads in the log file, updates it with derived data, and re-saves it under Logs. It then looks up unique observing configurations, and prints them to screen

Produces "Logs/configs.csv" and "Logs/3ms_configs.csv"

#2: sim_configs
The routine loops over all previously identified configs (above), and generates
CRACO beams for them.


#3: weight_configs.py
This loads in beam histograms, and weights them according to weights derived in configs.dat
It then generates final beam patters for CRACO 900 and CRACO 1300 MHz observations

MANUAL: Copy these to the CRAFT beam directory zdm/data/BeamData/

This also outputs the total effective beam sensitivity (\int B^-1.5 d \Omega).

#4: plot_beams
This simply plots the previously generated beams. It also generates numerous plots of the individual components, and a plot including the primary beamshape only.


#5: make_dm_response
Calculates a DM mask, which represents the different limitations of maximum DM over the survey. This should be copied to zdm/data/Efficiencies/


#6 plot_ASKAP_CRACO.py
This script plots the rate of FRB detections for these different surveys

#7 gen_diagnostics.py
This generates plots of mean frequency etc etc, and averaged weightings
factors over the entire survey

#8 plot_[900 or 1300]_{alternatives or improvements].py
Calculates total rates, and plots zDM curves, for various alternative configurations of CRACO, in order to evalaute the effect of various inefficiencies or future improvements


