

These files relate to the CRACO paper modelling

They process logs from CRACO obs, and generate necessary inputs to zDM


#1: get_configs
This routine reads in the log file, updates it with derived data, and re-saves it under Logs. It then looks up unique observing configurations, and prints them to screen

Produces "configs.csv"

#2: sim_configs
The routine loops over all previously identified configs, and generates
CRACO beams for them.


#3: weight_configs.py
This loads in beam histograms, and weights them according to weights derived in configs.dat
It then generates final beam patters for CRACO 900 and CRACO 1300 MHz observations

MANUAL: Copy these to the CRAFT beam directory zdm/data/BeamData/

This also outputs the total effective beam sensitivity (\int B^-1.5 d \Omega).

For CRACO beams, this is 
Total effective sensitivity of beam1 (906 MHz) is  0.00386
Total effective sensitivity of beam2 (1342 MHz) is  0.00391

For primary beams only, this is
Total effective sensitivity of beam1 (906 MHz) is  0.00589
Total effective sensitivity of beam2 (1342 MHz) is  0.00528

#4: plot_beams
This simply plots the previously generated beams. It also generates plots of the individual components, and a plot including the primary beamshape only


#5: make_dm_response
Calculates a DM mask, which represents the different limitations of maximum DM over the survey


#6 plot_ASKAP_CRACO.py
This script plots the rate of FRB detections for these different surveys

#7 print_weighting_factors.py
This generates plots of mean frequency etc etc, and averaged weightings
factors over the entire survey
