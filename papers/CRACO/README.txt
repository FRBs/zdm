

These files relate to the CRACO paper modelling

They process logs from CRACO obs, and generate necessary inputs to zDM


#1: get_configs
This routine reads in hte log file, updates it with derived data, and re-saves it under Logs. It then looks up unique observing configurations, and prints them to screen

MANUAL: save the screen output to "configs.dat"

#2: sim_configs
The routine loops over all previously identified configs, and generates
CRACO beams for them.

MANUAL: Shift all output to directory "BeamHistograms"

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




