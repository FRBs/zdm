#!/bin/bash

# script to produce p(z|DM) for given FRBs.
# Optional arguments are SNR and width; if these are not input,
# the distribution is for all FRBs above threshold, rather than
# at a particular SNR. Similar with width.

######## Example for 210912 #######
SNR=31.7
width=0.07
# scattering and intrinsic width

command="python pz_given_dm.py -d 1234. -i 30.9 -s CRAFT/ICS -H 50 -o 210912_pzgdm -S $SNR -w $width"

echo $command
$command

######## Example for 210912, excluding width and SNR #######
#command="python pz_given_dm.py -d 1458 -i 31.0 -s CRAFT/ICS -H 50 -o 220610_pzgdm -z 1.01"


