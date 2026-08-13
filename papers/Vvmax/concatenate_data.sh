#!/bin/bash

# This script takes V/Vmax data from the z_macquart
#sample, which naturally excludes FRBs with negative z,
#and addends data from the "minz" calculation which assume
#a minimum redshift

# list of files for different minz values
minzfiles=`ls MinzOutput/Minzmacquart_vvmax_data_NSFR_0_alpha_0_???.dat`
#echo $minzfiles

basefile='v2Output/v2macquart_vvmax_data_NSFR_0.0_alpha_0.dat'

for file in $minzfiles; do
    length=${#file}
    stop=$((length-4))
    newfile=${file:0:stop}"_concatenated.dat"
    echo $newfile
    echo "cat $file > $newfile"
    cat $file > $newfile
    echo "cat $basefile >> $newfile"
    cat $basefile >> $newfile
    
done
