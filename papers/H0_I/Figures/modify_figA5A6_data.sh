#!/bin/bash

# Script to modify MC data files generatd for FigureA5


basedir=$1

# this part re-labels MC data to give the false Galactic DM
# does this for DMG 500 to 0, 200 to 0, and 100 to 0
input=${basedir}/CRAFT_ICS_DMG_100.dat
output=${basedir}/CRAFT_ICS_DMG_100_as_0.dat
if ! test -f "$input"; then
    echo "Input file $input DNE, please run make_figA5_part1.py"
    exit
elif test -f "$output"; then
    echo "Output file $output already exists, skipping..."
else
    sed 's/ 100.0 / 0. /g' $input > $output
fi

input=${basedir}/CRAFT_ICS_DMG_200.dat
output=${basedir}/CRAFT_ICS_DMG_200_as_0.dat
if ! test -f "$input"; then
    echo "Input file $input DNE, please run make_figA5_part1.py"
    exit
elif test -f "$output"; then
    echo "Output file $output already exists, skipping..."
else
    sed 's/ 200.0 / 0. /g' $input > $output
fi

input=${basedir}/CRAFT_ICS_DMG_500.dat
output=${basedir}/CRAFT_ICS_DMG_500_as_0.dat
if ! test -f "$input"; then
    echo "Input file $input DNE, please run make_figA5_part1.py"
    exit
elif test -f "$output"; then
    echo "Output file $output already exists, skipping..."
else
    sed 's/ 500.0 / 0. /g' $input > $output
fi


##### create a mixed file with two different DM values
# needs to append these last lines
NFRB=1000
input1=${basedir}/CRAFT_ICS_DMG_0.dat
if ! test -f "$input1"; then
    echo "input file $input1 DNE, please run make_figA5_part1.py"
fi

input2=${basedir}/CRAFT_ICS_DMG_500.dat
output=${basedir}/CRAFT_ICS_DMG_0_500.dat
if test -f "output"; then
    echo "output file $output exists, skipping..."
else
    cat $input1 > $output
    tail -n $NFRB $input2 >> $output
fi

input2=${basedir}/CRAFT_ICS_DMG_200.dat
output=${basedir}/CRAFT_ICS_DMG_0_200.dat
if test -f "output"; then
    echo "output file $output exists, skipping..."
else
    cat $input1 > $output
    tail -n $NFRB $input2 >> $output
fi
