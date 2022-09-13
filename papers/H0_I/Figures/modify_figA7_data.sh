#!/bin/bash


THOUSAND=$1

infile=../../../zdm/craco/MC_Surveys/CRACO_std_May2022.dat
outfile1="FigureA7/CRACO_std_May2022_maxdm.dat"
outfile2="FigureA7/CRACO_std_May2022_missing.dat"
command="cp ../../../zdm/craco/MC_Surveys/CRACO_std_May2022.dat FigureA7/"
echo $command
$command
# checks if files already exists, and deletes them
if [ -f $outfile1 ]; then
    echo rm $outfile1
    rm $outfile1
fi

if [ -f $outfile2 ]; then
    echo rm $outfile2
    rm $outfile2
fi

STARTFRB=${THOUSAND}001
STOPFRB=$((THOUSAND+1))000

count=0
while read -r line; do
    read -ra words <<< "$line"
    if [[ "${words[0]}" == "FRB" ]]
    then
        if [ "${words[1]}" -lt "$STARTFRB" ]; then
            continue
        fi
        if [ "${words[1]}" -gt "$STOPFRB" ]; then
            continue
        fi
        
        IFS='.' read -ra dmbits <<< "${words[4]}"
        # check if DM is > threshold of 1000
        
        #if [ "${words[1]}" == "1001" ]; then
        #    exit
        #fi
        
        if [ ${dmbits[0]} -ge 1000 ]; then
            zbit=-1
        else
            zbit=${words[5]}
        fi
        newline="${words[0]} ${words[1]} ${words[2]} ${words[3]} ${words[4]} $zbit ${words[6]} ${words[7]}"
        echo $newline >> $outfile1
        mod3=$((${words[1]} % 3))
        if [ "$mod3" == 0 ]; then
            newline="${words[0]} ${words[1]} ${words[2]} ${words[3]} ${words[4]} -1 ${words[6]} ${words[7]}"
        fi
        echo $newline >> $outfile2
    elif [[ "${words[0]}" == "NFRB" ]]; then
        echo "NFRB 1000" >> $outfile1
        echo "NFRB 1000" >> $outfile2
    else
        echo $line >> $outfile1
        echo $line >> $outfile2
    fi
    
done < $infile
