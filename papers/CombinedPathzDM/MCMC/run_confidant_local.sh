#!/bin/bash

####### ACTUAL RUN #######
pfile="Input/bridget_mcmc_v1.json"
files="confidant_short_fake_DSAlike"
sdir="/Users/cjames/CRAFT/Git/zdm/papers/CombinedPathzDM/BridgetMC/Surveys/"
#files="CRAFT_average_ICS"

steps=1000
walkers=40
# use --rep_surveys to give repeater surveys
opfile="Output/f1.5_confidant_mcmc_v2_output_W"$walkers

# reduce size - this is CRACO data after all
Nz=300
Ndm=600
zmax=3
dmmax=3000

script="../../../zdm/scripts/MCMC/MCMC_wrap.py"

# todo: add pwb back in?
runcommand="python $script -f $files --opfile=$opfile --pfile=$pfile -s $steps -w $walkers --Nz=$Nz --Ndm=$Ndm --zmax=$zmax --dmmax=$dmmax --pwb --sdir=$sdir"

echo $runcommand
$runcommand
