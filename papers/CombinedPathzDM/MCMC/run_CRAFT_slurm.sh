#!/bin/bash
#SBATCH --job-name=craft_zdmp
#SBATCH --ntasks=10
#SBATCH --time=24:00:00
#SBATCH --export=NONE
#SBATCH --mem-per-cpu=5GB

# activate python environment
source  /fred/oz313/cwjames/virtual_environment/bin/activate

####### ACTUAL RUN #######
pfile="Input/craft_mcmc.json"
files="CRAFT_ICS_892 CRAFT_ICS_1300 CRAFT_ICS_1632"
#files="CRAFT_average_ICS"
# use --rep_surveys to give repeater surveys
opfile="Output/CRAFT_MCMC_NW_40"
Pn=True
ptauw=False
steps=200
walkers=40

# reduce size - this is ICS data after all!
Nz=300
Ndm=600
zmax=3
dmmax=3000

script="../../../zdm/scripts/MCMC/MCMC_wrap.py"

##### PATH info ######
path="Input/ics_path_mcmc_inputs.json"

runcommand="python $script -f $files --opfile=$opfile --pfile=$pfile -s $steps -w $walkers --Nz=$Nz --Ndm=$Ndm --zmax=$zmax --dmmax=$dmmax --path=$path --Pn --pwb"

# ozstar has a memory leak that cannot be reproduced locally.
# Thus so we use multiple runs of e.g. 200 walkers
# this forces a reset of the Python memory allocation

echo $runcommand
$runcommand

echo $runcommand
$runcommand

echo $runcommand
$runcommand

echo $runcommand
$runcommand

echo $runcommand
$runcommand

echo $runcommand
$runcommand

echo $runcommand
$runcommand

echo $runcommand
$runcommand

echo $runcommand
$runcommand

echo $runcommand
$runcommand

echo $runcommand
$runcommand

echo $runcommand
$runcommand

echo $runcommand
$runcommand

echo $runcommand
$runcommand

echo $runcommand
$runcommand
