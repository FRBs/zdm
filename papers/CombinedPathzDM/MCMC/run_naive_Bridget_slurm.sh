#!/bin/bash
#SBATCH --job-name=naive_Bridget
#SBATCH --ntasks=15
#SBATCH --time=21:00:00
#SBATCH --export=NONE
#SBATCH --mem-per-cpu=4GB

# activate python environment
source  /fred/oz313/cwjames/virtual_environment/bin/activate

#export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

####### ACTUAL RUN #######
pfile="Input/bridget_mcmc_v1.json"
files="short_fake_CRACO_900"
sdir="/fred/oz313/cwjames/zdm/papers/CombinedPathzDM/BridgetMC/Surveys/"

steps=100
walkers=60
# use --rep_surveys to give repeater surveys
opfile="Output/naive_Bridget_mcmc_output_W"$walkers

# reduce size - this is CRACO data after all
Nz=300
Ndm=600
zmax=3
dmmax=3000

galdir=`pwd`'/../BridgetMC/CandidateFiles/'
frbdir=`pwd`'/../BridgetMC/FRBFiles/'

export ZDM_PATH_FRBDIR=$frbdir
export ZDM_PATH_GALDIR=$galdir
echo $ZDM_PATH_FRBDIR
echo $ZDM_PATH_GALDIR

script="../../../zdm/scripts/MCMC/MCMC_wrap.py"

##### PATH info ######
path="Input/naive_path_inputs.json"

# todo: add pwb back in?
runcommand="python $script -f $files --opfile=$opfile --pfile=$pfile -s $steps -w $walkers --Nz=$Nz --Ndm=$Ndm --zmax=$zmax --dmmax=$dmmax --path=$path --pwb --sdir=$sdir"

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
