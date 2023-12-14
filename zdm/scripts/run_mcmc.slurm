#!/bin/bash
#SBATCH --job-name=test_reps
#SBATCH --output=../mcmc/test_reps.out
#SBATCH --ntasks=1
#SBATCH --time=1:00:00
#SBATCH --export=NONE
#SBATCH --mem=20GB
# SBATCH --mem-per-cpu=3GB

# source /fred/oz002/jhoffmann/FRB_library/setup.sh
# source /fred/oz002/jhoffmann/FRB_library/loadzdm.sh

source $ZDM/.zdm_env/bin/activate

cd $ZDM/zdm

outfile="test_reps"
walkers=20
steps=10

surveys=""
rep_surveys="test_reps"

# surveys="DSA,FAST,FAST2,CRAFT_class_I_and_II,private_CRAFT_ICS_892,private_CRAFT_ICS_1300,private_CRAFT_ICS_1632,parkes_mb_class_I_and_II"

# Used when treating each FRB as an individual survey in HOFFMANN2023
# cd data/Surveys/
# surveys=$(ls -d -- Hoffmann2023/*[0-9]_exact.ecsv)
# cd $ZDM/zdm

# cd data/Surveys/
# surveys=$(ls Hoffmann2023/*.ecsv)
# cd $ZDM/zdm

echo "Outfile: $outfile.h5"
echo "Walkers: $walkers"
echo "Steps: $steps"

python MCMC_wrap.py -f $surveys -r $rep_surveys -p data/MCMC/params.json -o $outfile -w $walkers -s $steps -n 1