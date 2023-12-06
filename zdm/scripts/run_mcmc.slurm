#!/bin/bash
#SBATCH --job-name=DSA_FAST_CRAFT_4
#SBATCH --output=../mcmc/DSA_FAST_CRAFT_4.out
#SBATCH --ntasks=20
#SBATCH --time=24:00:00
#SBATCH --export=NONE
# SBATCH --mem=50GB
#SBATCH --mem-per-cpu=3GB

# source /fred/oz002/jhoffmann/FRB_library/setup.sh
# source /fred/oz002/jhoffmann/FRB_library/loadzdm.sh

source $ZDM/.zdm_env/bin/activate

cd $ZDM/zdm

outfile="DSA_FAST_CRAFT_4"
walkers=40
steps=2000

surveys="DSA.ecsv FAST.ecsv FAST2.ecsv CRAFT_class_I_and_II.ecsv private_CRAFT_ICS_892.ecsv private_CRAFT_ICS_1300.ecsv private_CRAFT_ICS_1632.ecsv parkes_mb_class_I_and_II.ecsv"

# Used when treating each FRB as an individual survey in HOFFMANN2023
# cd data/Surveys/
# surveys=$(ls -d -- Hoffmann2023/*[0-9]_exact.ecsv)
# cd $ZDM/zdm

# cd data/Surveys/
# surveys=$(ls Hoffmann2023/*.ecsv)
# cd $ZDM/zdm

echo "Outfile: $outfile.out"
echo "Walkers: $walkers"
echo "Steps: $steps"

python MCMC_wrap.py $surveys -p tests/files/real_mini_mcmc2.json -o $outfile -w $walkers -s $steps -n 20