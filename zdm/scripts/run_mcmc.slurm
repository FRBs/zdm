#!/bin/bash
#SBATCH --job-name=mcmc/HALO_dd
#SBATCH --output=../mcmc/HALO_dd.out
#SBATCH --ntasks=20
#SBATCH --time=24:00:00
#SBATCH --export=NONE
# SBATCH --mem=50GB
#SBATCH --mem-per-cpu=3GB

# source /fred/oz002/jhoffmann/FRB_library/setup.sh
# source /fred/oz002/jhoffmann/FRB_library/loadzdm.sh

source $ZDM/.zdm_env/bin/activate

cd $ZDM/zdm

outfile="mcmc/HALO_dd"
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

echo "Outfile: $outfile.h5"
echo "Walkers: $walkers"
echo "Steps: $steps"

# python scripts/MCMC_wrap.py -f $surveys -r $rep_surveys -p data/MCMC/params.json -o $outfile -w $walkers -s $steps -n 1
python scripts/MCMC_wrap2.py -f $surveys -p data/MCMC/params.json -o $outfile -w $walkers -s $steps -n 20 --Pn
# python scripts/MCMC_wrap2.py -r $rep_surveys -p data/MCMC/params.json -o $outfile -w $walkers -s $steps -n 1