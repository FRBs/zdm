#!/bin/bash
#SBATCH --job-name=mcmc/BASE_limitH0
#SBATCH --output=../mcmc/mcmc/BASE_limitH0.out
#SBATCH --ntasks=20
#SBATCH --time=24:00:00
#SBATCH --export=NONE
#SBATCH --mem=40GB
# SBATCH --mem-per-cpu=5GB

# source /fred/oz002/jhoffmann/FRB_library/setup.sh
# source /fred/oz002/jhoffmann/FRB_library/loadzdm.sh

source $ZDM/.zdm_env/bin/activate

cd $ZDM/zdm

outfile="mcmc/BASE_limitH0"
walkers=40
steps=1000

surveys="DSA FAST CRAFT_class_I_and_II private_CRAFT_ICS_892_14 private_CRAFT_ICS_1300_14 private_CRAFT_ICS_1632_14 parkes_mb_class_I_and_II"
rep_surveys="CHIME/CHIME_decbin_0_of_6 CHIME/CHIME_decbin_1_of_6 CHIME/CHIME_decbin_2_of_6 CHIME/CHIME_decbin_3_of_6 CHIME/CHIME_decbin_4_of_6 CHIME/CHIME_decbin_5_of_6"
# cd data/Surveys/
# rep_surveys=$(ls CHIME/*)
# rep_surveys=${rep_surveys//".ecsv"/""}
# cd $ZDM/zdm

echo "Outfile: $outfile.h5"
echo "Walkers: $walkers"
echo "Steps: $steps"

# python MCMC_wrap.py -f $surveys -r $rep_surveys -p data/MCMC/params.json -o $outfile -w $walkers -s $steps -n 1
python MCMC_wrap2.py -f $surveys -p data/MCMC/params.json -o $outfile -w $walkers -s $steps -n 20 --Pn
# python MCMC_wrap2.py -r $rep_surveys -p data/MCMC/params.json -o $outfile -w $walkers -s $steps -n 1