#!/bin/bash
#SBATCH --job-name=dsa_meerkat_reps
#SBATCH --output=../../mcmc/dsa_meerkat_reps.out
#SBATCH --ntasks=20
#SBATCH --time=24:00:00
#SBATCH --export=NONE
# SBATCH --mem=100GB
#SBATCH --mem-per-cpu=8GB

###############################################################################
#   Author: Jordan Hoffmann                                                   #
#   Date:   04/06/2024                                                        #
#   Purpose:                                                                  #
#       Slurm script for an MCMC run.                                         #
#   Usage:                                                                    #
#       Change job-name and output in SBATCH commands                         #
#       Change outfile                                                        #
#       Check surveys to be used (assumed to be in default survey location)   #
#       Check command line parameters to run MCMC_wrap2.py                    #
###############################################################################

source $ZDM/.venv/bin/activate

cd $ZDM/zdm

outfile="mcmc/dsa_meerkat_reps"
walkers=40
steps=3000

surveys="DSA MeerTRAPcoherent MeerTRAPincoherent FAST CRAFT_class_I_and_II CRAFT_ICS_892 CRAFT_ICS_1300 CRAFT_ICS_1632 parkes_mb_class_I_and_II"
# surveys="DSA CRAFT_class_I_and_II private_CRAFT_ICS_892_14 private_CRAFT_ICS_1300_14 private_CRAFT_ICS_1632_14 parkes_mb_class_I_and_II"
rep_surveys="CHIME/CHIME_decbin_0_of_6 CHIME/CHIME_decbin_1_of_6 CHIME/CHIME_decbin_2_of_6 CHIME/CHIME_decbin_3_of_6 CHIME/CHIME_decbin_4_of_6 CHIME/CHIME_decbin_5_of_6"
# rep_surveys=CHIME/CHIME_decbin_3_of_6
# cd data/Surveys/
# rep_surveys=$(ls CHIME/*)
# rep_surveys=${rep_surveys//".ecsv"/""}
# cd $ZDM/zdm

echo "Outfile: $outfile.h5"
echo "Walkers: $walkers"
echo "Steps: $steps"

# command="python scripts/MCMC/MCMC_wrap.py -f $surveys -p data/MCMC/params.json -o $outfile -w $walkers -s $steps"
command="python scripts/MCMC/MCMC_wrap.py -f $surveys -r $rep_surveys -p data/MCMC/params4.json -o $outfile -w $walkers -s $steps"
echo $command
$command