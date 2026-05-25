#!/bin/bash
#SBATCH --job-name=reps_Pnr_noH0_pwb
#SBATCH --output=../../mcmc/reps_Pnr_noH0_pwb.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --time=48:00:00
#SBATCH --export=NONE
#SBATCH --mem=32GB
# SBATCH --mem-per-cpu=8GB

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

outfile="mcmc/reps_Pnr_noH0_pwb"
walkers=40
steps=3000

# surveys="DSA no_Tobs/MeerTRAPcoherent no_Tobs/MeerTRAPincoherent no_Tobs/FAST no_Tobs/CRAFT_class_I_and_II no_Tobs/parkes_mb_class_I_and_II"
surveys="DSA MeerTRAPcoherent MeerTRAPincoherent FAST CRAFT_class_I_and_II parkes_mb_class_I_and_II"

rep_surveys="CRAFT_average_ICS CHIME/CHIME_decbin_0_of_6 CHIME/CHIME_decbin_1_of_6 CHIME/CHIME_decbin_2_of_6 CHIME/CHIME_decbin_3_of_6 CHIME/CHIME_decbin_4_of_6 CHIME/CHIME_decbin_5_of_6"
# rep_surveys=CHIME/CHIME_decbin_3_of_6
# cd data/Surveys/
# rep_surveys=$(ls CHIME/*)
# rep_surveys=${rep_surveys//".ecsv"/""}
# cd $ZDM/zdm

echo "Outfile: $outfile.h5"
echo "Walkers: $walkers"
echo "Steps: $steps"

# command="python scripts/MCMC/MCMC_wrap.py -f $surveys -p data/MCMC/params.json -o $outfile -w $walkers -s $steps"
command="srun python scripts/MCMC/MCMC_wrap.py -f $surveys -r $rep_surveys -p data/MCMC/params3.json -o $outfile -w $walkers -s $steps --Pnr --pwb"
# command="python scripts/MCMC/MCMC_wrap.py -f $surveys -r $rep_surveys -p data/MCMC/params3.json -o $outfile -w $walkers -s $steps --Pn"
echo $command
$command