#!/bin/bash
#SBATCH --job-name=slice
#SBATCH --output=../cube/slice.out
#SBATCH --ntasks=1
#SBATCH --time=0:30:00
#SBATCH --export=NONE
#SBATCH --mem=4GB
# SBATCH --mem-per-cpu=5GB

source $ZDM/.venv/bin/activate

cd $ZDM/zdm

surveys="DSA_cat1_updated FAST CRAFT_class_I_and_II CRAFT_ICS_892 CRAFT_ICS_1300 CRAFT_ICS_1632 parkes_mb_class_I_and_II"
rsurveys="CHIME/CHIME_decbin_0_of_6 CHIME/CHIME_decbin_1_of_6 CHIME/CHIME_decbin_2_of_6 CHIME/CHIME_decbin_3_of_6 CHIME/CHIME_decbin_4_of_6 CHIME/CHIME_decbin_5_of_6"
# surveys=MC_CRAFT_ICS_1300_2
# surveys=CRAFT_class_I_and_II

python scripts/run_slice.py gamma -2.0 1.5 -f $surveys -n 20
# python scripts/run_slice.py lEmax 39.5 42 -f $surveys -n 10