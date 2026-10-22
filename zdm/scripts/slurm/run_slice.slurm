#!/bin/bash
#SBATCH --job-name=slice
#SBATCH --output=../../cube/slice_nsfr.out
#SBATCH --ntasks=1
#SBATCH --time=3:00:00
#SBATCH --export=NONE
#SBATCH --mem=4GB
# SBATCH --mem-per-cpu=5GB

source $ZDM/.venv/bin/activate

cd $ZDM/zdm

# surveys="DSA no_Tobs/FAST no_Tobs/CRAFT_class_I_and_II no_Tobs/parkes_mb_class_I_and_II no_Tobs/MeerTRAPcoherent no_Tobs/MeerTRAPincoherent"
surveys="DSA FAST CRAFT_class_I_and_II parkes_mb_class_I_and_II MeerTRAPcoherent MeerTRAPincoherent"
rsurveys="CRAFT_average_ICS CHIME/CHIME_decbin_0_of_6 CHIME/CHIME_decbin_1_of_6 CHIME/CHIME_decbin_2_of_6 CHIME/CHIME_decbin_3_of_6 CHIME/CHIME_decbin_4_of_6 CHIME/CHIME_decbin_5_of_6"
# surveys=MC_CRAFT_ICS_1300_2
# surveys=FAST
# rsurveys=CHIME/CHIME_decbin_4_of_6
# surveys="DSA_cat1_updated DSA_cat1_no_20220319D"
# surveys="CRAFT_ICS_1300_debug"

# command="python scripts/run_slice.py gamma -4 1 -f $surveys -r $rsurveys -n 20"
command="python scripts/run_slice.py sfr_n 0 10 -f $surveys -r $rsurveys -n 20"
echo $command
$command

# python scripts/run_slice.py lEmax 39.5 42.0 -f $surveys -n 10
# python scripts/run_slice.py lRmin -5.0 -1.0 -r $rsurveys -n 10