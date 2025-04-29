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

# surveys="DSA FAST CRAFT_class_I_and_II CRAFT_ICS_892 CRAFT_ICS_1300 CRAFT_ICS_1632 parkes_mb_class_I_and_II"
rsurveys="CHIME/CHIME_decbin_0_of_6 CHIME/CHIME_decbin_1_of_6 CHIME/CHIME_decbin_2_of_6 CHIME/CHIME_decbin_3_of_6 CHIME/CHIME_decbin_4_of_6 CHIME/CHIME_decbin_5_of_6"
surveys=MC_CRAFT_ICS_1300_2
# surveys=CRAFT_class_I_and_II

# python scripts/run_slice_benchmark.py $surveys DMhalo 100.0 100.0 -n 1
# python scripts/run_slice.py $surveys DMhalo 50.0 50.0 -n 1
# python scripts/run_slice.py $surveys lEmin 35.0 41.0 -n 20
# python scripts/run_slice_MCMC2.py $surveys lEmax 40.0 43.0 -n 21
# python scripts/run_slice.py $surveys lmean 0.5 2.5 -n 5
# python scripts/run_slice.py gamma -2.0 -0.5 -f $surveys -n 10
python scripts/run_slice.py Rgamma -3.0 -1.0 -r $rsurveys -n 10
# python scripts/run_slice.py lRmin -4.0 2.9 -r $rsurveys -n 5
# python scripts/run_slice.py $rsurveys lRmax -3.9 2.0 -n 10 -r
# python scripts/run_slice.py DMhalo 10.0 80.0 -f $surveys -n 15