#!/bin/bash
#SBATCH --job-name=zdm_mcmc_parallel
#SBATCH --output=mcmc/w32_s1000_wide.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=9:00:00
#SBATCH --export=NONE
#SBATCH --mem=20GB
# SBATCH --mem-per-cpu=2GB

# source /fred/oz002/jhoffmann/FRB_library/setup.sh
# source /fred/oz002/jhoffmann/FRB_library/loadzdm.sh

cd $ZDM

python MCMC_parallel.py CRAFT_class_I_and_II.ecsv CRAFT_ICS.ecsv CRAFT_ICS_892.ecsv CRAFT_ICS.ecsv parkes_mb_class_I_and_II.ecsv -p tests/files/real_mini_cube.json -o w32_s1000_wide -w 32 -s 1000