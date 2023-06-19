#!/bin/bash
#SBATCH --job-name=zdm_mcmc_parallel
#SBATCH --output=mcmc/w40_s2000_2.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --time=20:00:00
#SBATCH --export=NONE
#SBATCH --mem=30GB
# SBATCH --mem-per-cpu=2GB

# source /fred/oz002/jhoffmann/FRB_library/setup.sh
# source /fred/oz002/jhoffmann/FRB_library/loadzdm.sh

cd $ZDM

python MCMC_parallel.py CRAFT_class_I_and_II.ecsv CRAFT_ICS.ecsv CRAFT_ICS_892.ecsv CRAFT_ICS.ecsv parkes_mb_class_I_and_II.ecsv -p tests/files/real_mini_cube.json -o w40_s2000_2 -w 40 -s 2000