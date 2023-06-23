#!/bin/bash
#SBATCH --job-name=H0_slice
#SBATCH --output=H0_slice.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00
#SBATCH --export=NONE
#SBATCH --mem=1G

cd $ZDM/papers/FREDDA_efficiency/scripts

python H0_with_efficiencies.py