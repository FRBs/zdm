#!/bin/bash
#SBATCH --job-name=mc_sample
#SBATCH --output=../craco/mc_sample.out
#SBATCH --ntasks=1
#SBATCH --time=7:00:00
#SBATCH --export=NONE
#SBATCH --mem=10GB
# SBATCH --mem-per-cpu=5GB

###############################################################################
#   Author: Jordan Hoffmann                                                   #
#   Date:   10/07/2024                                                        #
#   Purpose:                                                                  #
#       Slurm script to generate an MC sample                                 #
###############################################################################

source $ZDM/.zdm_env/bin/activate

cd $ZDM/zdm/craco/

python mc.py
