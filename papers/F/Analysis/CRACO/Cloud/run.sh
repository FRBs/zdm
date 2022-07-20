#!/bin/bash

#SBATCH --job-name=craco_mini      # Job name
#SBATCH --partition=cpuq             # queue for job submission
#SBATCH --account=cpuq               # queue for job submission
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jmbaptis@ucsc.edu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=24:00:00
#SBATCH --output=craco_mini_%j.log

module load python/3.8.6

python3.8 run_craco_mini.py -n 25 -t 25 -b 1