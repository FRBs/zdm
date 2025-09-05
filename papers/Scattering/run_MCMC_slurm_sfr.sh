#!/bin/bash
#SBATCH --job-name=fit_scattering_test
#SBATCH --ntasks=10
#SBATCH --time=24:00:00
#SBATCH --export=NONE
#SBATCH --mem-per-cpu=700MB

# activate python environment
source /fred/oz313/cwjames/zdm/python_env/bin/activate


export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

####### ACTUAL RUN #######
version=$SLURM_ARRAY_TASK_ID
pfile="MCMC_inputs/scat_w_sfr.json"
files="CRAFT_ICS_892 CRAFT_ICS_1300 CRAFT_ICS_1632"
opfile="MCMC_outputs/v2mcmc_lognormal_sfr_v${version}"
Pn=False
ptauw=True
steps=500
walkers=20

Nz=100
Ndm=100
zmax=2
dmmax=2000

script="../../zdm/scripts/MCMC/MCMC_wrap.py"

runcommand="python $script -f $files --opfile=$opfile --pfile=$pfile --ptauw -s $steps -w $walkers --Nz=$Nz --Ndm=$Ndm --zmax=$zmax --dmmax=$dmmax"

echo $runcommand
$runcommand
