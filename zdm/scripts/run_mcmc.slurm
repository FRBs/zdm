#!/bin/bash
#SBATCH --job-name=zdm_mcmc_parallel
#SBATCH --output=../mcmc/FASTnpCRAFT_all.out
#SBATCH --ntasks=20
#SBATCH --time=15:00:00
#SBATCH --export=NONE
# SBATCH --mem=30GB
#SBATCH --mem-per-cpu=2GB

# source /fred/oz002/jhoffmann/FRB_library/setup.sh
# source /fred/oz002/jhoffmann/FRB_library/loadzdm.sh

source /fred/oz002/jhoffmann/FRB_library/zdm_env/bin/activate

cd $ZDM/zdm

outfile="FASTnpCRAFT_all"
walkers=40
steps=1000

# surveys="FAST.ecsv"
# prefix="FAST"

# surveys="CRAFT_class_I_and_II.ecsv CRAFT_ICS.ecsv CRAFT_ICS_892.ecsv parkes_mb_class_I_and_II.ecsv"
# prefix="CRAFT"

# surveys="FAST.ecsv CRAFT_class_I_and_II.ecsv CRAFT_ICS.ecsv CRAFT_ICS_892.ecsv parkes_mb_class_I_and_II.ecsv"
# prefix="FASTnCRAFT"

surveys="FAST.ecsv CRAFT_class_I_and_II.ecsv private_CRAFT_ICS_892.ecsv private_CRAFT_ICS_1300.ecsv private_CRAFT_ICS_1632.ecsv parkes_mb_class_I_and_II.ecsv"
prefix="FASTnpCRAFT"

echo "Outfile: $outfile"
echo "Walkers: $walkers"
echo "Steps: $steps"

python MCMC_parallel.py $surveys -p tests/files/real_mini_mcmc.json -o $outfile -w $walkers -s $steps -i $prefix