#!/bin/bash
#SBATCH --job-name=CRAFT
#SBATCH --output=../mcmc/Hoffmann2023_CRAFT_no_191228.out
#SBATCH --ntasks=20
#SBATCH --time=24:00:00
#SBATCH --export=NONE
# SBATCH --mem=50GB
#SBATCH --mem-per-cpu=3GB

# source /fred/oz002/jhoffmann/FRB_library/setup.sh
# source /fred/oz002/jhoffmann/FRB_library/loadzdm.sh

source $ZDM/.zdm_env/bin/activate

cd $ZDM/zdm

# outfile="Hoffmann2023_CRAFT_no_191228"
outfile="DSA"
walkers=40
steps=1000

surveys="DSA3.ecsv" #"parkes_mb_class_I_and_II.ecsv" # DSA3.ecsv FAST.ecsv CRAFT_class_I_and_II.ecsv private_CRAFT_ICS_892.ecsv private_CRAFT_ICS_1300.ecsv private_CRAFT_ICS_1632.ecsv 
prefix="DSAtest"

# cd data/Surveys/
# surveys=$(ls -d -- Hoffmann2023/*[0-9]_exact.ecsv)
# prefix="Hoffmann2023_exact_no_191228"
# cd $ZDM/zdm

# cd data/Surveys/
# surveys=$(ls Hoffmann2023/*.ecsv)
# prefix="Hoffmann2023_exact"
# cd $ZDM/zdm

# surveys="Hoffmann2023_CRAFT/private_CRAFT_ICS_892.ecsv Hoffmann2023_CRAFT/private_CRAFT_ICS_1300.ecsv"
# prefix="Hoffmann2023_CRAFT"

# surveys="Hoffmann2023_CRAFT/private_CRAFT_ICS_892.ecsv Hoffmann2023_CRAFT/private_CRAFT_ICS_1300_no_191228.ecsv"
# prefix="Hoffmann2023_CRAFT_no_191228"

# surveys="FAST.ecsv"
# prefix="FAST"

# surveys="CRAFT_class_I_and_II.ecsv CRAFT_ICS.ecsv CRAFT_ICS_892.ecsv parkes_mb_class_I_and_II.ecsv"
# prefix="CRAFT"

# surveys="FAST.ecsv CRAFT_class_I_and_II.ecsv CRAFT_ICS.ecsv CRAFT_ICS_892.ecsv parkes_mb_class_I_and_II.ecsv"
# prefix="FASTnCRAFT"

# surveys="FAST.ecsv CRAFT_class_I_and_II.ecsv private_CRAFT_ICS_892.ecsv private_CRAFT_ICS_1300.ecsv private_CRAFT_ICS_1632.ecsv parkes_mb_class_I_and_II.ecsv"
# prefix="FASTnpCRAFT"

# surveys="FAST_no_tobs.ecsv CRAFT_class_I_and_II.ecsv private_CRAFT_ICS_892.ecsv private_CRAFT_ICS_1300.ecsv private_CRAFT_ICS_1632.ecsv parkes_mb_class_I_and_II.ecsv"
# prefix="FAST_no_tobs"

echo "Outfile: $outfile.out"
echo "Walkers: $walkers"
echo "Steps: $steps"

python MCMC_wrap.py $surveys -p tests/files/real_mini_mcmc2.json -o $outfile -w $walkers -s $steps -i $prefix -n 1