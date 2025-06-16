#!/bin/bash

# script to run MCMC for CRAFT width parameters

#files="CRAFT_ICS_892 CRAFT_ICS_1300 CRAFT_ICS_1632"

# use this for a halflognormal distribution
#opfile="v3_mcmc_halflognormal"
#pfile="MCMC_inputs/scat_w_only_halflog.json"
# use this for a lognormal distribution
opfile="v6_mcmc_lognormal" # v4 is done with 300x300 nz ndm bins
pfile="MCMC_inputs/scat_w_only.json"

# LOG
# files="CRAFT_ICS_892 CRAFT_ICS_1300 CRAFT_ICS_1632"
# v2: 1000 I think
# v3: 5000
# v4: 300 x 300 zDM values
# v5: try with 100x100, but with 15 beam values instead of 5.
#v6 Unlocalised ones removed ("mod")
# [I still don't know the origin of these weird ridges in scat-sigmascat space]

# for v6: these are now all localised - z=-1 is removed
#files="modCRAFT_ICS_892 modCRAFT_ICS_1300 modCRAFT_ICS_1632"
#opfile="v6_mcmc_lognormal"

# for v7 - only one survey. Faster!  Turns off the P(w) function
files="modCRAFT_ICS_1300"
opfile="MCMC_outputs/v7_mcmc_lognormal"

# for v8 - turns off the Pscat | w function. (in p(2D) only)
#files="modCRAFT_ICS_1300" # takes away 1D as well, only 2D
#opfile="v8_mcmc_lognormal"

# for v8 - has 1000 internal bins in width, and 33 evaluation bins
files="modCRAFT_ICS_1300" # takes away 1D as well, only 2D
opfile="MCMC_outputs/v9_mcmc_lognormal"

Pn=False
ptauw=True
steps=1000
walkers=14

Nz=100
Ndm=100
zmax=2
dmmax=2000

script="../../zdm/scripts/MCMC/MCMC_wrap.py"

runcommand="python $script -f $files --opfile=$opfile --pfile=$pfile --ptauw -s $steps -w $walkers --Nz=$Nz --Ndm=$Ndm --zmax=$zmax --dmmax=$dmmax"

echo $runcommand
$runcommand
