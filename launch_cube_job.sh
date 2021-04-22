#!/bin/bash

script=steer_cube.sh

# defines the 'cube' of parameters over which to calculate
pfile=all_params.dat
# this file defines (3rd column) 1 x 11 x 7 x 11 x 26 x 10 x 10
# parameters (-1 for constant means direct optimisation is this parameters)
# If 260 jobs per GPU, and 24 GPUs/node, this means
# 11 x 7 x 11 x 26 x 10 x 10 / 24 / 260 = 353 jobs

here=`pwd`
logdir=$here/logs
mkdir -vp $logdir

#number of GPUs on node
nprocess=24

# number of cube calculations per node
# 26 at a time since iterating over number of 'n' is quick
# other parameters are slower (order of iteration is
# partially optimsied)
# this took about 3000 seconds on Pawsey's magnus cluster
many=260

# required total memory - this is approximate
mem=$((nprocess*700))

# other
opts="--cpus-per-task=1 --ntasks-per-node=$nprocess --account=ja3 -p workq --time=01:00:00 --output=$logdir/mpi3-%j.log --error=$logdir/mpi3-%j.err --nodes=1 --mem=$mem"


nthousand=0
# submit jobs 1 to 300.
for hundred in `seq 0 2`; do
	start=${hundred}01
	hundred2=$((hundred+1))
	stop=${hundred2}00
	
	command="sbatch $opts --array=$start-$stop $script $nthousand $pfile $many $nprocess"
	echo $command
	$command
done
