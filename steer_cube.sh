#!/bin/bash -l


############# python library loading #########


# these are the modules that must be loaded for Pawsey's Magnus cluster
pythonfile=cube.py
module load python
module load scipy
module load numpy
module load matplotlib
module load argparse

###### job specific input ########

# this is which job is in the array
sid=$SLURM_ARRAY_TASK_ID

# in case we have to loop over more than 1000 array jobs
nthousand=$1
n2=$((nthousand*1000+sid))

# this is the file name containing the variables to loop over
pfile=$2

# this is the number of runs per process
many=$3

# this is the number of independent processes
nprocess=$4

# we now calculate the starting point: n2*nprocess. Ooh... maybe difficult to start and end at the right point?
nstart=$((n2*nprocess-nprocess+1))
nend=$((n2*nprocess))

fullpath=/group/askap/cjames/FRB_library/Cube/$pfile

opdir=/group/askap/cjames/FRB_library/Cube
logdir=/group/askap/cjames/FRB_library/logs
t0=`date +%s`
for n in `seq $nstart $nend`; do
	echo "Starting $n at time $t0"
	opfile=$opdir/${n}_${many}_$pfile
	logfile=$logdir/log_${n}_${many}_$pfile
	command="python3 $pythonfile -n $n -m $many -p $fullpath -o $opfile"
	echo "$command >> $logfile &"
	$command >> $logfile &
done

echo "launching finished"
wait
t1=`date +%s`
dt=$((t1-t0))
echo "everything done after time $dt"

