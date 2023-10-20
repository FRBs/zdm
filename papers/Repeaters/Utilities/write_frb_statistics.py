import utilities as ute
import numpy as np


names,decs,dms,dmegs,snrs,reps,ireps,widths,nreps=ute.get_chime_data(snrcut=12)


# FRB20190417A is not a repeater

nreps = np.array(nreps)
OK = np.where(nreps > 1.)[0]
nreps = nreps[OK]


Nreps = len(nreps)
# method of crawford

Nsingles = len(np.where(reps==0)[0])

print(Nreps)

M = Nreps + Nsingles
thesum = np.sum(np.log(nreps))
print("The sum found to be ",thesum)
inva = thesum/M
print("The average log is ",inva)
a = inva**-1
print("Found a to be ",a)
