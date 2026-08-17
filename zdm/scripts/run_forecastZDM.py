from forecastZDM import pullTrigger
import numpy as np

args = []
gammas = np.array([-1])
ns = np.array([1])
print('pulling the trigger in serial')
for i in range(len(gammas)):
    for j in range(len(ns)):
        print('----->>> Executing: gamma:'+str(gammas[i])+', n:'+str(ns[j]))
        pullTrigger(0.18, 'Abell2218', gammas[i], ns[j], opdir='/arc/projects/chime_frb/msammons/clusterLensing/testZDM')

print('fin')
