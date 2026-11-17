import initialiseClusterContributions as iCC
import os



print('initialising magnifications')
iCC.initialise_Magni('/arc/projects/chime_frb/msammons/clusterLensing/testZDM', 0.18, 'Abell2218', 'RadialABELL_2218.fits', 0.18)

print('initialising DM and Scattering')
iCC.initialise_DM_Scattering('/arc/projects/chime_frb/msammons/clusterLensing/testZDM', 0.18, 'Abell2218', 'RadialABELL_2218.fits', 0.18 ,-1.0)



