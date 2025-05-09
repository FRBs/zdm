"""
This script performs simple calculations to estimate the true energy of the FRB
"""

from zdm.MC_sample import loading
import numpy as np
from zdm import cosmology as cos

state = loading.set_state()
cos.set_cosmology(state)
cos.init_dist_measures()

SEFD=2000 #Jy ms
zarray = np.linspace(0.1,2,20)
SNR=34.2 #From Ryan Shannon, intrinsic uncertainty +-1
BW = 336e6
t=1.182e-3 # sampling time in seconds
w=6 # bin width used by Ryan
npol=2
NANT=23
RMS = SEFD / (NANT * npol * BW)**0.5 * (t*w)**0.5
F = SNR * RMS * 1e3 # converts to Jy ms not Jy s

print("Estimated fluence IF it was 1 ms is ",F)

# distances from beam centre in degrees
#olddbeam = 0.72
newdbeam = 0.475
#newdbeam = 0.72
# Gaussian correction
c_light = 299792458.0
fbar = 1.2755e9
wavelength = c_light/fbar
ASKAP_D = 12. # diameter in metres
FWHM = 1.1 * wavelength/ASKAP_D
sigma = FWHM/2.355 * 180./np.pi
print("Sigma is ",FWHM,sigma)
#oldB = np.exp(-0.5*olddbeam**2/sigma**2)
newB = np.exp(-0.5*newdbeam**2/sigma**2)
F = F/newB
print("Beam correction is ",newB," so new fluence is ",F)

for i,z in enumerate(zarray):
    E1 = cos.F_to_E(F,z,alpha=0,bandwidth=1e9)
    E2 = cos.F_to_E(F,z,alpha=1.5,bandwidth=1e9)
    print("At redshift of ",z," we have energy of ",E1," with k-correction ",E2)

