"""
Simple script to plot the "avals": values of
beam sensitivity for ASKAP as a function of
distance from boresight.

These "avals" are taken from James et al.

"""

import numpy as np
from matplotlib import pyplot as plt


#### Defines theta offsets from boresights [deg] #####
thetas = np.linspace(0,4,51)

##### Loads in avals from Hotan et al #####
data = np.loadtxt("avals.dat")
x = data[:,0]
y = data[:,1]
h_avals = np.interp(thetas,x,y)

##### Calculates avals from James et al. #####
theta_off = 0.8
sigma_theta = 3.47

small = np.where(thetas < theta_off)

ftheta = np.exp(-0.5 * ((thetas-theta_off)/sigma_theta)**2.)
ftheta[small] = 1.

##### plots the values ######
plt.figure()

plt.plot(thetas,ftheta,label="James et al")
plt.plot(thetas,h_avals,label="Hotan et al")
plt.xlim(0,4.)
plt.ylim(0.63,1.06)
plt.xlabel("Offset from boresight [deg]")
plt.ylabel("Relative beam amplitude [B/B(0)]")
plt.legend()
plt.tight_layout()
plt.savefig('avals.pdf')
plt.close()
