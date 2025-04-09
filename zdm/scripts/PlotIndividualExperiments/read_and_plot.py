"""
A simpel script for reading in and plotting numpy files saved
by the plotting scripts
"""

import numpy as np
from matplotlib import pyplot as plt


# example
name="MeerTRAP"
indir=name+"/"

# loads data
pz = np.load(indir+name+"_pz.npy")
zvals = np.load(indir+"zvalues.npy")
    
pdm = np.load(indir+name+"_pDM.npy")
dmvals = np.load(indir+"DMvalues.npy")
    
zDM = np.load(indir+name+"_zDM.npy")


# makes p(DM) plot
plt.figure()
plt.plot(dmvals,pdm)
plt.xlabel("${\\rm DM}_{\\rm EG}$ [pc cm$^{-3}$]")
plt.ylabel("$p({\\rm DM}_{\\rm EG})$ [a.u.]")
plt.tight_layout()
plt.savefig(indir+name+"_pDM.pdf")
plt.close()

# makes p(z) plot
plt.figure()
plt.plot(zvals,pz)
plt.xlabel("$z$")
plt.ylabel("$p(z)$ [a.u.]")
plt.tight_layout()
plt.savefig(indir+name+"_pz.pdf")
plt.close()

# makes p(z-DM) plot. Note that the normalisation of this one makes sense!
plt.figure()
# makes this logscale
logzDM = np.log10(zDM)
# note: 'extent' is slightly wrong, since these values are bin centres, while
# extent plots these as box edges...
extent=[zvals[0],zvals[-1],dmvals[0],dmvals[-1]]
plt.xlabel('$z$')
plt.ylabel('${\\rm DM}_{\\rm EG}$')
aspect = zvals[-1]/dmvals[-1]
# plots the z-DM grid. The units are log10(FRBs per 317.5 days per cell)
im=plt.imshow(logzDM.T,origin='lower',extent=extent,aspect=aspect)
cbar = plt.colorbar()
themin = np.floor(np.max(logzDM)-4)
themax = np.ceil(np.max(logzDM))
im.set_clim(themin,themax)
cbar.set_label("$\\log_{10} (N_{\\rm FRB} / {\\rm bin})$")
plt.tight_layout()
plt.savefig(indir+name+"_zDM.pdf")
plt.close()
