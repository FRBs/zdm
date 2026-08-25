from magnificationMapper import normalisedLensFuncsAcrossBeam
from astropy.io import fits
from astropy import wcs
import astropy
from astropy import units as u
from astropy import constants as const

import numpy as np
from zdm import survey
from matplotlib import pyplot as plt




magni = fits.getdata('hlsp_frontier_model_macs0717_bradac_v1_z01-magnif.fits')
info = fits.getheader('hlsp_frontier_model_macs0717_bradac_v1_z01-magnif.fits')
proj = wcs.WCS(info)

xMagni = np.meshgrid(np.arange(0,len(magni[:,0]),1), np.arange(0,len(magni[0,:]),1))
tempCoords = proj.array_index_to_world_values(xMagni[0], xMagni[1])
relBeamPositions = np.load('relBeamPos.npy')
surveyName = 'CHORD'
ratesArr=np.zeros(len(relBeamPositions[:,0]))

for i in range(len(relBeamPositions[:,0])):
    print('---Beam Pos:', i)
    name = 'CHORD_BeamPos_'+str(i)
    mux, pmux = normalisedLensFuncsAcrossBeam(48*u.m, 900*u.MHz, 1e-3, 10, np.array([np.mean(tempCoords[0])+relBeamPositions[i,0], np.mean(tempCoords[1])+relBeamPositions[i,1]]), proj, magni, name)
    np.save('mux_BP_'+str(i), np.log10(mux))
    np.save('pmux_BP_'+str(i), pmux)
    np.save('mus', np.log10(mux))
    np.save('pmus', pmux)
    fig, ax = plt.subplots()
    ax.plot(np.log10(mux), np.log10(pmux))
    ax.set_xlabel('$\log_{10}\\mu$')
    ax.set_ylabel('$\log_{10}p(\\mu)$')
    fig.savefig('beamMagDist/probs'+format(i, '02d'))
    plt.close()
