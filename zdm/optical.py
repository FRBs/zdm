"""
This file contains functions related to optical observations.

Currently, this is simply the approximate fraction of
detectable FRBs from Marnoch et al (https://doi.org/10.1093/mnras/stad2353)
"""


import numpy as np
from matplotlib import pyplot as plt



def p_unseen(zvals,plot=False):
    """
    Returns probability of a hist being unseen in typical VLT
    observations.
    
    Inputs:
        zvals [float, array]: array of redshifts
    
    Returns:
        fitv [float, array]: p(Unseen) for redshift zvals
    
    """
    # approx digitisation of Figure 3 p(U|z)
    # from Marnoch et al.
    # https://doi.org/10.1093/mnras/stad2353
    rawz=[0.,0.7,0.8,0.9,0.91,0.98,1.15,1.17,1.25,
        1.5,1.7,1.77,1.85,1.95,2.05,2.4,2.6,2.63,
        2.73,3.05,3.75,4.5,4.9,5]
    rawz=np.array(rawz)
    prawz = np.linspace(0,1.,rawz.size)
    
    pz = np.interp(zvals, rawz, prawz)
    
    
    coeffs=np.polyfit(rawz[1:],prawz[1:],deg=3)
    fitv=np.polyval(coeffs,zvals)
    
    if plot:
        plt.figure()
        plt.xlim(0,5.)
        plt.xlabel("$z$")
        plt.ylim(0,1.)
        plt.ylabel('$p_{\\rm U}(z)$')
        plt.plot(rawz,prawz,label='Marnoch et al.')
        plt.plot(zvals,pz,label='interpolation')
        plt.plot(zvals,fitv,label='quadratic')
        plt.legend()
        plt.tight_layout()
        plt.savefig('p_unseen.pdf')
        plt.close()
    
    # checks against values which are too low
    toolow = np.where(fitv<0.)[0]
    fitv[toolow]=0.
    
    # checks against values which are too high
    toohigh = np.where(fitv>1.)[0]
    fitv[toohigh]=1.
    
    return fitv
