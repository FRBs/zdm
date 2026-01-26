""" 
This script creates zdm grids and plots localised FRBs

It can also generate a summed histogram from all CRAFT data

"""
import os

from zdm import cosmology as cos
from zdm import misc_functions
from zdm import parameters
from zdm import survey
from zdm import pcosmic
from zdm import iteration as it
from zdm.craco import loading
from zdm import io
from magnificationMapper import normalisedLensFuncsAcrossBeam
from astropy.io import fits
from astropy import wcs
import astropy
from astropy import units as u
from astropy import constants as const
import pickle
import numpy as np
from zdm import survey
from matplotlib import pyplot as plt

def renormalise(enorm,emin,emax,gamma):
    oldNorm = (emin**gamma-emax**gamma)
    newNorm = (enorm**gamma-emax**gamma)
    renormFactor = oldNorm/newNorm #multiply existing rates by this
    return renormFactor

def rollFRBSample(grid, N,renormEnergy=1e39):
    renormF = renormalise(renormEnergy, 10**Ugrid.state.energy.lEmin,10**Ugrid.state.energy.lEmax, np.asarray([Ugrid.state.energy.gamma]))
    zdmMesh = np.meshgrid(Ugrid.dmvals,Ugrid.zvals)
    tempBase = Ugrid.rates*10**Ugrid.state.FRBdemo.lC*renormF[0]*1024
    tempInts = np.random.choice(np.arange(len(tempBase.flatten())), p=tempBase.flatten()/np.sum(tempBase),size=N)
    es = np.linspace(Ugrid.state.energy.lEmin, Ugrid.state.energy.lEmax, 10000)
    efracs = Ugrid.array_cum_lf(10**es, 10**Ugrid.state.energy.lEmin, 10**Ugrid.state.energy.lEmax, Ugrid.state.energy.gamma, Ugrid.use_log10)
    efunc = interp1d(np.log10(efracs),es)
    subfractions = np.random.uniform(size=N)
    randEs = efunc(np.log10((Ugrid.fractions.flatten()/np.sum(Ugrid.beam_o))[tempInts]*subfractions))
    randZs = (zdmMesh[1].flatten())[tempInts]
    randDMs = (zdmMesh[0].flatten())[tempInts]
    return randZs, randDMs, randEs

def main(gamma, n, save_grid=False):
    # in case you wish to switch to another output directory
    #opdir = "Localised_FRBs/"
    #clusterRedshift = 0.38
    opdir = "~/"

    # Initialise surveys and grids

    sdir = "../data/Surveys/"
    state = parameters.State()
    state.energy.lEmax = 41.63
    state.energy.lEmin = 30
    state.energy.gamma = gamma
    state.energy.alpha = 1.03
    state.FRBdemo.sfr_n = n
    state.host.lsigma = 0.57
    state.host.lmean = 2.22
   #state.FRBdemo.lC = -0.49
    state.FRBdemo.lC = 2.3-9
    state.energy.luminosity_function=0
    state.FRBdemo.alpha_method=1


    surveyName = 'CHIME_SynthBeam'
    s,g = loading.survey_and_grid(survey_name=surveyName, opdir=opdir,
        NFRB=None,sdir=sdir,init_state=state)

    z,d,e = rollFRBSample(g, N_frbs)

    np.save(opdir+'Unlensedthresh5Gamma'+str("{:.2f}".format(gamma))+'SFRn'+str("{:.2f}".format(n))+'N'+str(N_frb)+'sampleRedshift',z)
    np.save(opdir+'Unlensedthresh5Gamma'+str("{:.2f}".format(gamma))+'SFRn'+str("{:.2f}".format(n))+'N'+str(N_frb)+'sampleDM',d)
    np.save(opdir+'Unlensedthresh5Gamma'+str("{:.2f}".format(gamma))+'SFRn'+str("{:.2f}".format(n))+'N'+str(N_frb)+'sampleEnergy',e)
   
    
    
    # Save
    if save_grid:
        print('saving grid')
        with open(opdir+'GridUnlensedthresh5Gamma'+str("{:.2f}".format(gamma))+'SFRn'+str("{:.2f}".format(n)), "wb") as f:
            pickle.dump(g, f)
    print('fin')

if __name__ == "__main__":
    gamma=-1
    nsfr = 1
    N_frbs
    main(gamma, nsfr, N_frbs)

