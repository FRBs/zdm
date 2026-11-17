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
import astropy
from astropy import units as u
from astropy import constants as const
import pickle
import numpy as np
from zdm import survey
from matplotlib import pyplot as plt
import scipy.stats
from astropy.cosmology import Planck18 as cosmo
from scipy.interpolate import interp1d

def renormalise(enorm,emin,emax,gamma):
    oldNorm = (emin**gamma-emax**gamma)
    newNorm = (enorm**gamma-emax**gamma)
    renormFactor = oldNorm/newNorm #multiply existing rates by this
    return renormFactor

def F_to_E(F,z,alpha=0, bandwidth=1e9, Fobs=1.3e9, Fref=1.3e9):                               
    """ Converts a fluence in Jy ms to an energy in erg
    Formula from Macquart & Ekers 2018
    Works with an array of z.
    
    Arguments are:
        Fluence: of an FRB [Jy ms]
        
        Redshift: assumed redshift of an FRB producing the fluence F.
            Standard cosmological definition [unitless]
        
        alpha: F(\nu)~\nu^-\alpha. Note that this is an internal definition.
            The paper uses ^alpha, not ^-alpha. [unitless]
    
        Bandwidth: over which to integrate fluence [Hz] 
        
        Fobs: the observation frequency [Hz]
        
        Fref: reference frequency at which FRB energies E are normalised.
            It defaults to 1.3 GHz (ASKAP lat50, Parkes).
    
    Return value: energy [erg]
    
    """
    E=F*4*np.pi*(cosmo.luminosity_distance(z).value)**2/(1.+z)**(2.-alpha)
    # now convert from dl in MPc and F in Jy ms
    # 10^-26 from Jy to W per m2 per Hz
    # 1e-3 from Jy ms to J per m2 per Hz
    # (3.086e16 m in 1 pc x 10^6 Mpc)^2 for dl in m
    # 1e7 from J to erg
    # total factor is 9.523396e22
    E *= 9.523396e22*bandwidth
 
    # now corrects for reference frequency
    # according to value of alpha
    # effectively: if fluence was X at F0, it was X*(F0/Fref)**alpha at Fref
    # i.e. if alpha is positive (stronger at low frequencies), we reduce E
    # This acts to reduce the telescope threshold at higher frequencies
    E *= (Fobs/Fref)**alpha
 
    return E


def rollFRBSample(Ugrid, N,renormEnergy=1e39):
    renormF = renormalise(renormEnergy, 10**Ugrid.state.energy.lEmin,10**Ugrid.state.energy.lEmax, np.asarray([Ugrid.state.energy.gamma]))
    zdmMesh = np.meshgrid(Ugrid.dmvals,Ugrid.zvals)
    tempBase = Ugrid.rates*10**Ugrid.state.FRBdemo.lC*renormF[0]*1024
    tempInts = np.random.choice(np.arange(len(tempBase.flatten())), p=tempBase.flatten()/np.sum(tempBase),size=N)
    es = np.linspace(Ugrid.state.energy.lEmin, Ugrid.state.energy.lEmax+2, 10000)
    efracs = Ugrid.array_cum_lf(10**es, 10**Ugrid.state.energy.lEmin, 10**Ugrid.state.energy.lEmax, Ugrid.state.energy.gamma, Ugrid.use_log10)
    efunc = interp1d(np.log10(efracs),es)
    subfractions = np.random.uniform(size=N)
    randZs = (zdmMesh[1].flatten())[tempInts]
    randDMs = (zdmMesh[0].flatten())[tempInts]
    Fth=5
    eth = F_to_E(Fth,randZs)
    ethFrac = Ugrid.array_cum_lf(eth, 10**Ugrid.state.energy.lEmin, 10**Ugrid.state.energy.lEmax, Ugrid.state.energy.gamma)
    randEs = efunc(np.log10((ethFrac)*subfractions))
    return randZs, randDMs, randEs

def main(gamma, n, emax, N_frbs, save_grid=False):
    # in case you wish to switch to another output directory
    #opdir = "Localised_FRBs/"
    #clusterRedshift = 0.38
    opdir = "./"

    # Initialise surveys and grids

    sdir = "../data/Surveys/"
#    state = parameters.State()
#    state.energy.lEmax = 42.63
#    state.energy.lEmin = 30
#    state.energy.gamma = gamma
#    state.energy.alpha = 1.03
#    state.FRBdemo.sfr_n = n
#    state.host.lsigma = 0.57
#    state.host.lmean = 2.22
#   #state.FRBdemo.lC = -0.49
#    state.FRBdemo.lC = 2.3-9
#    state.energy.luminosity_function=0
#    state.FRBdemo.alpha_method=1

    state = parameters.State()
    state.energy.lEmax = emax
    state.energy.lEmin = 39
    state.energy.gamma = gamma
    state.energy.alpha = 0
    state.FRBdemo.sfr_n = n
    state.host.lsigma = 0.41
    state.host.lmean = 1.93
    state.FRBdemo.lC = 2.3-9
    #state.FRBdemo.lC = 2.3-9+1.31
    state.energy.luminosity_function=2
    state.FRBdemo.alpha_method=0


    surveyName = 'CHIME_SynthBeam'
    s,g = loading.survey_and_grid(survey_name=surveyName, opdir=opdir,
        NFRB=None,sdir=sdir,init_state=state)

    z,d,e = rollFRBSample(g, N_frbs)

    np.save(opdir+'KaitUnlensedthresh5Gamma'+str("{:.2f}".format(gamma))+'SFRn'+str("{:.2f}".format(n))+'Emax'+str("{:.2f}".format(emax))+'N'+str(N_frbs)+'sampleRedshift',z)
    np.save(opdir+'KaitUnlensedthresh5Gamma'+str("{:.2f}".format(gamma))+'SFRn'+str("{:.2f}".format(n))+'Emax'+str("{:.2f}".format(emax))+'N'+str(N_frbs)+'sampleDM',d)
    np.save(opdir+'KaitUnlensedthresh5Gamma'+str("{:.2f}".format(gamma))+'SFRn'+str("{:.2f}".format(n))+'Emax'+str("{:.2f}".format(emax))+'N'+str(N_frbs)+'sampleEnergy',e)
   
    
    
    # Save
    if save_grid:
        print('saving grid')
        with open(opdir+'KaitGridUnlensedthresh5Gamma'+str("{:.2f}".format(gamma))+'SFRn'+str("{:.2f}".format(n))+'Emax'+str("{:.2f}".format(emax)), "wb") as f:
            pickle.dump(g, f)
    print('fin')

if __name__ == "__main__":
    gamma=-0.01
    nsfr = 0.96
    N_frbs=1000
    emax = 41.38
    main(gamma, nsfr, emax, N_frbs)

