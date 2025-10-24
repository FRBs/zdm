"""
This file keeps a record of known states from specific papers
"""

from zdm import parameters
import numpy as np
from astropy.cosmology import Planck18

def load_state(case="HoffmannHalo25",scat=None,rep=None):
    """
    Routine to set state variables according to the methods used
    in specific previous works.
    
    Papers implemented are:
    
    JamesScattering25:
        Updated scattering model
    
    HoffmannHalo25:
        Investigation of MW halo
    
    HoffmannEmin25:
        Implementation of MCMC algorithm with FAST and DSA
    
    James2: Original zDM paper
    """
    
    #### primary fit params ####
    cases=["JamesSFR22","JamesH022","BaptisaF24",\
        "HoffmannEmin25","HoffmannHalo25"]
    
    if not case in cases:
        raise ValueError("Case ",case," Undefined, please choose from ",cases)
    
    # parameter dict to hold values to be passed to state
    vparams={}
    vparams = set_fit_params(vparams,case)
    
    ###### scattering ######
    scats=["orig","CHIME","updated"]
    if scat is not None:
        # just use the user-defined scattering version. Check if this exists!
        if not scat in scats:
            raise ValueError("Case ",scat," undefined, please choose from ",scats)
    elif case == "JamesSFR22":
        scat="orig"
        vparams = set_orig_scat(vparams)
    elif case == "JamesH022" or case == "BaptisaF24" \
            or case == "HoffmannEmin25" or case == "HoffmannHalo25":
        scat="CHIME"
        vparams = set_chime_scat(vparams)
    else:
        scat="updated"
        vparams = set_updated_scat(vparams)
    
    
    #### repetition ####
    # four cases for CHIME repeaters paper
    reps = ["a","b","c","d"]
    if rep is not None:
        if not rep in reps:
            raise ValueError("repeaters ",rep," undefined, please choose from",reps)
        else:
            print(rep)
            vparams = set_reps(vparams,rep)
    
    # initialise state with cosmological parameters
    state = parameters.State()
    state.set_astropy_cosmo(Planck18) # we always do this
    
    # update with relevant values
    state.update_param_dict(vparams)
    
    return state

def set_reps(vparams,rep):
    """
    Sets repetition parameters from JamesRepeaters24.
    
    There are four cases to choose from - see that paper
    """
    if not "rep" in vparams:
        vparams["rep"] = {}
    
    vparams["rep"]["RE0"] = 1.e39 # normalisation energy in ergs
    
    if rep == "a":
        vparams["rep"]["lRmin"] = -1.73
        vparams["rep"]["lRmin"] = -0.25
        vparams["rep"]["Rgamma"] = -1.9
    elif rep== "b":
        vparams["rep"]["lRmin"] = -1.23
        vparams["rep"]["lRmin"] = -0.25
        vparams["rep"]["Rgamma"] = -3
    elif rep== "c":
        vparams["rep"]["lRmin"] = -1.38
        vparams["rep"]["lRmin"] = 3
        vparams["rep"]["Rgamma"] = -3
    elif rep== "d":
        vparams["rep"]["lRmin"] = -4.54
        vparams["rep"]["lRmin"] = 3
        vparams["rep"]["Rgamma"] = -2.1
    return vparams

def set_fit_params(vparams,case):
    """
    sets best-fit standard fit parameters,
    as returned from the MCMC
    
    
    """
    
    
    # adds any missing categories
    for param in ['FRBdemo','MW','cosmo','IGM','energy','rep','IGM','host']:
        if not param in vparams:
            vparams[param] = {}
    
    
    if case == "JamesSFR22": #
        vparams['energy']['lEmin'] = 30
        vparams['energy']['lEmax'] = 41.7
        vparams['energy']['alpha'] = 1.55
        vparams['energy']['gamma'] = -1.09
        vparams['energy']['luminosity_function'] = 0
        
        vparams['FRBdemo']['alpha_method'] = 0
        vparams['FRBdemo']['source_evolution'] = 0
        vparams['FRBdemo']['sfr_n'] = 1.67
        #vparams['FRBdemo']['lC'] = 3.15 # incorrect, check
        
        vparams['host']['lmean'] = 2.11
        vparams['host']['lsigma'] = 0.53
        
        vparams['MW']['DMhalo']=50
        vparams['MW']['halo_method']=0
        vparams['MW']['sigmaHalo']=0
        vparams['MW']['sigmaDMG']=0
        
        vparams['IGM']['logF'] = np.log10(0.32)
        
        vparams['cosmo']['H0'] = 70.
        
        
    elif case == "JamesH022":
        vparams['energy']['lEmin'] = 30
        vparams['energy']['lEmax'] = 41.26
        vparams['energy']['alpha'] = 0.99
        vparams['energy']['gamma'] = -0.95
        vparams['energy']['luminosity_function'] = 2
        
        vparams['FRBdemo']['alpha_method'] = 1
        vparams['FRBdemo']['source_evolution'] = 0
        vparams['FRBdemo']['sfr_n'] = 1.13
        #vparams['FRBdemo']['lC'] = 3.15 # incorrect, check
        
        vparams['host']['lmean'] = 2.27
        vparams['host']['lsigma'] = 0.55
        
        vparams['MW']['DMhalo']=50
        vparams['MW']['halo_method']=0
        vparams['MW']['sigmaHalo']=0
        vparams['MW']['sigmaDMG']=0
        
        vparams['IGM']['logF'] = np.log10(0.32)
        
        vparams['cosmo']['H0'] = 73.0
    
    elif case == "BaptisaF24":
        vparams['energy']['lEmin'] = 30
        vparams['energy']['lEmax'] = 41.40
        vparams['energy']['alpha'] = 0.65
        vparams['energy']['gamma'] = -1.01
        vparams['energy']['luminosity_function'] = 2
        
        vparams['FRBdemo']['alpha_method'] = 1
        vparams['FRBdemo']['source_evolution'] = 0
        vparams['FRBdemo']['sfr_n'] = 0.73
        #vparams['FRBdemo']['lC'] = 3.15 # incorrect, check
        
        vparams['host']['lmean'] = 2.18
        vparams['host']['lsigma'] = 0.48
        
        vparams['MW']['DMhalo']=50
        vparams['MW']['halo_method']=0
        vparams['MW']['sigmaHalo']=0
        vparams['MW']['sigmaDMG']=0
        
        vparams['IGM']['logF'] = -0.49
        
        vparams['cosmo']['H0'] = 67.66
        
    elif case == "HoffmannEmin25":
        # this case is for H0 fixed to a small range,
        # and including P(N)
        vparams['energy']['lEmin'] = 39.47
        vparams['energy']['lEmax'] = 41.37
        vparams['energy']['alpha'] = -0.11
        vparams['energy']['gamma'] = -1.04
        vparams['energy']['luminosity_function'] = 2
        
        vparams['FRBdemo']['alpha_method'] = 1
        vparams['FRBdemo']['source_evolution'] = 0
        vparams['FRBdemo']['sfr_n'] = 0.21
        #vparams['FRBdemo']['lC'] = 3.15 # incorrect, check
        
        vparams['host']['lmean'] = 2.18
        vparams['host']['lsigma'] = 0.42
        
        vparams['MW']['DMhalo']=50
        vparams['MW']['halo_method']=0
        vparams['MW']['sigmaHalo']=0
        vparams['MW']['sigmaDMG']=0
        
        vparams['IGM']['logF'] = np.log10(0.32)
        
        vparams['cosmo']['H0'] = 70.23
        
        
        vparams['energy']['luminosity_function'] = 2
    elif case == "HoffmannHalo25":
        # this case is for H0 fixed to a small range,
        # and including P(N)
        vparams['energy']['lEmin'] = 38.22
        vparams['energy']['lEmax'] = 40.9
        vparams['energy']['alpha'] = 1.55
        vparams['energy']['gamma'] = -1.12
        vparams['energy']['luminosity_function'] = 2
        
        vparams['FRBdemo']['alpha_method'] = 1
        vparams['FRBdemo']['source_evolution'] = 0
        vparams['FRBdemo']['sfr_n'] = 2.88
        #vparams['FRBdemo']['lC'] = 3.15 # incorrect, check
        
        vparams['host']['lmean'] = 2.13
        vparams['host']['lsigma'] = 0.46
        
        vparams['MW']['DMhalo']=68
        vparams['MW']['halo_method']=0
        vparams['MW']['sigmaHalo']=15
        vparams['MW']['sigmaDMG']=0
        
        vparams['IGM']['logF'] = np.log10(0.32)
        
        vparams['cosmo']['H0'] = 70.63
        
        
        vparams['energy']['luminosity_function'] = 2
    else:
        raise ValueError("Unrecognised case. Please select one of ",cases)
    
    return vparams

def set_orig_scat(vparams):
    """
    Set width/scattering method to original version from zDM
    This introduces no scattering, and treats both width and
    scattering as a single combined distribution.
    Note that currently width method is given in the surveys.
    This will be altered.
    
    
    Args:
        vparams: existing dict containing state variables
    
    Returns:
        vparams: updated dict with width and scattering methods added
    """
    
    
    
    if not 'width' in vparams:
        vparams['width'] = {}
    vparams['width']['Wlogmean'] = 0.74
    vparams['width']['Wlogsigma'] = 1.07 # expressed as 2.46 in natural log space
    vparams['width']['WNbins'] = 5
    vparams['width']['WidthFunction'] = 1 # lognormal
    vparams['width']['Wthresh'] = 0.5
    # approximately the same width treatment - 
    # the functionality is changed, so it's not *quite*
    # identical
    vparams['width']['WMin'] = 0.1
    vparams['width']['WMax'] = 100
    
    #scattering was not actually included, but this gets set to zero
    # to ensure it doesn't contribute by accident
    if not 'scat' in vparams:
        vparams['width'] = {}
    vparams['scat'] = {}
    vparams['scat']['Slogmean'] = -2
    vparams['scat']['Slogsigma'] = 0.2
    vparams['scat']['ScatFunction'] = 1 # lognormal
    vparams['scat']['Sfnorm'] = 600
    vparams['scat']['Sfpower'] = 0.
    
    return vparams
    

def set_chime_scat(vparams):
    """
    Sets the width and scattering variables to those from CHIME,
    which were used in zDM between James22H0 and HoffmannEmin25
    
    Args:
        vparams: existing dict containing state variables
    
    Returns:
        vparams: updated dict with width and scattering methods added
    """
    
    if not 'width' in vparams:
        vparams['width'] = {}
    vparams['width']['Wlogmean'] = 0
    vparams['width']['Wlogsigma'] = 0.42
    vparams['width']['WNbins'] = 5
    vparams['width']['WidthFunction'] = 1 # lognormal
    vparams['width']['Wthresh'] = 0.5
    # approximately the same width treatment - 
    # the functionality is changed, so it's not *quite*
    # identical
    vparams['width']['WMin'] = 0.1
    vparams['width']['WMax'] = 100
    vparams['width']['WNInternalBins'] = 100
    
    
    if not 'scat' in vparams:
        vparams['width'] = {}
    vparams['scat'] = {}
    vparams['scat']['Slogmean'] = 0.305
    vparams['scat']['Slogsigma'] = 0.75
    vparams['scat']['ScatFunction'] = 1 # lognormal
    vparams['scat']['Sfnorm'] = 600
    vparams['scat']['Sfpower'] = -4.
    vparams['scat']['Smaxsigma'] = 3
    
    return vparams
    
def set_updated_scat(vparams):
    """
    Sets the width and scattering variables to those from
    James et al 2025, which focusses on width and scattering
    """
    
    if not 'width' in vparams:
        vparams['width'] = {}
    vparams['width']['Wlogmean'] = -0.29
    vparams['width']['Wlogsigma'] = 0.65
    vparams['width']['WNbins'] = 12
    vparams['width']['WidthFunction'] = 2 # half-lognormal
    vparams['width']['Wthresh'] = 0.5
    # approximately the same width treatment - 
    # the functionality is changed, so it's not *quite*
    # identical
    vparams['width']['WMin'] = 0.01
    vparams['width']['WMax'] = 100
    vparams['width']['WNInternalBins'] = 1000
    
    
    if not 'scat' in vparams:
        vparams['width'] = {}
    vparams['scat'] = {}
    vparams['scat']['Slogmean'] = -1.3
    vparams['scat']['Slogsigma'] = 0.2
    vparams['scat']['ScatFunction'] = 2 # half-lognormal
    vparams['scat']['Sfnorm'] = 1000
    vparams['scat']['Sfpower'] = -4.
    vparams['scat']['Smaxsigma'] = 3
    vparams['scat']['Sbackproject'] = False
    
    return vparams
