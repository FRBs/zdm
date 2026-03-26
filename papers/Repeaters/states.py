"""
This file contains defined critical values of states,
i.e. sets of parameters that define the population of
FRBs, and functions related to loading them.

The values are based on the best-fit results of
James et al (H0 paper), the 90% uncertainties
therein, and the best fits of Shin et al.


"""

import numpy as np
from zdm.MC_sample import loading
Planck_H0=67.4

def shin_fit():
    """
    Returns best-fit parameters from Shin et al.
    https://arxiv.org/pdf/2207.14316.pdf
    
    """
    
    pset={}
    pset["lEmax"] = np.log10(2.38)+41.
    pset["alpha"] = -1.39
    pset["gamma"] = -1.3
    pset["sfr_n"] = 0.96
    pset["lmean"] = 1.93
    pset["lsigma"] = 0.41
    pset["lC"] = np.log10(7.3)+4.
    
    return pset

def james_fit():
    """
    Returns best-fit parameters from James et al 2022 (Hubble paper)
    """
    
    pset={}
    pset["lEmax"] = 41.63
    pset["alpha"] = -1.03
    pset["gamma"] = -0.948
    pset["sfr_n"] = 1.15
    pset["lmean"] = 2.22
    pset["lsigma"] = 0.57
    pset["lC"] = 1.443
    
    return pset



def read_extremes(infile='planck_extremes.dat',H0=Planck_H0):
    """
    reads in extremes of parameters from a get_extremes_from_cube
    """
    f = open(infile)
    
    sets=[]
    
    for pset in np.arange(6):
        # reads the 'getting' line
        line=f.readline()
        
        pdict={}
        # gets parameter values
        for i in np.arange(6):
            line=f.readline()
            words=line.split()
            param=words[0]
            val=float(words[1])
            pdict[param]=val
        pdict["H0"]=H0
        pdict["alpha"] = -pdict["alpha"] # alpha is reversed!
        sets.append(pdict)
        
        pdict={}
        # gets parameter values
        for i in np.arange(6):
            line=f.readline()
            words=line.split()
            param=words[0]
            val=float(words[1])
            pdict[param]=val
        pdict["H0"]=H0
        pdict["alpha"] = -pdict["alpha"] # alpha is reversed!
        sets.append(pdict)
    return sets

def set_state(pset,chime_response=True,ischime=True):
    """
    Sets the state parameters
    
    ischime:
        if True, indicates we are setting for CHIME
        if False, indicates we are setting for other surveys
    """
    
    state = loading.set_state(alpha_method=1)
    state_dict = dict(cosmo=dict(fix_Omega_b_h2=True))
    state.energy.luminosity_function = 2 # this is Schechter
    state.update_param_dict(state_dict)
    
    # changes the beam method to be the "exact" one, otherwise sets up for FRBs
    #originally, this was a state function, now in the survey
    #if ischime:
    #    state.beam.Bmethod=3
    
    
    # updates to most recent best-fit values
    state.cosmo.H0 = 67.4
    
    if chime_response:
        state.width.Wmethod=0 #only a single width bin
        state.width.Wbias="CHIME"
    
    state.energy.lEmax = pset['lEmax']
    state.energy.gamma = pset['gamma']
    state.energy.alpha = pset['alpha']
    state.FRBdemo.sfr_n = pset['sfr_n']
    state.host.lsigma = pset['lsigma']
    state.host.lmean = pset['lmean']
    # state.FRBdemo.lC = pset['lC']
    
    return state



def get_states(ischime=True):  
    """
    Gets the states corresponding to plausible fits to single CHIME data
    
    Originally, about five sets were investigated - but only a couple
    were plausible within the CHIME response
    
    Setting nochime=False tells the program to never use the CHIME response
    function, i.e. only do this when modelling something that is not CHIME!
    """
    psets=read_extremes()
    psets.insert(0,shin_fit())
    psets.insert(1,james_fit())
    
    
    # gets list of psets compatible (ish) with CHIME
    chime_psets=[4]
    chime_names = ["CHIME min $\\alpha$"]
    
    # list of psets compatible (ish) with zdm
    zdm_psets = [1,2,7,12]
    zdm_names = ["zDM best fit","zDM min $\\E_{\\rm max}$","zDM max $\\gamma$","zDM min $\sigma_{\\rm host}$"]
    
    names=[]
    # loop over chime-compatible state
    for i,ipset in enumerate(chime_psets):
        
        if ischime:
            state=set_state(psets[ipset],chime_response=True,ischime=True)
        else:
            state=set_state(psets[ipset],chime_response=False,ischime=False)
        if i==0:
            states=[state]
        else:
            states.append(states)
        names.append(chime_names[i])
    
    for i,ipset in enumerate(zdm_psets):
        state=set_state(psets[ipset],chime_response=False)
        states.append(state)
        names.append(zdm_names[i])
    
    return states,names
