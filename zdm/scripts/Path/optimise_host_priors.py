"""
This file illustrates how to optimise the host prior
distribution using an MCMC approach.

WARNING: this is NOT the correct method! That would require using
a catalogue of galaxies to sample from to generate fake opotical fields.
But nonetheless, this tests the power of estimating FRB host galaxy
contributions using zDM to set priors for 

WARNING2: To do this properly also requires inputting the posterior POx
for host galaxies into zDM! This simulation does not do that either. Yet...

(this file is under development)
"""


#standard Python imports
import numpy as np
from matplotlib import pyplot as plt

# imports from the "FRB" series
from zdm import optical as opt
from zdm import optical_params as op
from zdm import loading
from zdm import cosmology as cos
from zdm import parameters
from zdm import loading

import utilities as ute
import astropath.priors as pathpriors



def main():
    """
    Main function
    Contains outer loop to iterate over parameters
    
    """
    
    ######### List of all ICS FRBs for which we can run PATH #######
    # hard-coded list of FRBs with PATH data in ice paper
    frblist=['FRB20180924B','FRB20181112A','FRB20190102C','FRB20190608B',
        'FRB20190611B','FRB20190711A','FRB20190714A','FRB20191001A',
        'FRB20191228A','FRB20200430A','FRB20200906A','FRB20210117A',
        'FRB20210320C','FRB20210807D','FRB20211127I','FRB20211203C',
        'FRB20211212A','FRB20220105A','FRB20220501C',
        'FRB20220610A','FRB20220725A','FRB20220918A',
        'FRB20221106A','FRB20230526A','FRB20230708A', 
        'FRB20230731A','FRB20230902A','FRB20231226A','FRB20240201A',
        'FRB20240210A','FRB20240304A','FRB20240310A']
    
    # Initlisation of zDM grid
    # Eventually, this should be part of the loop, i.e. host IDs should
    # be re-fed into FRB surveys. However, it will be difficult to do this
    # with very limited redshift estimates. That might require posterior
    # estimates of redshift given the observed galaxies. Maybe.
    state = parameters.State()
    cos.set_cosmology(state)
    cos.init_dist_measures()
    names=['CRAFT_ICS_892','CRAFT_ICS_1300','CRAFT_ICS_1632']
    ss,gs = loading.surveys_and_grids(survey_names=names)
    
    # Initialisation of model
    opt_params = op.Hosts()
    model = opt.host_model(opstate = opt_params)
    model.init_zmapping(gs[0].zvals)
    
    NFRB,AppMags,AppMagPriors,ObsMags,ObsPosteriors,PUobs,PUprior = calc_path_priors(frblist,ss,gs,model)
    
    # we re-normalise the sum of PUs by NFRB
    calculate_goodness_statistic(AppMags,AppMagPriors,ObsMags,ObsPosteriors,PUobs/NFRB,PUprior/NFRB)

def make_cdf(xs,ys,ws,norm = True):
    """
    makes a cumulative distribution in terms of
    the x-values x, observed values y, and weights w
    
    """
    cdf = np.zeros([xs.size])
    for i,y in enumerate(ys):
        OK = np.where(xs > y)[0]
        cdf[OK] += ws[i]
    if norm:
        cdf /= cdf[-1]
    return cdf

def calculate_goodness_statistic(AppMags,AppMagPriors,ObsMags,ObsPosteriors,PUobs,PUprior):
    """
    Compares magnitude posteriors to model priors, and updates them
    
    Inputs:
        AppMags: array listing apparent magnitudes
        AppMagPrior: array giving prior on AppMags
        ObsMags: list of observed magnitudes
        ObsPosteriors: list of posterior values corresponding to ObsMags
        PUobs: posterior on unseen probability
        PUprior: prior on PU
    
    Returns:
        k-like statistic of biggest obs/prior difference
    """
    # we calculate a probability using a cumulative distribution
    prior_dist = np.cumsum(AppMagPriors)
    # might want to be careful here, and preserve this normalisation
    prior_dist *= (1.-PUprior)/prior_dist[-1]
    
    obs_dist = make_cdf(AppMags,ObsMags,ObsPosteriors,norm=True)
    obs_dist *= PUprior
    
    # we calculate something like the k-statistic
    diff = obs_dist - prior_dist
    stat = np.max(np.abs(diff))
    return stat
    
def calc_path_priors(frblist,ss,gs,model):
    """
    Inner loop. Gets passed model parameters, but assumes everything is
    initialsied from there.
    """
    
    NFRB = len(frblist)
    
    
    
    # we assume here that the model has just had a bunch of parametrs updated
    # within it. Must be done once for any fixed zvals. If zvals change,
    # then we have another issue
    model.reinit_model()
    
    # do this once per "model" objects
    pathpriors.USR_raw_prior_Oi = model.path_raw_prior_Oi
    
    allmags = None
    allPOx = None
    allpriors = None
    appmags = model.AppMags
    allPU = 0.
    allPUx = 0.
    nfitted = 0
    
    for i,frb in enumerate(frblist):
        # interates over the FRBs. "Do FRB"
        # P_O is the prior for each galaxy
        # P_Ox is the posterior
        # P_Ux is the posterior for it being unobserved
        # mags is the list of galaxy magnitudes
        
        # determines if this FRB was seen by the survey, and
        # if so, what its DMEG is
        for j,s in enumerate(ss):
            imatch = ute.matchFRB(frb,s)
            if imatch is not None:
                # this is the survey to be used
                g=gs[j]
                break
        
        if imatch is None:
            print("Could not find ",frb," in any survey")
            continue
        
        nfitted += 1
        
        DMEG = s.DMEGs[imatch]
        # this is where the particular survey comes into it
        priors = model.init_path_raw_prior_Oi(DMEG,g)
        mag_limit=26
        PU = model.estimate_unseen_prior(mag_limit) # might not be correct
        bad = np.where(priors > mag_limit)
        priors[bad] = 0.
        P_O,P_Ox,P_Ux,mags = ute.do_frb(frb,model,usemodel=True,PU = PU)
        
        mags = np.array(mags)
        
        if allmags is None:
            allmags = mags
            allPOx = P_Ox
            allpriors = priors
        else:
            allmags = np.append(allmags,mags)
            allPOx = np.append(allPOx,P_Ox)
            allpriors += priors
        allPU += PU
        allPUx += P_Ux
            
    return nfitted,appmags,allpriors,allmags,allPOx,allPU,allPUx


main()
