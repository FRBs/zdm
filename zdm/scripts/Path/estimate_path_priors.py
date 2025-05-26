"""
Script showing how to use zDM as priors for CRAFT
host galaxy magnitudes.

It requirses the FRB and astropath modules to be installed.

This does NOT include optimisation of any parameters
"""

#standard Python imports
import numpy as np
from matplotlib import pyplot as plt

# imports from the "FRB" series
from zdm import optical as opt
from zdm import loading
from zdm import cosmology as cos
from zdm import parameters
from zdm import loading

import utilities as ute
import astropath.priors as pathpriors

import ics_frbs as ics

def calc_path_priors():
    """
    Loops over all ICS FRBs
    """
    
    frblist = ics.frblist
    
    NFRB = len(frblist)
    
    # here is where I should initialise a zDM grid
    state = parameters.State()
    cos.set_cosmology(state)
    cos.init_dist_measures()
    model = opt.host_model()
    name='CRAFT_ICS_1300'
    ss,gs = loading.surveys_and_grids(survey_names=[name])
    g = gs[0]
    s = ss[0]
    # must be done once for any fixed zvals
    model.init_zmapping(g.zvals)
    
    # do this only for a particular FRB
    # it gives a prior on apparent magnitude and pz
    #AppMagPriors,pz = model.get_posterior(g,DMlist)
    
    # do this once per "model" objects
    pathpriors.USR_raw_prior_Oi = model.path_raw_prior_Oi
    
    allmags = None
    allPOx = None
    
    for frb in frblist:
        # interates over the FRBs. "Do FRB"
        # P_O is the prior for each galaxy
        # P_Ox is the posterior
        # P_Ux is the posterior for it being unobserved
        # mags is the list of galaxy magnitudes
        
        # determines if this FRB was seen by the survey, and
        # if so, what its DMEG is
        imatch = ute.matchFRB(frb,s)
        if imatch is None:
            print("Could not find ",frb," in survey")
            continue
        
        DMEG = s.DMEGs[imatch]
        
        # original calculation
        P_O1,P_Ox1,P_Ux1,mags1 = ute.run_path(frb,model,usemodel=False,PU=0.1)
        
        model.init_path_raw_prior_Oi(DMEG,g)
        PU = model.estimate_unseen_prior(mag_limit=26) # might not be correct
        P_O2,P_Ox2,P_Ux2,mags2 = ute.run_path(frb,model,usemodel=True,PU = PU)
        
        if False:
            # compares outcomes
            print("FRB ",frb)
            print(" m_r               P_O: old               new               P_Ox: old               new")
            for i,P_O in enumerate(P_O1):
                print(i,mags1[i],P_O1[i],P_O2[i],P_Ox1[i],P_Ox2[i])
            print("\n")
        
        # keep cumulative histogram of posterior magnitude distributions
        #allmags.append(mags2)
        #allPOx.append(P_Ox2)
        mags2 = np.array(mags2)
        
        if allmags is None:
            allmags = mags2
            allPOx = P_Ox2
        else:
            allmags = np.append(allmags,mags2)
            allPOx = np.append(allPOx,P_Ox2)
    
    Nbins = int(model.Appmax - model.Appmin)+1
    bins = np.linspace(model.Appmin,model.Appmax,Nbins)
    plt.figure()
    plt.hist(allmags,weights = allPOx, bins = bins,label="Posterior")
    plt.legend()
    plt.tight_layout()
    plt.savefig("posterior_pOx.png")
    plt.close()
    


 

if __name__ == "__main__":
    
    calc_path_priors()

    
    
