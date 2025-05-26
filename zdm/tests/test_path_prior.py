"""
Script testing the use of zDM to generate priors for
host galaxy magnitudes.

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

def test_path_priors():
    """
    Loops over all ICS FRBs
    """
    
    ######### List of all ICS FRBs for which we can run PATH #######
    # Just a single ICS FRB for this test
    frblist=['FRB20180924B']
    
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
    
    
    for frb in frblist:
        # interates over the FRBs. "Do FRB"
        # P_O is the prior for each galaxy
        # P_Ox is the posterior
        # P_Ux is the posterior for it being unobserved
        # mags is the list of galaxy magnitudes
        
        # determines if this FRB was seen by the survey, and
        # if so, what its DMEG is
        imatch = opt.matchFRB(frb,s)
        if imatch is None:
            raise ValueError("Could not find ",frb," in survey")
            # should be in this file
        
        DMEG = s.DMEGs[imatch]
        
        prior = model.init_path_raw_prior_Oi(DMEG,g)
        PU = model.estimate_unseen_prior(mag_limit=26) # might not be correct
        
        # the model should have calculated a valid unseen probability
        if PU < 0. or PU > 1.:
            raise ValueError("Unseen probability calculated to be ",PU)
        
        if not np.isfinite(PU):
            raise ValueError("Unseen probability PU is ",PU)
        
        bad = np.where(prior < 0.)[0]
        if len(bad) > 0:
            raise ValueError("Some elements of model prior have negative probability")
        
        OK = np.all(np.isfinite(prior))
        if not OK:
            raise ValueError("Some elements of magnitude priors are not finite")

test_path_priors()
