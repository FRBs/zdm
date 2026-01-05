"""
Contains files related to numerical optimisation
of FRB host galaxy parameters. Similar to iteration.py
for the grid.
"""

import numpy as np
from zdm import optical as op
from matplotlib import pyplot as plt

def function(x,args):
    """
    This is a function for input into the scipi.optimize.minimise routine.
    
    It calculates a set of PATH priors for that model, and then calculates
    a test statistic for that set.
    
    Args:
        frblist: list of TNS FRB names
        ss: list of surveys in which the FRB may exist
        gs: list of grids corresponding to those surveys
        model: optical model class which takes arguments x to be minimised. i.e.
            the function call model.AbsPrior = x must fully specify the model.
    
    
    """
    
    frblist = args[0]
    ss = args[1]
    gs=args[2]
    model=args[3]
    
    # initialises model to the priors
    # generates one per grid, due to possible different zvals
    
    model.init_args(x)
    wrappers = make_wrappers(model,gs)
    
    NFRB,AppMags,AppMagPriors,ObsMags,ObsPosteriors,PUprior,PUobs,sumPUprior,sumPUobs = calc_path_priors(frblist,ss,gs,wrappers,verbose=False)
    
    # we re-normalise the sum of PUs by NFRB
    
    # prevents infinite plots being created
    stat = calculate_goodness_statistic(NFRB,AppMags,AppMagPriors,ObsMags,ObsPosteriors,sumPUobs,sumPUprior,plotfile=None)
    
    return stat

def make_wrappers(model,grids):
    """
    returns a list of model wrapper objects for given model and grids
    
    Args:
        model: one of the optical model class objects
        grids: list of grid class objects
    
    Returns:
        wrappers: list of wrappers around model, one for each grid
    """
    wrappers = []
    for i,g in enumerate(grids):
        wrappers.append(op.model_wrapper(model,g.zvals))
    return wrappers
    

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
    

def calc_path_priors(frblist,ss,gs,wrappers,verbose=True,usemodel=True):
    """
    Inner loop. Gets passed model parameters, but assumes everything is
    initialsied from there.
    
    Inputs:
        FRBLIST: list of FRBs to retrieve data for
        ss: list of surveys modelling those FRBs (searches for FRB in data)
        gs: list of zDM grids modelling those surveys
        wrappers: list of optical wrapper class objects used to calculate priors on magnitude
        verbose (bool): Set to true to generate further output
    
    Returns:
        Number of FRBs fitted
        AppMags: list of apparent magnitudes used internally in the model
        allMagPriors: summed array of magnitude priors calculated by the model
        allObsMags: list of observed magnitudes of candidate hosts
        allPOx: list of posterior probabilities calculated by the model
        allPU: summed values of unobserved prior
        allPUx: summed values of posterior of being unobserved
    """
    
    NFRB = len(frblist)
    
    allObsMags = None
    allPOx = None
    allpriors = None
    sumPU = 0.
    sumPUx = 0.
    allPU = []
    allPUx = []
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
            imatch = op.matchFRB(frb,s)
            if imatch is not None:
                # this is the survey to be used
                g=gs[j]
                s = ss[j]
                wrapper = wrappers[j]
                break
        
        if imatch is None:
            if verbose:
                print("Could not find ",frb," in any survey")
            continue
        
        nfitted += 1
        
        AppMags = wrapper.AppMags
        
        DMEG = s.DMEGs[imatch]
        # this is where the particular survey comes into it
        
        # Must be priors on magnitudes for this FRB
        wrapper.init_path_raw_prior_Oi(DMEG,g)
        
        # extracts priors as function of absolute magnitude for this grid and DMEG
        MagPriors = wrapper.priors
        
        mag_limit=26  # might not be correct. TODO! Should be in FRB object
        
        # calculates unseen prior
        if usemodel:
            PU = wrapper.estimate_unseen_prior(mag_limit)
        else:
            PU = 0.1
            MagPriors[:] = 1./len(MagPriors) # log-uniform priors when no model used
        
        # sets magnitude priors to zero when they are above the magnitude limit
        bad = np.where(AppMags > mag_limit)[0]
        MagPriors[bad] = 0.
        
        P_O,P_Ox,P_Ux,ObsMags = op.run_path(frb,usemodel=usemodel,PU = PU)
        
        ObsMags = np.array(ObsMags)
        
        if allObsMags is None:
            allObsMags = ObsMags
            allPOx = P_Ox
            allMagPriors = MagPriors
        else:
            allObsMags = np.append(allObsMags,ObsMags)
            allPOx = np.append(allPOx,P_Ox)
            allMagPriors += MagPriors
        
        sumPU += PU
        sumPUx += P_Ux
        allPU.append(PU)
        allPUx.append(P_Ux)
    return nfitted,AppMags,allMagPriors,allObsMags,allPOx,allPU,allPUx,sumPU,sumPUx


def calculate_goodness_statistic(NFRB,AppMags,AppMagPriors,ObsMags,ObsPosteriors,PUobs,PUprior,plotfile=None):
    """
    Calculates a ks-like statistics to be proxy for goodness-of-fit
    We must set each AppMagPriors to 1.-PUprior at the limiting magnitude for each observation,
    and sum the ObsPosteriors to be equal to 1.-PUobs at that magnitude.
    Then these are what gets summed.
    
    This can be readily done by combining all ObsMags and ObsPosteriors into a single long list,
    since this should already be correctly normalised. Priors require their own weight.
    
    Inputs:
        AppMags: array listing apparent magnitudes
        AppMagPrior: array giving prior on AppMags
        ObsMags: list of observed magnitudes
        ObsPosteriors: list of posterior values corresponding to ObsMags
        PUobs: posterior on unseen probability
        PUprior: prior on PU
        Plotfile: set to name of output file for comparison plot
    
    Returns:
        k-like statistic of biggest obs/prior difference
    """
    
    # we calculate a probability using a cumulative distribution
    prior_dist = np.cumsum(AppMagPriors)
    
    # the above is normalised to NFRB. We now divide it by this
    # might want to be careful here, and preserve this normalisation
    prior_dist /= NFRB #((NFRB-PUprior)/NFRB) / prior_dist[-1]
    
    
    obs_dist = make_cdf(AppMags,ObsMags,ObsPosteriors,norm=False)
    
    obs_dist /= NFRB
    
    # we calculate something like the k-statistic. Includes NFRB normalisation
    diff = obs_dist - prior_dist
    stat = np.max(np.abs(diff))
    
    if plotfile is not None:
        plt.figure()
        plt.xlabel("Apparent magnitude $m_r$")
        plt.ylabel("Cumulative host galaxy distribution")
        #cx,cy = make_cdf_for_plotting(ObsMags,weights=ObsPosteriors)
        plt.plot(AppMags,obs_dist,label="Observed")
        plt.plot(AppMags,prior_dist,label="Prior")
        plt.legend()
        plt.tight_layout()
        plt.savefig(plotfile)
        plt.close()
        
    
    return stat
