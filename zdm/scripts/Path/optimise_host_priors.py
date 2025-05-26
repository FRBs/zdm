"""
This file illustrates how to optimise the host prior
distribution by fitting to CRAFT ICS optical observations.
It fits a model of absolute galaxy magnitude distributions,
uses zDM to predict redshifts and hence apparent magntidues,
runs PATH using that prior, and tries to get priors to match posteriors.

WARNING: this is NOT the optimal method! That would require using
a catalogue of galaxies to sample from to generate fake opotical fields.
But nonetheless, this tests the power of estimating FRB host galaxy
contributions using zDM to set priors for apparent magnitudes.

WARNING2: To do this properly also requires inputting the posterior POx
for host galaxies into zDM! This simulation does not do that either.

WARNING3: this program takes O~1 hr to run
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
from scipy.optimize import minimize

import ics_frbs as ics

def main():
    """
    Main function
    Contains outer loop to iterate over parameters
    
    """
    
    ######### List of all ICS FRBs for which we can run PATH #######
    # hard-coded list of FRBs with PATH data in ice paper
    frblist=ics.frblist
    
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
    opt_params.AbsModelID = 1
    model = opt.host_model(opstate = opt_params)
    model.init_zmapping(gs[0].zvals)
    
    x0 = model.AbsPrior
    
    args=[frblist,ss,gs,model]
    Nparams = len(x0)
    bounds = [(0,1)]*Nparams
    result = minimize(function,x0 = x0,args=args,bounds = bounds)
    
    # Recording the current spline best-fit here
    #x = [0.00000000e+00 0.00000000e+00 7.05155614e-02 8.39235326e-01
    #    3.27794398e-01 1.00182186e-03 0.00000000e+00 3.46702511e-04
    #    2.17040011e-03 9.72472750e-04]
    
    # recording the current non-spline best fit here
    #x = [ 1.707e-04,  8.649e-02,  9.365e-01,  9.996e-01,  2.255e-01,\
    #         3.493e-02,  0.000e+00,  0.000e+00,  0.000e+00,  1.000e-01]
    #x = np.array(x)
    
    print("Best fit result is ",result.x)
    x = result.x
    
    # analyses final result
    x /= np.sum(x)
    model.AbsPrior = x
    model.reinit_model()
    outfile = "best_fit_apparent_magnitudes.png"
    NFRB,AppMags,AppMagPriors,ObsMags,ObsPosteriors,PUprior,PUobs,sumPUprior,sumPUobs = calc_path_priors(frblist,ss,gs,model,verbose=False)
    stat = calculate_goodness_statistic(NFRB,AppMags,AppMagPriors,ObsMags,ObsPosteriors,sumPUobs,sumPUprior,plotfile=outfile)
    
    # calculates the original PATH result
    #outfile = "original_fit_apparent_magnitudes.png"
    NFRB2,AppMags2,AppMagPriors2,ObsMags2,ObsPosteriors2,PUprior2,PUobs2,sumPUprior2,sumPUobs2 = calc_path_priors(frblist,ss,gs,model,verbose=False,usemodel=False)
    #stat = calculate_goodness_statistic(NFRB,AppMags,AppMagPriors,ObsMags,ObsPosteriors,sumPUobs,sumPUprior,plotfile=outfile)
    
    
    # plots original vs updated posteriors
    plt.figure()
    plt.xlabel("Original P")
    plt.ylabel("Best fit P")
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.scatter(ObsPosteriors2,ObsPosteriors,label="Hosts",marker='x')
    plt.scatter(PUobs2,PUobs,label="Unobserved",marker='+')
    plt.legend()
    plt.tight_layout()
    plt.savefig("Scatter_plot_comparison.png")
    plt.close()
    
    
    
    # plots final result on absolute magnitudes
    plt.figure()
    plt.xlabel("Absolute magnitude, $M_r$")
    plt.ylabel("$p(M_r)$")
    plt.plot(model.AbsMags,model.AbsMagWeights/np.max(model.AbsMagWeights),label="interpolation")
    plt.plot(model.ModelBins,x/np.max(x),marker="o",linestyle="",label="Model Parameters")
    plt.legend()
    plt.tight_layout()
    plt.savefig("best_fit_absolute_magnitudes.pdf")
    plt.close()

def function(x,args):
    """
    function to be minimised
    """ 
    frblist = args[0]
    ss = args[1]
    gs=args[2]
    model=args[3]
    
    # initialises model to the priors
    # technically, there is a redundant normalisation here
    model.AbsPrior = x
    
    NFRB,AppMags,AppMagPriors,ObsMags,ObsPosteriors,PUprior,PUobs,sumPUprior,sumPUobs = calc_path_priors(frblist,ss,gs,model,verbose=False)
    
    # we re-normalise the sum of PUs by NFRB
    
    # prevents infinite plots being created
    stat = calculate_goodness_statistic(NFRB,AppMags,AppMagPriors,ObsMags,ObsPosteriors,sumPUobs,sumPUprior,plotfile=None)
    
    return stat
    
    
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

def make_cdf_for_plotting(xvals,weights=None):
    """
    Creates a cumulative distribution function
    
    xvals,yvals: values of data points
    """
    N = xvals.size
    cx = np.zeros([2*N])
    cy = np.zeros([2*N])
    
    order = np.argsort(xvals)
    xvals = xvals[order]
    if weights is None:
        weights = np.linspace(0.,1.,N+1)
    else:
        weights = weights[order]
        weights = np.cumsum(weights)
        weights /= weights[-1]
        
    for i,x in enumerate(xvals):
        cx[2*i] = x
        cx[2*i+1] = x
        cy[2*i] = weights[i]
        cy[2*i+1] = weights[i+1]
    return cx,cy

def calculate_goodness_statistic(NFRB,AppMags,AppMagPriors,ObsMags,ObsPosteriors,PUobs,PUprior,plotfile=None):
    """
    Calculates a ks-like statistics to be proxzy for goodness-of-fit
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
    
def calc_path_priors(frblist,ss,gs,model,verbose=True,usemodel=True):
    """
    Inner loop. Gets passed model parameters, but assumes everything is
    initialsied from there.
    
    Inputs:
        FRBLIST: list of FRBs to retrieve data for
        ss: list of surveys modelling those FRBs (searches for FRB in data)
        gs: list of zDM grids modelling those surveys
        model: host_model class object used to calculate priors on magnitude
        verbose (bool): guess
    
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
    
    # we assume here that the model has just had a bunch of parametrs updated
    # within it. Must be done once for any fixed zvals. If zvals change,
    # then we have another issue
    model.reinit_model()
    
    # do this once per "model" objects
    pathpriors.USR_raw_prior_Oi = model.path_raw_prior_Oi
    
    allObsMags = None
    allPOx = None
    allpriors = None
    AppMags = model.AppMags
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
            imatch = ute.matchFRB(frb,s)
            if imatch is not None:
                # this is the survey to be used
                g=gs[j]
                break
        
        if imatch is None:
            if verbose:
                print("Could not find ",frb," in any survey")
            continue
        
        nfitted += 1
        
        DMEG = s.DMEGs[imatch]
        # this is where the particular survey comes into it
        MagPriors = model.init_path_raw_prior_Oi(DMEG,g)
        mag_limit=26  # might not be correct
        PU = model.estimate_unseen_prior(mag_limit)
        bad = np.where(AppMags > mag_limit)[0]
        MagPriors[bad] = 0.
        
        P_O,P_Ox,P_Ux,ObsMags = ute.run_path(frb,model,usemodel=usemodel,PU = PU)
        
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


main()
