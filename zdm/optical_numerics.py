"""
Numerical routines for evaluating and optimising FRB host galaxy magnitude models.

This module is the numerical workhorse for the PATH integration in zdm,
analogous to ``iteration.py`` for the zdm grid. It provides:

- **``function``** objective function passed to ``scipy.optimize.minimize``
  that evaluates a goodness-of-fit statistic for a given set of host model
  parameters against the CRAFT ICS optical data.

- **``calc_path_priors``** inner loop that runs PATH on a list of FRBs
  across one or more surveys/grids, collecting priors, posteriors, and
  undetected-host probabilities for each FRB.

- **``run_path``** runs the PATH algorithm for a single named FRB,
  loading its candidate host galaxies from the ``frb`` package data
  and applying colour corrections to convert to r-band.

- **``calculate_likelihood_statistic``** and **``calculate_ks_statistic``**
  — goodness-of-fit statistics comparing the model apparent magnitude prior
  to the observed PATH posteriors across all FRBs.

- **``make_cumulative_plots``** plotting routine for visualising
  cumulative magnitude distributions for one or more models simultaneously.

- **``make_wrappers``**, **``make_cdf``**, **``flatten``**,
  **``get_cand_properties``** supporting utilities.
"""

import os
from importlib import resources
import numpy as np
from matplotlib import pyplot as plt
import pandas

from zdm import optical as op

from frb.frb import FRB
from astropath.priors import load_std_priors
from astropath.path import PATH
from astropath import chance
from frb.associate import frbassociate
    
def function(x,args):
    """
    Objective function for ``scipy.optimize.minimize`` over host model parameters.

    Updates the host magnitude model with parameter vector ``x``, runs PATH
    on all FRBs, computes the chosen goodness-of-fit statistic, and returns
    a scalar value suitable for minimisation (i.e. smaller is better).

    Parameters
    ----------
    x : np.ndarray
        Parameter vector passed to ``model.init_args(x)``. Its meaning
        depends on the model (e.g. absolute magnitude bin weights for
        ``simple_host_model``, or ``fSFR`` for ``loudas_model``).
    args : list
        Packed argument tuple with the following elements, in order:

        - ``frblist`` (list of str): TNS names of FRBs to evaluate.
        - ``ss`` (list of Survey): surveys in which the FRBs may appear.
        - ``gs`` (list of Grid): zdm grids corresponding to those surveys.
        - ``model``: host magnitude model instance (must implement
          ``init_args``).
        - ``POxcut`` (float or None): if not None, restrict the statistic
          to FRBs whose best host candidate has P(O|x) > POxcut.
        - ``istat`` (int): statistic to use 0 for KS-like statistic,
          1 for maximum-likelihood (returned as negative log-likelihood
          so that minimisation maximises the likelihood).

    Returns
    -------
    stat : float
        Goodness-of-fit statistic (smaller is better). For ``istat=1``
        this is the negative log-likelihood.
    """
    
    frblist = args[0]
    ss = args[1]
    gs=args[2]
    model=args[3]
    POxcut=args[4] # either None, or a cut such as 0.9
    istat=args[5]
    
    # initialises model to the priors
    # generates one per grid, due to possible different zvals
    model.init_args(x)
    wrappers = make_wrappers(model,gs)
    
    
    results = calc_path_priors(frblist,ss,gs,wrappers,verbose=False)
    #NFRB,AppMags,AppMagPriors,ObsMags,ObsPriors,ObsPosteriors,PUprior,PUobs,sumPUprior,sumPUobs,frbs,dms = calc_path_priors(frblist,ss,gs,wrappers,verbose=False)
    
    # we re-normalise the sum of PUs by NFRB
    
    # prevents infinite plots being created
    if istat==0:
        stat = calculate_ks_statistic(results["NFRB"],results["AppMags"],results["AppMagPriors"],
                                    results["ObsMags"],results["POx"],
                                    results["sumPUx"],results["sumPU"],plotfile=None,POxcut=POxcut)
    elif istat==1:
        stat = calculate_likelihood_statistic(results["NFRB"],results["AppMags"],results["AppMagPriors"],
                                    results["ObsMags"],results["POx"],results["PUx"],results["PU"],
                                    plotfile=None,POxcut=POxcut)
        # need to construct stat so that small values are good! Log-likelihood being good means large!
        stat *= -1
    
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
    

def make_cdf(xs,ys,ws,norm=True):
    """
    Build a weighted empirical CDF evaluated on a fixed grid.

    For each grid point ``x`` in ``xs``, accumulates the weights ``ws[i]``
    of all observations ``ys[i]`` that fall below ``x``. The result is a
    non-decreasing array that can be compared to a model prior CDF.

    Parameters
    ----------
    xs : np.ndarray
        Grid of x values at which to evaluate the CDF (e.g. apparent
        magnitude bin centres). Must be sorted in ascending order.
    ys : array-like
        Observed data values (e.g. host galaxy apparent magnitudes).
    ws : array-like
        Weight for each observation in ``ys`` (e.g. PATH posteriors P_Ox).
        Must have the same length as ``ys``.
    norm : bool, optional
        If True (default), normalise the CDF so that its maximum value
        is 1. Set to False to preserve the raw cumulative weight sum.

    Returns
    -------
    cdf : np.ndarray, shape (len(xs),)
        Weighted empirical CDF evaluated at each point in ``xs``.
    """
    cdf = np.zeros([xs.size])
    for i,y in enumerate(ys):
        OK = np.where(xs > y)[0]
        cdf[OK] += ws[i]
    if norm:
        cdf /= cdf[-1]
    return cdf

    
def calc_path_priors(frblist,ss,gs,wrappers,verbose=True,usemodel=True,P_U=0.1,
                    failOK=False,doz=True,field=None,pzlist=None,scale=0.5):
    """
    Run PATH on a list of FRBs and return priors, posteriors, and P_U values.

    For each FRB in ``frblist``, searches all surveys in ``ss`` for a match,
    computes the zdm-derived apparent magnitude prior (if ``usemodel=True``),
    and runs PATH to produce host association posteriors. Results for all FRBs
    are collected into parallel lists (one entry per FRB).

    Also writes a CSV file ``allgalaxies.csv`` (if it does not already exist)
    containing the magnitude and VLT/FORS2 R-band columns for all candidate
    host galaxies across all FRBs.

    Parameters
    ----------
    frblist : list of str
        TNS names of FRBs to process (e.g. ``['FRB20180924B', ...]``).
    ss : list of Survey
        Survey objects to search for each FRB. The first survey containing
        a given FRB is used.
    gs : list of Grid
        zdm grids corresponding to each survey in ``ss``.
    wrappers : list of model_wrapper
        One ``model_wrapper`` per grid (from ``make_wrappers``), used to
        compute DM-dependent apparent magnitude priors.
    verbose : bool, optional
        If True, print a warning for each FRB not found in any survey.
        Defaults to True.
    usemodel : bool, optional
        If True, use the zdm-derived magnitude prior from ``wrappers`` and
        estimate P_U from the model. If False, use PATH's built-in inverse
        prior and the supplied fixed ``P_U``. Defaults to True.
    P_U : float, optional
        Fixed prior probability that the host galaxy is undetected. Only
        used when ``usemodel=False``. Defaults to 0.1.
    failOK : bool, optional
        If True, allows failed attempts to find an FRB - simply skips cases where
        no FRB data could be found
    doz : bool, optional
        If true, calculate p(z) for both field and host galaxies for these FRBs
    fiedd: optical.Field object, optional
        Option to pass existing field object for speedup purposes
    pzlist: list of np.ndarray, optional
        If given, must be [NFRB x g.zvals] in szie list giving p(z) for
        each FRB in the list
    
    Returns
    -------
    nfitted : int
        Number of FRBs successfully matched to a survey and processed.
    AppMags : np.ndarray
        Internal apparent magnitude grid (from the last processed wrapper).
    allMagPriors : list of np.ndarray
        One array per FRB giving p(m_r | DM_EG) on the ``AppMags`` grid.
        Entries are ``None`` when ``usemodel=False``.
    allObsMags : list of np.ndarray
        One array per FRB listing the r-band magnitudes of PATH candidate
        host galaxies.
    allPO : list of np.ndarray
        One array per FRB giving the PATH prior P_O for each candidate.
    allPOx : list of np.ndarray
        One array per FRB giving the PATH posterior P(O|x) for each candidate.
    allPU : list of float
        Prior P_U (probability of unseen host) for each FRB.
    allPUx : list of float
        Posterior P(U|x) (probability host is unseen, given data) for each FRB.
    sumPU : float
        Sum of ``allPU`` across all FRBs.
    sumPUx : float
        Sum of ``allPUx`` across all FRBs.
    frbs : list of str
        TNS names of the FRBs that were successfully matched and processed.
    dms : list of float
        Extragalactic DM (pc cm⁻³) for each FRB in ``frbs``.
    """
    
    NFRB = len(frblist)
    
    # old version creating 1D lists
    #allObsMags = None
    #allPOx = None
    #allMagPriors = None
    
    # new version recording one list per FRB. For max likelihood functionality
    allObsMags = []
    allPOx = []
    allPxO = []
    allPO = []
    allMagPriors = []
    
    sumPU = 0.
    sumPUx = 0.
    allPU = []
    allPUx = []
    allPm = []
    allrhom = []
    nfitted = 0
    
    frbs=[]
    dms=[]
    
    OKlist = []
    OKfrb = []
    
    allpz = []
    allpf = []
    
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
                if usemodel:
                    wrapper = wrappers[j]
                jmatch = j
                frbs.append(frb)
                break
        
            if imatch is None:
                if verbose:
                    print("Could not find ",frb," in any survey")
                continue
        
        nfitted += 1
        
        if usemodel:
            AppMags = wrapper.AppMags
        else:
            AppMags = None
        
        # record this info  
        DMEG = s.DMEGs[imatch]
        dms.append(DMEG)
        
        if usemodel:
            
            # this is where the particular survey comes into it
            # Must be priors on magnitudes for this FRB
            if pzlist is not None:
                wrapper.init_path_raw_prior_Oi(DMEG,pz=pzlist[i])
            else:
                wrapper.init_path_raw_prior_Oi(DMEG,grid=g)
        
            # extracts priors as function of absolute magnitude for this grid and DMEG
            MagPriors = wrapper.priors
        else:
            MagPriors = None
        
        # calculates unseen prior
        if usemodel:
            P_U = wrapper.estimate_unseen_prior()
        
        result = run_path(frb,usemodel=usemodel,P_U = P_U, failOK = failOK, scale=scale)
        if result["Error"]:
            if failOK:
                continue
            else:
                print("run_path failed unexpectedly, quitting...")
                exit()
        
        OKlist.append(i)
        OKfrb.append(frb)
        
        if doz:
            # calculate p(z) for model galaxies for result["mags"]
            # we assume it's the most likely galaxy that has a redshift.
            # this functionality NEEDS to change!
            if field is None:
                field = op.Field()
            
            if s.Zs[imatch] > 0.:
                pz = wrapper.get_pz_g_mr(result["mags"][0],s.Zs[imatch])
                pf = field.get_pzgm(result["mags"][0],s.Zs[imatch])
            else:
                pz=1. # simply unity, i.e. meaningless
                pf=1.
            allpz.append(pz)
            allpf.append(pf)
        
        # adds m and driver rho values to results
        if usemodel:
            result["Pm"] = wrapper.path_base_prior(result["mags"])
            result["rhom"] = chance.differential_driver_sigma(result["mags"])
            
        if i==0:
            allgals = result["ptbl"]
        else:
            allgals = pandas.concat([allgals,result["ptbl"]], ignore_index=True)
        
        ObsMags = np.array(result["mags"])
        
        # new version creating a list of lists
        allObsMags.append(ObsMags)
        allPOx.append(result["POx"])
        allPxO.append(result["PxO"])
        allPO.append(result["PO"])
        allMagPriors.append(MagPriors)
        if usemodel:
            allPm.append(result["Pm"])
            allrhom.append(result["rhom"])
        
        sumPU += P_U
        sumPUx += result["PUx"]
        allPU.append(P_U)
        allPUx.append(result["PUx"])
    
    # once-off code for exporting
    #subset = allgals[['frb','mag','VLT_FORS2_R']].copy()
    # saves all galaxies
    #if not os.path.exists("allgalaxies.csv"):
    #    subset.to_csv("allgalaxies.csv",index=False)
    
    
    # creates a dict to hold results
    results = {}
    results["NFRB"] = nfitted
    results["AppMags"] = AppMags
    results["AppMagPriors"] = allMagPriors
    results["ObsMags"] = allObsMags
    results["PO"] = allPO
    results["POx"] = allPOx
    results["PxO"] = allPxO
    results["PU"] = allPU
    results["PUx"] = allPUx
    results["sumPU"] = sumPU
    results["sumPUx"] = sumPUx
    results["frbs"] = frbs
    results["dms"] = dms
    results["OK"] = OKlist
    results["pz"] = allpz
    results["pf"] = allpf
    results["OKlist"] = OKlist
    results["frblist"] = OKfrb
    results["Pm"] = allPm
    results["rhom"] = allrhom
    
    return results


def calculate_likelihood_statistic(NFRB,AppMags,AppMagPriors,ObsMags,ObsPosteriors,PUobs,
                                PUprior,plotfile=None,POxcut=None):
    """
    Compute the total log-likelihood of the observed PATH posteriors given the model prior.

    For each FRB, evaluates log10(Σ P(O_i|x) / rescale + P_U_prior), where the
    rescale factor accounts for PATH's internal renormalisation of posteriors
    relative to the model prior. Summing over all FRBs gives the total
    log-likelihood returned to the caller.

    Parameters
    ----------
    NFRB : int
        Number of FRBs to sum over.
    AppMags : np.ndarray, shape (NMAG,)
        Apparent magnitude grid used to compute the model prior (not used
        directly in this function, but kept for API consistency with
        ``calculate_ks_statistic``).
    AppMagPriors : list of np.ndarray, length NFRB
        Model prior p(m_r | DM_EG) on the ``AppMags`` grid, one array per FRB.
    ObsMags : list of np.ndarray, length NFRB
        Observed r-band magnitudes of PATH candidate host galaxies, one array
        per FRB (length NCAND varies by FRB).
    ObsPosteriors : list of np.ndarray, length NFRB
        PATH posterior P(O_i|x) for each candidate, one array per FRB.
    PUobs : list of float, length NFRB
        PATH posterior P(U|x) — probability that the true host is undetected —
        for each FRB, as returned by PATH after renormalisation.
    PUprior : list of float, length NFRB
        Model prior P_U for each FRB, as estimated by
        ``wrapper.estimate_unseen_prior()``.
    plotfile : str or None, optional
        If provided, save a diagnostic plot comparing prior and posterior
        magnitude distributions to this file path. Defaults to None.
    POxcut : float or None, optional
        If not None, restrict the statistic to FRBs whose maximum P(O|x)
        exceeds this threshold (simulates requiring a confident host ID).
        Defaults to None.

    Returns
    -------
    stat : float
        Total log10-likelihood summed over all NFRB FRBs. Larger values
        indicate a better fit. Multiply by -1 for use as a minimisation
        objective.
    """
    # calculates log-likelihood of observation
    stat=0
    
    for i in np.arange(NFRB):
        # sums the likelihoods over each galaxy: p(xi|oi)*p(oi)/Pfield
        
        # calculate the factor by which the p...|x probabilities have been rescaled.
        # allows us to undo this effect
        rescale = PUobs[i]/PUprior[i]
        # the problem is that the posteriors have been rescaled by some factor
        # we do not want this! Hence, we work out the rescale factor by comparing
        # the rescale on the unseen prior. Then we undo this factor
        # (Note: PUobs / rescale = PUprior, hence must divide)
        sumpost = np.sum(ObsPosteriors[i])/rescale+PUprior[i]
        
        if False:
            plt.figure()
            plt.plot(AppMags,AppMagPriors[i]/np.max(AppMagPriors[i]),label="priors from model")
            for j,mag in enumerate(ObsMags[i]):
                plt.scatter(ObsMags[i],ObsPosteriors[i],label="posteriors")
           
            print("Sum gives ",sumpost, " of which PU is ",PUprior[i])
            plt.show()
            plt.close()
        ll = np.log10(sumpost)
        stat += ll
    
        
    return stat

def flatten(xss):
    """
    Flatten a list of lists into a single flat list.
    """
    return [x for xs in xss for x in xs]

def calculate_ks_statistic(NFRB,AppMags,AppMagPriors,ObsMags,ObsPosteriors,PUobs,
                                PUprior,plotfile=None,POxcut=None,plotlabel=None,abc=None,tag=""):
    """
    Compute a KS-like goodness-of-fit statistic between model prior and observed posteriors.

    Builds cumulative magnitude distributions for both the model prior and the
    PATH posteriors, normalised by the number of FRBs, and returns the maximum
    absolute difference between them — analogous to the KS test statistic.

    Optionally produces a plot comparing the two cumulative distributions.

    Parameters
    ----------
    NFRB : int
        Number of FRBs used for normalisation.
    AppMags : np.ndarray, shape (NMAG,)
        Apparent magnitude grid on which priors are defined.
    AppMagPriors : list of np.ndarray, length NFRB
        Model prior p(m_r | DM_EG) on ``AppMags``, one array per FRB.
    ObsMags : list of np.ndarray, length NFRB
        Observed r-band magnitudes of PATH candidate galaxies, one per FRB.
    ObsPosteriors : list of np.ndarray, length NFRB
        PATH posteriors P(O_i|x) for each candidate, one array per FRB.
    PUobs : list of float
        Posterior P(U|x) for each FRB (not used directly in the statistic,
        kept for API consistency).
    PUprior : list of float
        Prior P_U for each FRB (not used directly, kept for API consistency).
    plotfile : str or None, optional
        If provided, save a CDF comparison plot to this path. Defaults to None.
    POxcut : float or None, optional
        If not None, restrict to candidates with P(O|x) > POxcut and
        normalise both CDFs to unity (simulates the approach of selecting
        only confidently identified hosts). Defaults to None.
    plotlabel : str or None, optional
        Text label placed in the centre-bottom of the plot. Defaults to None.
    abc : str or None, optional
        Panel label (e.g. ``'(a)'``) placed in the upper-left corner of the
        figure in figure-coordinate space. Defaults to None.
    tag : str, optional
        String prefix added to the legend labels ``"Observed"`` and
        ``"Prior"``. Defaults to ``""``.

    Returns
    -------
    stat : float
        Maximum absolute difference between the observed and prior cumulative
        distributions. Smaller values indicate a better fit.
    """
    # sums the apparent mag priors over all FRBs to create a cumulative distribution
    fAppMagPriors = np.zeros([len(AppMags)])
    
    for i,amp in enumerate(AppMagPriors):
        fAppMagPriors += amp
    
    
    fObsPosteriors = np.array(flatten(ObsPosteriors))
    
    fObsMags = np.array(flatten(ObsMags))
    
    # we calculate a probability using a cumulative distribution
    prior_dist = np.cumsum(fAppMagPriors)
    
    if POxcut is not None:
        # cuts data to "good" FRBs only
        OK = np.where(fObsPosteriors > POxcut)[0]
        Ndata = len(OK)
        fObsMags = fObsMags[OK]
        fObsPosteriors = np.full([Ndata],1.) # effectively sets these to unity
    
    
    # makes a cdf in units of AppMags, with observations ObsMags weighted by ObsPosteriors
    obs_dist = make_cdf(AppMags,fObsMags,fObsPosteriors,norm=False)
    
    if POxcut is not None:
        # current techniques just assume we have the full distribution
        obs_dist /= obs_dist[-1]
        prior_dist /= prior_dist[-1]
    else:
        # the above is normalised to NFRB. We now divide it by this
        # might want to be careful here, and preserve this normalisation
        obs_dist /= NFRB
        prior_dist /= NFRB #((NFRB-PUprior)/NFRB) / prior_dist[-1]
    
    # we calculate something like the k-statistic. Includes NFRB normalisation
    diff = obs_dist - prior_dist
    stat = np.max(np.abs(diff))
    
    if plotfile is not None:
        plt.figure()
        plt.xlabel("Apparent magnitude $m_r$")
        plt.ylabel("Cumulative host galaxy distribution")
        plt.ylim(0,1)
        
        # calcs lowest x that is essentially at max
        ixmax = np.where(prior_dist > prior_dist[-1]*0.999)[0][0]
        # rounds it up to multiple of 5
        xmax = 5 * (int(AppMags[ixmax]/5.)+1)
        ixmin = np.where(prior_dist < 0.01)[0][-1]
        xmin = 5*(int(AppMags[ixmin]/5.))
        plt.xlim(xmin,xmax)
        
        #cx,cy = make_cdf_for_plotting(ObsMags,weights=ObsPosteriors)
        plt.plot(AppMags,obs_dist,label=tag+"Observed",color="black")
        plt.plot(AppMags,prior_dist,label=tag+"Prior",linestyle=":")
        plt.legend()
        
        # adds label to plot
        if plotlabel is not None:
            plt.text((xmin+xmax)/2.,0.05,plotlabel)
        
        if abc is not None:
            plt.text(0.02,0.9,abc,fontsize=16, transform=plt.gcf().transFigure)
        
        plt.tight_layout()
        plt.savefig(plotfile)
        plt.close()
        
    
    return stat

def make_cumulative_plots(NMODELS,NFRB,AppMags,AppMagPriors,ObsMags,ObsPosteriors,PUobs,
                                PUprior,plotfile,plotlabel,POxcut=None,abc=None,onlyobs=None,
                                greyscale=[],addpriorlabel=True):
    """
    Plot cumulative apparent magnitude distributions for multiple host models on one figure.

    Computes the same normalised prior and observed CDFs as
    ``calculate_ks_statistic``, but for ``NMODELS`` models simultaneously,
    overlaying them on a single figure with distinct line styles.

    All list-valued parameters that appear in ``calculate_ks_statistic``
    gain an additional leading dimension of size ``NMODELS`` here.

    Parameters
    ----------
    NMODELS : int
        Number of models to plot.
    NFRB : list of int, length NMODELS
        Number of FRBs for each model, used for normalisation.
    AppMags : list of np.ndarray, length NMODELS
        Apparent magnitude grid for each model.
    AppMagPriors : list of lists of np.ndarray, shape (NMODELS, NFRB, NMAG)
        Model prior p(m_r | DM_EG) for each model and FRB.
    ObsMags : list of lists of np.ndarray, shape (NMODELS, NFRB, NCAND)
        Observed candidate magnitudes for each model and FRB.
    ObsPosteriors : list of lists of np.ndarray, shape (NMODELS, NFRB, NCAND)
        PATH posteriors P(O_i|x) for each model and FRB.
    PUobs : list, length NMODELS
        Posterior P(U|x) per model (not used directly in the plot).
    PUprior : list, length NMODELS
        Prior P_U per model (not used directly in the plot).
    plotfile : str
        Output file path for the saved figure.
    plotlabel : list of str, length NMODELS
        Legend label prefix for each model.
    POxcut : float or None, optional
        If not None, restrict to candidates with P(O|x) > POxcut and
        normalise CDFs to unity. Defaults to None.
    abc : str or None, optional
        Panel label (e.g. ``'(a)'``) placed in the upper-left corner in
        figure-coordinate space. Defaults to None.
    onlyobs : int or None, optional
        If not None, only draw the observed CDF for the model with this
        index (useful when all models share the same observations). The
        observed line is then labelled ``"Observed"`` without a model prefix.
        Defaults to None.
    greyscale : list of int, optional
        Indices of models whose observed CDF should additionally be drawn
        in grey (for background reference). Defaults to ``[]``.
    addpriorlabel : bool, optional
        If True (default), append ``": Prior"`` to each model's legend entry.
        Set to False to use only ``plotlabel[imodel]`` as the label.

    Returns
    -------
    None
    """
    
    # arrays to hold created observed and prior distributions
    prior_dists = []
    obs_dists = []
    linestyles=[":","--","-.","-"]
    
    # loops over models to create prior distributions
    for imodel in np.arange(NMODELS):
        # sums the apparent mag priors over all FRBs to create a cumulative distribution
        fAppMagPriors = np.zeros([len(AppMags[imodel])])
    
        for i,amp in enumerate(AppMagPriors[imodel]):
            fAppMagPriors += amp
    
        fObsPosteriors = np.array(flatten(ObsPosteriors[imodel]))
        
        fObsMags = np.array(flatten(ObsMags[imodel]))
    
        # we calculate a probability using a cumulative distribution
        prior_dist = np.cumsum(fAppMagPriors)
        
        if POxcut is not None:
            # cuts data to "good" FRBs only
            OK = np.where(fObsPosteriors > POxcut)[0]
            Ndata = len(OK)
            fObsMags = fObsMags[OK]
            fObsPosteriors = np.full([Ndata],1.) # effectively sets these to unity
    
        
        # makes a cdf in units of AppMags, with observations ObsMags weighted by ObsPosteriors
        obs_dist = make_cdf(AppMags[imodel],fObsMags,fObsPosteriors,norm=False)
        
        if POxcut is not None:
            # current techniques just assume we have the full distribution
            obs_dist /= obs_dist[-1]
            prior_dist /= prior_dist[-1]
        else:
            # the above is normalised to NFRB. We now divide it by this
            # might want to be careful here, and preserve this normalisation
            obs_dist /= NFRB[imodel]
            prior_dist /= NFRB[imodel] #((NFRB-PUprior)/NFRB) / prior_dist[-1]
        
        # we calculate something like the k-statistic. Includes NFRB normalisation
        diff = obs_dist - prior_dist
        stat = np.max(np.abs(diff))
        
        obs_dists.append(obs_dist)
        prior_dists.append(prior_dist)
        
    # plotting!
    plt.figure()
    plt.xlabel("Apparent magnitude $m_r$")
    plt.ylabel("Cumulative host galaxy distribution")
    plt.ylim(0,1)
    
    for imodel in np.arange(NMODELS):
        
        if onlyobs is None or onlyobs == imodel:
            if onlyobs is not None:
                color = 'black'
                label = "Observed" # don't sub-label, since this stands in for all observed
            else:
                color=plt.gca().lines[-1].get_color()
                label = plotlabel[imodel]+": Observed"
            
            plt.plot(AppMags[imodel],obs_dists[imodel],label=label,
                        color=color)
        
        # adds gryescale 'background' plots of observed distributions
        if imodel in greyscale:
            plt.plot(AppMags[imodel],obs_dists[imodel],color="gray")
            # add these in greyscale, to highlight they are 'background' plots
            # this option never used, but experimented with.
            
        
        # calcs lowest x that is essentially at max
        ixmax = np.where(prior_dist > prior_dist[-1]*0.999)[0][0]
        # rounds it up to multiple of 5
        xmax = 5 * (int(AppMags[imodel][ixmax]/5.)+1)
        ixmin = np.where(prior_dist < 0.001)[0][-1]
        xmin = 5*(int(AppMags[imodel][ixmin]/5.))
        
        # sets this for each one - yes, it's random which is which, oh well!
        plt.xlim(xmin,xmax)
        
        #cx,cy = make_cdf_for_plotting(ObsMags,weights=ObsPosteriors)
        if addpriorlabel:
            label = plotlabel[imodel]+": Prior"
        else:
            label = plotlabel[imodel]
        
        plt.plot(AppMags[imodel],prior_dists[imodel],label=label,
                        linestyle=linestyles[imodel%4])
        
            
            
    if abc is not None:
        plt.text(0.02,0.9,abc,fontsize=16, transform=plt.gcf().transFigure)
    plt.legend(fontsize=12,loc="upper left")
    plt.tight_layout()
    plt.savefig(plotfile)
    plt.close()
    
    return None

def get_cand_properties(frblist):
    """
    Load PATH candidate host galaxy properties for a list of FRBs.

    Reads the pre-generated PATH CSV files from the ``frb`` package data
    directory (``frb/data/Galaxies/PATH/<FRB>_PATH.csv``) and extracts
    the columns ``['ang_size', 'mag', 'ra', 'dec', 'separation']`` for
    each FRB.

    Args:
        frblist (list of str): TNS FRB names (e.g. ``['FRB20180924B', ...]``).

    Returns:
        all_candidates (list of pd.DataFrame): one DataFrame per FRB,
            each with columns ``ang_size``, ``mag``, ``ra``, ``dec``,
            and ``separation``.
    """
    
    all_candidates=[]
    for i,name in enumerate(frblist):
    
        ######### Loads FRB, and modifes properties #########
        my_frb = FRB.by_name(name)
        #this_path = frbassociate.FRBAssociate(my_frb, max_radius=10.)
        
        # reads in galaxy info
        ppath = os.path.join(resources.files('frb'), 'data', 'Galaxies', 'PATH')
        pfile = os.path.join(ppath, f'{my_frb.frb_name}_PATH.csv')
        ptbl = pandas.read_csv(pfile)
        candidates = ptbl[['ang_size', 'mag', 'ra', 'dec', 'separation']]
        all_candidates.append(candidates)
    return all_candidates

def run_path(name,P_U=0.1,usemodel=False,sort=False,failOK=False,scale=0.5,ppath=None):
    """
    Run the PATH algorithm on a single FRB and return host association results.

    Loads the FRB object and its pre-generated PATH candidate table from the
    ``frb`` package, applies colour corrections to convert candidate magnitudes
    to r-band (using fixed offsets: IR: +0.65, gR: 0.65), sets up the
    FRB localisation ellipse and offset prior, and evaluates PATH posteriors.

    The magnitude prior used for the candidates is:

    - ``usemodel=False``: PATH's built-in ``'inverse'`` prior (uniform in log
      surface density).
    - ``usemodel=True``: the ``'user'`` prior, which must be set externally by
      pointing ``pathpriors.USR_raw_prior_Oi`` at a ``model_wrapper`` method
      before calling this function (typically done by
      ``wrapper.init_path_raw_prior_Oi``).

    The offset prior is always the ``'exp'`` model from PATH's ``'adopted'``
    standard priors, with scale 0.5 arcsec.

    Parameters
    ----------
    name : str
        TNS name of the FRB (e.g. ``'FRB20180924B'``).
    P_U : float, optional
        Prior probability that the true host galaxy is undetected. Defaults
        to 0.1.
    usemodel : bool, optional
        If True, use the externally set user prior for candidate magnitudes.
        Defaults to False.
    sort : bool, optional
        If True, sort the returned arrays by P(O|x) in ascending order.
        Defaults to False.
    failOK : bool, optional
        If True, allows a return without crashing
    ppath : string, optional
        If given, search this directory for optical data
    Returns
    -------
    P_O : np.ndarray
        Prior probability P(O_i) for each candidate host galaxy.
    P_Ox : np.ndarray
        Posterior probability P(O_i|x) for each candidate.
    P_Ux : float
        Posterior probability P(U|x) that the true host is undetected.
    mags : np.ndarray
        R-band apparent magnitudes of the candidates (after colour correction).
    ptbl : pd.DataFrame
        Full PATH candidate table loaded from the CSV file, with an
        additional ``'frb'`` column set to ``name``.
    """
    
    result={}
    
    # checks to see the name is prefaced by "FRB"
    if name[0:3] != "FRB":
        name = "FRB"+name
    
    ######### Loads FRB, and modifes properties #########
    try:
        my_frb = FRB.by_name(name)
    except:
        if failOK:
            #print("Warning - could not find frb file for ",name)
            result["Error"]=1
            return result
        else:
            print("Could not run_path - no frb data for FRB ",name)
            exit()
    result["Error"]=0 # it's OK!
    
    this_path = frbassociate.FRBAssociate(my_frb, max_radius=10.)
    
    # reads in galaxy info
    if ppath is None:
        ppath = os.path.join(resources.files('frb'), 'data', 'Galaxies', 'PATH')
    pfile = os.path.join(ppath, f'{my_frb.frb_name}_PATH.csv')
    
    try:
        ptbl = pandas.read_csv(pfile)
    except:
        if failOK:
            #print("Warning - could not find optical data for ",name)
            result["Error"]=1
            return result
        else:
            print("Could not run_path - no optical data for FRB ",name)
            exit()
    
    ngal = len(ptbl)
    ptbl["frb"] = np.full([ngal],name)
    
    # Load prior
    priors = load_std_priors()
    prior = priors['adopted'] # Default
    
    theta_new = dict(method='exp', 
                    max=priors['adopted']['theta']['max'], 
                    scale=scale)
    prior['theta'] = theta_new
    
    # change this to something depending on the FRB DM
    prior['U']=P_U
    
    candidates = ptbl[['ang_size', 'mag', 'ra', 'dec', 'separation']]
    
    # implements a correction to their relative magnitudes.
    # note that order is R, then I, then g
    if "VLT_FORS2_R" in ptbl:
        mags = np.array(candidates.mag.values)
    elif "VLT_FORS2_I" in ptbl:
        mags = np.array(candidates.mag.values) + 0.65
    elif "VLT_FORS2_g" in ptbl:
        mags = np.array(candidates.mag.values) - 0.65
    elif "GMOS_S_i" in ptbl:
        mags = np.array(candidates.mag.values) + 0.65
    elif "LRIS_I" in ptbl:
        mags = np.array(candidates.mag.values) + 0.65
    else:
        raise ValueError("Cannot implement colour correction")
        
    
    #this_path = PATH()
    this_path.init_candidates(candidates.ra.values,
                         candidates.dec.values,
                         candidates.ang_size.values,
                         mag=mags)
    this_path.frb = my_frb
    
    frb_eellipse = dict(a=np.abs(my_frb.sig_a),
                    b=np.abs(my_frb.sig_b),
                    theta=my_frb.eellipse['theta'])
    
    this_path.init_localization('eellipse', 
                            center_coord=this_path.frb.coord,
                            eellipse=frb_eellipse)
    
    # this results in a prior which is uniform in log space
    # when summed over all galaxies with the same magnitude
    if usemodel:
        this_path.init_cand_prior('user', P_U=prior['U'])
    else:
        this_path.init_cand_prior('inverse', P_U=prior['U'])
    
    # this is for the offset
    this_path.init_theta_prior(prior['theta']['method'], 
                            prior['theta']['max'],
                            prior['theta']['scale'])
    
    # in the case of a user-specified model, this P(O) is
    P_O=this_path.calc_priors()
    
    # Calculate p(O_i|x)
    debug = True
    P_Ox,P_Ux = this_path.calc_posteriors('fixed', 
                         box_hwidth=10., 
                         max_radius=10., 
                         debug=debug)
    
    # probability of x given O, and P(O)
    # note - if method is "user", then P(O) includes 1/density
    # if it is not, P(O) is renormalised, and includes 1/cumulative
    P_xO = this_path.p_xOi
    
    # mags already defined above
    #mags = candidates['mag']
    
    if sort:
        indices = np.argsort(P_Ox)
        P_O = P_O[indices]
        P_Ox = P_Ox[indices]
        mags = mags[indices]
        P_xO = P_xO[indices]
    
    
    result["PUx"] = P_Ux
    result["PO"] = P_O
    result["PxO"] = P_xO
    result["POx"] = P_Ox
    result["mags"] = mags
    result["ptbl"] = ptbl
    
    return result

