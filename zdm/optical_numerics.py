"""
Contains files related to numerical optimisation
of FRB host galaxy parameters. Similar to iteration.py
for the grid.
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
from frb.associate import frbassociate
    
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
        istat [int]: which stat to use? 0 = ks stat. 1 = mak likelihood
    
    
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
    
    NFRB,AppMags,AppMagPriors,ObsMags,ObsPriors,ObsPosteriors,PUprior,PUobs,sumPUprior,sumPUobs,frbs,dms = calc_path_priors(frblist,ss,gs,wrappers,verbose=False)
    
    # we re-normalise the sum of PUs by NFRB
    
    # prevents infinite plots being created
    if istat==0:
        stat = calculate_ks_statistic(NFRB,AppMags,AppMagPriors,ObsMags,ObsPosteriors,
                sumPUobs,sumPUprior,plotfile=None,POxcut=POxcut)
    elif istat==1:
        stat = calculate_likelihood_statistic(NFRB,AppMags,AppMagPriors,ObsMags,ObsPosteriors,
                PUobs,PUprior,plotfile=None,POxcut=POxcut)
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

    
def calc_path_priors(frblist,ss,gs,wrappers,verbose=True,usemodel=True,P_U=0.1):
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
    
    # old version creating 1D lists
    #allObsMags = None
    #allPOx = None
    #allMagPriors = None
    
    # new version recording one list per FRB. For max likelihood functionality
    allObsMags = []
    allPOx = []
    allPO = []
    allMagPriors = []
    
    sumPU = 0.
    sumPUx = 0.
    allPU = []
    allPUx = []
    nfitted = 0
    
    frbs=[]
    dms=[]
    
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
            wrapper.init_path_raw_prior_Oi(DMEG,g)
        
            # extracts priors as function of absolute magnitude for this grid and DMEG
            MagPriors = wrapper.priors
        else:
            MagPriors = None
            
        # defunct now
        #mag_limit=26  # might not be correct. TODO! Should be in FRB object
        
        # calculates unseen prior
        if usemodel:
            P_U = wrapper.estimate_unseen_prior()
        #MagPriors[:] = 1./len(MagPriors) # log-uniform priors when no model used
        
        
        P_O,P_Ox,P_Ux,ObsMags,ptbl = run_path(frb,usemodel=usemodel,P_U = P_U)
        
        
        # replaces PO value with raw PO value, i.e. excluding the driver sigma
        if usemodel:
            P_O = wrapper.path_base_prior(ObsMags)
            
        # kept here for debugging
        if False:
            print("P_U is ",P_U)
            print("P_O is ",P_O)
            print("P_Ox is ",P_Ox)
            plt.figure()
            plt.plot(AppMags,MagPriors)
            plt.show()
            plt.close()
        
        if i==0:
            allgals = ptbl
        else:
            allgals = pandas.concat([allgals,ptbl], ignore_index=True)
        
        ObsMags = np.array(ObsMags)
        
        # new version creating a list of lists
        allObsMags.append(ObsMags)
        allPOx.append(P_Ox)
        allPO.append(P_O)
        allMagPriors.append(MagPriors)
        
        sumPU += P_U
        sumPUx += P_Ux
        allPU.append(P_U)
        allPUx.append(P_Ux)
    
    subset = allgals[['frb','mag','VLT_FORS2_R']].copy()
    
    # saves all galaxies
    if not os.path.exists("allgalaxies.csv"):
        subset.to_csv("allgalaxies.csv",index=False)
    
    return nfitted,AppMags,allMagPriors,allObsMags,allPO,allPOx,allPU,allPUx,sumPU,sumPUx,frbs,dms


def calculate_likelihood_statistic(NFRB,AppMags,AppMagPriors,ObsMags,ObsPosteriors,PUobs,
                                PUprior,plotfile=None,POxcut=None):
    """
    Calculates a likelihood for each of the FRBs, and returns the log-likelihood.
    
    The inputs are in two categories. One is a form of lists of lists, where there is one list for
    each FRB, and one entry in that list for each host galaxy candidate. Size is NFRB x NCAND
    
    The other input is where the length of the list matches the internal array size used to
    calculate priors on host magnitudes. Size is either NMAG or NFRBxNMAG
    
    Inputs:
        AppMags [array of floats: NMAG]: array listing apparent magnitudes used to calculate priors
        AppMagPrior [array of floats NFRB xNMAG]: array giving prior on AppMags
        ObsMags: list of lists of floats giving observed magnitudes m_r of host candidates 
        ObsPosteriors: list of lists float of posterior values P(O|x) corresponding to ObsMags
        PUobs [float]: posterior on unseen probability
        PUprior [float]: prior on PU
        plotfile: set to name of output file for comparison plot
        POxcut: if not None, cut data to fixed POx. Used to simulate current techniques
    
    Returns:
        log likelihood of the observation
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
    Turns a list of lists into a single list
    """
    return [x for xs in xss for x in xs]

def calculate_ks_statistic(NFRB,AppMags,AppMagPriors,ObsMags,ObsPosteriors,PUobs,
                                PUprior,plotfile=None,POxcut=None,plotlabel=None,abc=None,tag=""):
    """
    Calculates a ks-like statistic to be proxy for goodness-of-fit
    We must set each AppMagPriors to 1.-PUprior at the limiting magnitude for each observation,
    and sum the ObsPosteriors to be equal to 1.-PUobs at that magnitude.
    Then these are what gets summed.
    
    This can be readily done by combining all ObsMags and ObsPosteriors into a single long list,
    since this should already be correctly normalised. Priors require their own weight.
    
    Inputs:
        AppMags: array listing apparent magnitudes
        AppMagPriors: list of lists giving priors on AppMags for each FRB
        ObsMags: list of observed magnitudes
        ObsPosteriors: list of posterior values corresponding to ObsMags
        PUobs: posterior on unseen probability
        PUprior: prior on PU
        plotfile: set to name of output file for comparison plot
        POxcut: if not None, cut data to fixed POx. Used to simulate current techniques
        abc [None]: add label, e.g. (a), to upper left
        tag [string]: string to prefix labels
    
    Returns:
        k-like statistic of biggest obs/prior difference
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
                                PUprior,plotfile,plotlabel,POxcut=None,abc=None,onlyobs=None):
    """
    Creates cumulative plots of KS-like behaviour for multiple fit outcomes
    
    Inputs: see "calculate_ks_statistic" except:
        - NMODELS (int): number of models to plot
        - abc remains unchanged
        - NFRB remains unchanged
        - plotfile remains unchanged
        - onlyobs (int): if not None, only plot observed distribution for this case
        - all other parameters have a leading dimension of NMODELS
    
    Inputs from "calculate_ks_statistic" with extra NMODELS dimension:
        AppMags: array listing apparent magnitudes
        AppMagPriors: list of lists giving priors on AppMags for each FRB
        ObsMags: list of observed magnitudes
        ObsPosteriors: list of posterior values corresponding to ObsMags
        PUobs: posterior on unseen probability
        PUprior: prior on PU
        POxcut: if not None, cut data to fixed POx. Used to simulate current techniques
    
    Returns:
        None
    """
    
    # arrays to hold created observed and prior distributions
    prior_dists = []
    obs_dists = []
    
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
        
        # calcs lowest x that is essentially at max
        ixmax = np.where(prior_dist > prior_dist[-1]*0.999)[0][0]
        # rounds it up to multiple of 5
        xmax = 5 * (int(AppMags[imodel][ixmax]/5.)+1)
        ixmin = np.where(prior_dist < 0.001)[0][-1]
        xmin = 5*(int(AppMags[imodel][ixmin]/5.))
        
        # sets this for each one - yes, it's random which is which, oh well!
        plt.xlim(xmin,xmax)
        
        #cx,cy = make_cdf_for_plotting(ObsMags,weights=ObsPosteriors)
        plt.plot(AppMags[imodel],prior_dists[imodel],label=plotlabel[imodel]+": Prior",
                        linestyle=":")
        if onlyobs is None or onlyobs == imodel:
            if onlyobs is not None:
                color='black'
            else:
                color=plt.gca().lines[-1].get_color()
            plt.plot(AppMags[imodel],obs_dists[imodel],label=plotlabel[imodel]+": Observed",
                        color=color)
            
            
    if abc is not None:
        plt.text(0.02,0.9,abc,fontsize=16, transform=plt.gcf().transFigure)
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(plotfile)
    plt.close()
    
    return None

    

def get_cand_properties(frblist):
    """
    Returns properties of galaxy candidates for FRBs
    
    Args:
        frblist: list of strings giving FRB names
    
    Returns:
        all_candidates: list of pandas dataframes containing candidate info
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
        
def run_path(name,P_U=0.1,usemodel = False, sort=False):
    """
    evaluates PATH on an FRB
    
    Args:
        P_U [float]: unseen prior
        usemodel [bool]: if True, use user-defined P_O|x model
        sort [bool]: if True, sort candidates by posterior
    
    """
    
    ######### Loads FRB, and modifes properties #########
    my_frb = FRB.by_name(name)
    this_path = frbassociate.FRBAssociate(my_frb, max_radius=10.)
    
    
    # do NOT do the below method!
    #
    
    # do NOT do the below!!
    #my_frb.set_ee(my_frb.sig_a,my_frb.sig_b,my_frb.eellipse['theta'],
    #            my_frb.eellipse['cl'],True)
    
    # reads in galaxy info
    ppath = os.path.join(resources.files('frb'), 'data', 'Galaxies', 'PATH')
    pfile = os.path.join(ppath, f'{my_frb.frb_name}_PATH.csv')
    ptbl = pandas.read_csv(pfile)
    
    ngal = len(ptbl)
    ptbl["frb"] = np.full([ngal],name)
    
    # Load prior
    priors = load_std_priors()
    prior = priors['adopted'] # Default
    
    theta_new = dict(method='exp', 
                    max=priors['adopted']['theta']['max'], 
                    scale=0.5)
    prior['theta'] = theta_new
    
    # change this to something depending on the FRB DM
    prior['U']=P_U
    
    candidates = ptbl[['ang_size', 'mag', 'ra', 'dec', 'separation']]
    
    
    #this_path = PATH()
    this_path.init_candidates(candidates.ra.values,
                         candidates.dec.values,
                         candidates.ang_size.values,
                         mag=candidates.mag.values)
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
    
    P_O=this_path.calc_priors()
    
    # Calculate p(O_i|x)
    debug = True
    P_Ox,P_Ux = this_path.calc_posteriors('fixed', 
                         box_hwidth=10., 
                         max_radius=10., 
                         debug=debug)
    
    mags = candidates['mag']
    
    if sort:
        indices = np.argsort(P_Ox)
        P_O = P_O[indices]
        P_Ox = P_Ox[indices]
        mags = mags[indices]
    
    return P_O,P_Ox,P_Ux,mags,ptbl

