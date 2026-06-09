""" 

This script runs the likelihood calculation for MC truth parameters,
(which are reproduced from emcee fitting), and analyses the posterior
distribution of observables in host magnitude and angular offset.

It requires one to have first run "run_path_only.py", since it 
makes comparison plots of posterior observables to those derived
when running PATH.
"""
import os
import time
import pandas as pd

from zdm import iteration as it
from zdm import loading
from zdm import optical as opt
from zdm import optical_params as op
from zdm import states

import numpy as np

from matplotlib import pyplot as plt
import importlib.resources as resources

import matplotlib
defaultsize=14
ds=4
font = {'family' : 'Helvetica',
        'weight' : 'normal',
        'size'   : defaultsize}
matplotlib.rc('font', **font)


def main(prefix,pathfile,wpathfile,hostfile):
    """
    Calculates likelihoods for fake survey
    """
    opdir = "LikelihoodTests/"
    if not os.path.exists(opdir):
        os.mkdir(opdir)
    
    opdir = opdir+prefix
    
    state = states.load_state("HoffmannHalo25") # old scattering
    state.MW.sigmaHalo=0.
    state.MW.sigmaDM=0.
    state.energy.lEmin = 30.
    
    # survey file remains the same, regardless of the localisation
    name = "short_fake_CRACO_900" # do this with very short
    #name = "1M_fake_CRACO"
    sdir = resources.files('zdm').joinpath('../papers/CombinedPathzDM/BridgetMC/Surveys/')
    surveys, grids = loading.surveys_and_grids(survey_names = [name],repeaters=False,
                                                sdir=sdir,init_state=state)
    
    s = surveys[0]
    g = grids[0]
    
    opstate = op.OpticalState()
    opstate.loudas.fsfr=0.5
    opstate.id.pU_method=1
    opstate.id.pU_min=14
    opstate.id.pU_max=22
    model = opt.loudas_model(opstate)
    wrapper = opt.model_wrapper(model,g.zvals)
    
    ### sets system variables to point towards fake FRB data
    relgaldir="../papers/CombinedPathzDM/BridgetMC/"+prefix+"CandidateFiles"
    galdir = resources.files('zdm').joinpath(relgaldir)
    relfrbdir = "../papers/CombinedPathzDM/BridgetMC/"+prefix+"FRBFiles"
    frbdir = resources.files('zdm').joinpath(relfrbdir)
    os.environ["ZDM_PATH_FRBDIR"] = str(frbdir)
    os.environ["ZDM_PATH_GALDIR"] = str(galdir)
    
    # adds this flag. Actual value doesn't matter, provided it is set
    os.environ["ZDM_CORRECT_DRIVER_FLAG"] = "False"
    
    lltot,results = it.get_joint_path_zdm_likelihoods(g, s, wrapper, norm=True, psnr=True, Pn=False,
                                    pdmz=True, pNreps=True, ptauw=False, pwb=True,
                                    return_all=True)
    
    # adds this flag. Actual value doesn't matter, provided it is set
    os.environ["ZDM_CORRECT_DRIVER_FLAG"] = "True"
                         
    lltot2,results2 = it.get_joint_path_zdm_likelihoods(g, s, wrapper, norm=True, psnr=True, Pn=False,
                                    pdmz=True, pNreps=True, ptauw=False, pwb=True,
                                    return_all=True)
    
    
    # we now construct posterior distributions
    # loop over FRBs. CHECK: if this total number of FRBs, or total with PATH results?
    weights = []
    thetas = []
    thetas2 = []
    mags = []
    path_results = results['path_s']
    zdm_results = results['zdm_s']
    csum_pux = path_results["sumPUx"]
    
    # results when using the correction function
    cpath_results = results2['path_s']
    czdm_results = results2['zdm_s']
    cweights = []
    cthetas = []
    cthetas2 = []
    cmags = []
    csum_pux = cpath_results["sumPUx"]
    
    # print out the keys so we know what's available!
    for i in np.arange(path_results["NFRB"]):
        
        for j in np.arange(path_results["Ncand"][i]):
            weights = weights+[path_results["POx"][i][j]]
            mags = mags+[path_results["ObsMags"][i][j]]
            thetas = thetas+[path_results["seps"][i][j]/path_results["sizes"][i][j]]
            effr = (path_results["sizes"][i][j]**2 + 0.5**2)**0.5
            thetas2 = thetas2+[path_results["seps"][i][j]/effr]
        
        
        # should actually be the same loop as above
        for j in np.arange(cpath_results["Ncand"][i]):
            cweights = cweights+[cpath_results["POx"][i][j]]
            cmags = cmags+[cpath_results["ObsMags"][i][j]]
            cthetas = cthetas+[cpath_results["seps"][i][j]/cpath_results["sizes"][i][j]]
            ceffr = (cpath_results["sizes"][i][j]**2 + 0.5**2)**0.5
            cthetas2 = cthetas2+[cpath_results["seps"][i][j]/ceffr]
    
    # we now construct the expectation
    # does this for each point on the zdm grid
    mvals = wrapper.AppMags
    pms = np.zeros([mvals.size])
    for i,z in enumerate(g.zvals):
        weight = np.sum(g.rates[i,:])
        pmag = wrapper.get_pm_g_z(z)
        pms[:] += pmag * weight
    
    
    mags = np.array(mags)
    weights = np.array(weights)
    
    cmags = np.array(cmags)
    cweights = np.array(cweights)
    
    # ensure correct weighting
    NGal = len(mags)
    weights *= NGal/np.sum(weights)
    pms *= NGal/np.sum(pms)
    
    
    # adds in simple PATH cuts
    path = pd.read_csv(pathfile)
    if len(path) > 0:
        dopath = True
    else:
        dopath=False
    
    if dopath:
        pweights = np.full([len(path)],NGal / len(path))
    
    
    # gets true assigned host distribution
    assigned = pd.read_csv(hostfile)
    truemags = assigned["mag"].values
    truemags = np.sort(truemags)
    NT = truemags.size
    OK = np.where(truemags >= 14.)
    truemags = truemags[OK]
    OK = np.where(truemags <= 22.)
    truemags = truemags[OK]
    NT = truemags.size
    ctruemags = np.linspace(1./NT,1.,NT)
    
    wpath = pd.read_csv(wpathfile)
    wpweights = wpath["POx"] * NGal / np.sum(wpath["POx"])
    
    trueweights = np.full([NT],NGal/NT)
    
    plt.figure()
    bins = wrapper.AppBins
    plt.hist(mags,bins=bins,label="All host candidates")
    plt.hist(mags,weights=weights,bins=bins,label="Posterior weights",alpha=0.7)
    if dopath:
        plt.hist(path["mags"],weights=pweights,bins=bins,label="Standard PATH",alpha=0.5)
        #plt.text('a): $\\sigma_{\\rm frb} = 0.5^{"}$')
    #   else:
        #plt.text('b): $\\sigma_{\\rm frb} = 30^{"}$')
    plt.hist(wpath["mags"],weights=wpweights,bins=bins,label="weighted PATH",alpha=0.3)
    plt.plot(mvals,pms,label="Truth")
    plt.hist(truemags,bins=bins,weights=trueweights,label="Assigned")
    
   
    
    plt.xlim(13,23)
    plt.xlabel("$m_r$")
    plt.ylabel("$N(m_r)$")
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(opdir+"post_mags.png")
    plt.close()
    
    # performs a CDF plot of each
    iorder = np.argsort(mags)
    x = mags[iorder]
    weights = weights[iorder]
    # unweighted
    raw_cdf = np.linspace(1./mags.size,1.,mags.size)
    # posterior
    post_cdf = np.cumsum(weights)
    post_cdf /= post_cdf[-1]
    
    
    # Corrected post CDF
    iorder = np.argsort(cmags)
    cx = cmags[iorder]
    cweights = cweights[iorder]
    # unweighted
    craw_cdf = np.linspace(1./mags.size,1.,mags.size)
    # posterior
    cpost_cdf = np.cumsum(cweights)
    cpost_cdf /= cpost_cdf[-1]
    
    # traditional_path
    if dopath:
        xpath = np.sort(path["mags"])
        trad_cdf = np.linspace(1./xpath.size,1.,xpath.size)
    
    # weighted_path
    wxmags = wpath["mags"].values
    iwxpath = np.argsort(wxmags)
    wxpath = wxmags[iwxpath]
    wp_cdf = wpath["POx"].values
    wp_cdf = wp_cdf[iwxpath]
    wp_cdf = np.cumsum(wp_cdf)
    wp_cdf /= wp_cdf[-1]
    
    # truth
    OK = np.where(mvals > 14)[0]
    mvals = mvals[OK]
    pms = pms[OK]
    OK = np.where(mvals < 22)[0]
    mvals = mvals[OK]
    pms = pms[OK]
    pms = np.cumsum(pms)
    pms /= pms[-1]
    
    # plot
    plt.figure()
    plt.plot(mvals,pms,label="Simulated truth",color="black",linestyle=":")
    plt.plot(truemags,ctruemags,label="Assigned hosts",color="black",linestyle="-")
    
    plt.plot(x,raw_cdf,label="All candidates",linestyle=":")
    plt.plot(cx,cpost_cdf,label="zDM+PATH (this work)",linestyle="-")
    plt.plot(x,post_cdf,label="  (uncorrected $\\rho_{D16}$)",linestyle=":",color=plt.gca().lines[-1].get_color())
    
    if dopath:
        plt.plot(xpath,trad_cdf,label="PATH: $P(O|x)>0.95$",linestyle="--")
        plt.text(13,1.05,"a): $\\sigma_{\\rm FRB} = 0.5^{''}$")
    else:
        plt.text(13,1.05,"b): $\\sigma_{\\rm FRB} = 30^{''}$")
    plt.plot(wxpath,wp_cdf,label="PATH: $\\propto P(O|x)$",linestyle="-.",color="red")
    
    plt.xlabel("$m_r$")
    plt.ylabel("CDF$(m_r)$")
    plt.legend(fontsize=12)
    plt.xlim(14,22)
    plt.ylim(0,1)
    plt.tight_layout()
    plt.savefig(opdir+"mag_dist_cumulative.png")
    plt.close()
    
    
    plt.figure()
    bins = np.linspace(0,6.,61)
    
    # note: "truth" does *NOT* account for angular smearing of FRB localisation
    # However, PATH does! So we don't expect agreement
    truth = bins*np.exp(-bins/0.5) * np.sum(weights)/10
    truth *= NGal / np.sum(truth)
    plt.hist(thetas,bins=bins,label="All host candidates")
    plt.hist(thetas,bins=bins,weights=weights,label="zDM+PATH: weighted",alpha=0.5)
    #plt.hist(thetas2,bins=bins,weights=weights,label="Adjusted posterior weights",alpha=0.3)
    plt.plot(bins,truth,label="Simulated truth",color="black")
    plt.legend()
    plt.xlim(0,6)
    plt.xlabel("$\\theta/\\phi$")
    plt.ylabel("$N(\\theta/\\phi)$")
    plt.tight_layout()
    plt.savefig(opdir+"post_offsets.png")
    plt.close()

def get_convolution():
    """
    convolves gaussian with exponential
    """
    N=7
    xvals = np.linspace(-6,6,N)
    xvals = np.repeat(xvals,N)
    xvals = xvals.reshape([N,N])
    print(xvals)
    
    yvals = xvals.T
    print(yvals)
    
    rs = (xvals**2 + yvals**2)**0.5
    p1 = np.exp(xvals**2 + yvals**2)

pathfile = "m14cut_hosts_1000.csv"
wpathfile = "w_m14cut_hosts_1000.csv"
prefix=""
hostfile = "craco_assigned_galaxies.csv"
main(prefix,pathfile,wpathfile,hostfile)

hostfile = "loc30_craco_assigned_galaxies.csv"
pathfile = "loc30_m14cut_hosts_1000.csv"
wpathfile = "loc30_w_m14cut_hosts_1000.csv"
prefix="loc30"
main(prefix,pathfile,wpathfile,hostfile)
