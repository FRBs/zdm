""" 
Demonstrates how to set likelihood components for the MC analysis

Also plots posteriors for best-fitting parameters

"""
import os
import time

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


def main():
    """
    Calculates likelihoods for fake survey
    """
    
    opdir = "CRAFTLikelihood/"
    if not os.path.exists(opdir):
        os.mkdir(opdir)
    
    state = states.load_state("HoffmannHalo25") # old scattering
    
    name = "CRAFT_ICS_892"
    sdir=None
    surveys, grids = loading.surveys_and_grids(survey_names = [name],repeaters=False,\
                                            sdir=sdir,init_state=state)
    
    s = surveys[0]
    g = grids[0]
    
    opstate = op.OpticalState()
    model = opt.loudas_model(opstate)
    wrapper = opt.model_wrapper(model,g.zvals)
    
    
    lltot,results = it.get_joint_path_zdm_likelihoods(g, s, wrapper, norm=True, psnr=True, Pn=False,
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
    print("Got optical results for ",path_results["OK"])
    for key in path_results.keys():
        if np.iterable(path_results[key]):
            print(key,len(path_results[key]))
    # print out the keys  so we know what's available!
    for i,OK in enumerate(path_results["OK"]):
        pxrad=zdm_results["pxrad"][i] # probability of observing radio properties
        
        for j in np.arange(path_results["Ncand"][i]):
            weights = weights+[path_results["POx"][i][j]]
            mags = mags+[path_results["ObsMags"][i][j]]
            thetas = thetas+[path_results["seps"][i][j]/path_results["sizes"][i][j]]
            effr = (path_results["sizes"][i][j]**2 + 0.5**2)**0.5
            thetas2 = thetas2+[path_results["seps"][i][j]/effr]
    
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
    
    # ensure correct weighting
    NGal = len(mags)
    weights *= NGal/np.sum(weights)
    pms *= NGal/np.sum(pms)
    
    plt.figure()
    bins = wrapper.AppBins
    
    
    plt.hist(mags,bins=bins,label="All host candidates")
    plt.hist(mags,weights=weights,bins=bins,label="Posterior weights",alpha=0.5)
    plt.plot(mvals,pms,label="Truth")
    
    plt.xlim(13,23)
    plt.xlabel("$m_r$")
    plt.ylabel("$N(m_r)$")
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(opdir+"post_mags.png")
    plt.close()
    
    
    plt.figure()
    bins = np.linspace(0,6.,61)
    
    # note: "truth" does *NOT* account for angular smearing of FRB localisation
    # However, PATH does! So we don't expect agreement
    truth = bins*np.exp(-bins/0.5) * np.sum(weights)/10
    truth *= NGal / np.sum(truth)
    plt.hist(thetas,bins=bins,label="All host candidates")
    plt.hist(thetas,bins=bins,weights=weights,label="Posterior weights",alpha=0.5)
    #plt.hist(thetas2,bins=bins,weights=weights,label="Adjusted posterior weights",alpha=0.3)
    plt.plot(bins,truth,label="truth")
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

    

main()
