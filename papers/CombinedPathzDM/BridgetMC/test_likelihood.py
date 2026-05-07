""" 
Demonstrates how to set likelihood components for the MC analysis
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

def main():
    """
    Calculates likelihoods for fake survey
    """
    
    state = states.load_state("HoffmannHalo25") # old scattering
    #name = "CRAFT_CRACO_900"
    name = "short_fake_CRACO_900"
    sdir = resources.files('zdm').joinpath('../papers/CombinedPathzDM/BridgetMC/Surveys/')
    surveys, grids = loading.surveys_and_grids(survey_names = [name],repeaters=False,
                                                sdir=sdir,init_state=state)
    
    s = surveys[0]
    g = grids[0]
    
    opstate = op.OpticalState()
    model = opt.loudas_model(opstate)
    wrapper = opt.model_wrapper(model,g.zvals)
    
    ### sets system variables to point towards fake FRB data
    galdir = resources.files('zdm').joinpath('../papers/CombinedPathzDM/BridgetMC/CandidateFiles')
    frbdir = resources.files('zdm').joinpath('../papers/CombinedPathzDM/BridgetMC/FRBFiles')
    os.environ["ZDM_PATH_FRBDIR"] = str(frbdir)
    os.environ["ZDM_PATH_GALDIR"] = str(galdir)
    
    lltot,results = it.get_joint_path_zdm_likelihoods(g, s, wrapper, norm=True, psnr=True, Pn=False,
                                    pdmz=True, pNreps=True, ptauw=False, pwb=True,
                                    return_all=True)
    
    # we now construct posterior distributions
    # loop over FRBs. CHECK: if this total number of FRBs, or total with PATH results?
    
    
    weights = []
    thetas = []
    mags = []
    path_results = results['path_s']
    zdm_results = results['zdm_s']
    
    # print out the keys  so we know what's available!
    print(path_results.keys())
    print(zdm_results.keys())
    
    for i in np.arange(path_results["NFRB"]):
        pxrad=zdm_results["pxrad"] # probability of observing radio properties
        
        for j in np.arange(path_results["Ncand"][i]):
            weights = weights+[path_results["POx"][i][j]]
            mags = mags+[path_results["ObsMags"][i][j]]
            thetas = thetas+[path_results["seps"][i][j]/path_results["sizes"][i][j]]
    
    
    # we now construct the expectation
    # does this for each point on the zdm grid
    mvals = wrapper.AppMags
    pms = np.zeros([mvals.size])
    for i,z in enumerate(g.zvals):
        weight = np.sum(g.rates[i,:])
        pmag = wrapper.get_pm_g_z(z)
        pms[:] += pmag * weight
    
    
    plt.figure()
    bins = wrapper.AppBins
    plt.hist(mags,weights=weights,bins=bins,label="Posteriors")
    norm = np.sum(pms)/np.sum(weights)
    plt.plot(mvals,pms/norm,label="Truth")
    plt.xlim(10,30)
    plt.xlabel("magnitudes")
    plt.ylabel("posterior")
    plt.tight_layout()
    plt.savefig("post_mags.png")
    plt.close()
    
    
    plt.figure()
    bins = np.linspace(0,6.,61)
    
    truth = np.exp(-bins/0.5) * np.sum(weights)/10
    
    plt.hist(thetas,bins=bins,weights=weights,label="posteriors")
    plt.plot(bins,truth,label="truth")
    plt.legend()
    plt.xlim(0,6)
    plt.xlabel("theta/phi")
    plt.tight_layout()
    plt.savefig("post_offsets.png")
    plt.close()
    
main()
