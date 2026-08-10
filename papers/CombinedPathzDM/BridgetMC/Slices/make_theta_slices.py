""" 
This script makes slices in likelihood space to test the evaluation of various parameters.

It does this for the artificial simulation based on the methods of Bridget Andersen

In this script, we use several alternatives while varying theta
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
import multiprocessing as mp

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
    
    # directory to store output
    opdir="OP/"
    if not os.path.exists(opdir):
        os.mkdir(opdir)
    
    # declare some globals
    global names,state,opstate,sdir,psnr,Pn,pdmz,pNreps,ptauw,pwb,PATH,norm,OneDOnly
    psnr=True
    Pn=False
    pdmz=True
    pNreps=False
    ptauw=False
    pwb=True
    PATH=True
    norm=True
    OneDOnly=False
    
    global nz,ndm,zmax,dmmax
    nz = 500
    ndm = 1400
    zmax=5
    dmmax=7000
    
    state = states.load_state("HoffmannHalo25") # old scattering
    # increase number of width bins?
    
    
    name = "short_fake_CRACO_900"
    #name = "very_short_fake_CRACO_900_v2"
    sdir = resources.files('zdm').joinpath('../papers/CombinedPathzDM/BridgetMC/Surveys/')
    names = [name]
    
    ### sets system variables to point towards fake FRB data
    galdir = resources.files('zdm').joinpath('../papers/CombinedPathzDM/BridgetMC/CandidateFiles')
    frbdir = resources.files('zdm').joinpath('../papers/CombinedPathzDM/BridgetMC/FRBFiles')
    os.environ["ZDM_PATH_FRBDIR"] = str(frbdir)
    os.environ["ZDM_PATH_GALDIR"] = str(galdir)
    
    opstate = op.OpticalState()
    opstate.loudas.fsfr=0.5
    opstate.id.pU_method=1
    opstate.id.pU_min=14
    opstate.id.pU_max=22
    
    N=61
    thetas = np.linspace(0.3,0.6,N)
    
    lls = []
    labels = []
    
    # always the case
    state.MW.sigmaHalo=0.
    state.MW.sigmaDM=0.
    state.energy.lEmin=30
    
    
    
    ##### Updated PATH with proper selection cuts #######
    # twice as wide for candidate searches
    names=["short_fake_CRACO_900"]
    opfile = opdir+"wx6_theta_std.npy"
    if os.path.exists(opfile):
        ll = np.load(opfile)
        lls.append(ll)
        labels.append("$60^{''} \\times 60^{''}$")
    else:
        galdir = resources.files('zdm').joinpath('../papers/CombinedPathzDM/BridgetMC/wx6CandidateFiles')
        os.environ["ZDM_PATH_GALDIR"] = str(galdir)
        Pool = mp.get_context('fork').Pool
        
        # full calculation with zdm, psnr, etc etc etc
        with Pool() as pool:
            ll = pool.map(wrap,thetas)
        ll = np.array(ll)
        np.save(opfile,ll)
        exit()
    
    
    ##### Updated PATH with proper selection cuts #######
    # twice as wide for candidate searches
    #names=["short_fake_CRACO_900"]
    #opfile = opdir+"wx2_theta_std.npy"
    #if os.path.exists(opfile):
    #    ll = np.load(opfile)
    #    lls.append(ll)
    #    labels.append("$20^{''} \\times 20^^{''} image$")
    #else:
    #    galdir = resources.files('zdm').joinpath('../papers/CombinedPathzDM/BridgetMC/wx2CandidateFiles')
    #    os.environ["ZDM_PATH_GALDIR"] = str(galdir)
    #    Pool = mp.get_context('fork').Pool
        
    #    # full calculation with zdm, psnr, etc etc etc
    #    with Pool() as pool:
    #        ll = pool.map(wrap,thetas)
    #    ll = np.array(ll)
    #    np.save(opfile,ll)
    #    exit()
    
    
    ##### Updated PATH with proper selection cuts #######
    
    names=["short_fake_CRACO_900"]
    opfile = opdir+"bkgddiv10_theta_std.npy"
    if os.path.exists(opfile):
        ll = np.load(opfile)
        lls.append(ll)
        labels.append("$0.1 \\rho(m)$")
    else:
        
        print("Before running this, alter the background by hand in optical.py.")
        print("Do this via by adding 'Sigma_ms /= 10.' in path_raw_prior_Oi")
        print(" Then comment out this message")
        exit()
        Pool = mp.get_context('fork').Pool
        
        # full calculation with zdm, psnr, etc etc etc
        with Pool() as pool:
            ll = pool.map(wrap,thetas)
        ll = np.array(ll)
        np.save(opfile,ll)
        exit()
    
    ##### Updated PATH with proper selection cuts #######
    
    names=["short_fake_CRACO_900"]
    opfile = opdir+"bkgdx10_theta_std.npy"
    if os.path.exists(opfile):
        ll = np.load(opfile)
        lls.append(ll)
        labels.append("$10 \\rho(m)$")
    else:
        print("Before running this, alter the background by hand in optical.py.")
        print("Do this via by adding 'Sigma_ms *= 10.' in path_raw_prior_Oi")
        print(" Then comment out this message")
        exit()
        Pool = mp.get_context('fork').Pool
        
        # full calculation with zdm, psnr, etc etc etc
        with Pool() as pool:
            ll = pool.map(wrap,thetas)
        ll = np.array(ll)
        np.save(opfile,ll)
        exit()
    
    names=["short_fake_CRACO_900"]
    opfile = opdir+"theta_std.npy"
    if os.path.exists(opfile):
        ll = np.load(opfile)
        lls.append(ll)
        labels.append("$10^{''} \\times 10^{''}$")
    else:
        
        Pool = mp.get_context('fork').Pool
        
        # full calculation with zdm, psnr, etc etc etc
        with Pool() as pool:
            ll = pool.map(wrap,thetas)
        ll = np.array(ll)
        np.save(opfile,ll)
        exit()
    
    
    
    ##### Updated PATH with proper selection cuts #######
    
    #names=["fake_CRACO_900"]
    #opfile = opdir+"long_theta_std.npy"
    #if os.path.exists(opfile):
    #    ll = np.load(opfile)
    #    lls.append(ll)
    #    labels.append("10,000 FRBs")
    #else:
    #    
    #    Pool = mp.get_context('fork').Pool
    #    
    #    # full calculation with zdm, psnr, etc etc etc
    #    with Pool(6) as pool:
    #        ll = pool.map(wrap,thetas)
    #    ll = np.array(ll)
    #    np.save(opfile,ll)
    #    exit()
    
    
    # re-ordering for plotting purposes
    #newll=[lls[3],lls[4],lls[2],lls[1],lls[0]]
    #newlabels=[labels[3],labels[4],labels[2],labels[1],labels[0]]
    #labels = newlabels
    #lls=newll
    
    #newll=[lls[3],lls[4],lls[2],lls[1],lls[0]]
    #newlabels=[labels[3],labels[4],labels[2],labels[1],labels[0]]
    newll=[lls[3],lls[2],lls[1],lls[0]]
    newlabels=[labels[3],labels[2],labels[1],labels[0]]
    labels = newlabels
    lls=newll
    
    
    ########## PLOTTING #########
    
    styles=["-","--",":","-."]
    
    plt.figure()
    ax = plt.gca()
    ax.tick_params(top=True)
    plt.minorticks_on()
    for i,ll in enumerate(lls):
        ll -= np.max(ll)
        plt.plot(thetas,ll,label=labels[i],linestyle=styles[i%4])
    plt.ylim(-2,0)
    plt.xlim(0.3,0.6)
    plt.yticks(np.linspace(-2,0,5))
    plt.plot([0.5,0.5],[-40.,0.],color="black",linestyle="--",label="Truth")
    plt.legend()
    plt.xlabel("$\\theta_0$")
    plt.ylabel("${\\mathcal{L}}(\\theta_0)$")
    plt.tight_layout()
    plt.savefig(opdir+"ll_scan_theta.png")
    plt.close()
    
def wrap(theta):
    """
    
    """
    global state,opstate,sdir,psnr,Pn,pdmz,pNreps,ptauw,pwb,PATH,norm,OneDOnly
    global nz,ndm,zmax,dmmax
    opstate.path.Scale=theta
    res = get_likelihood(names,state,opstate,sdir,psnr,Pn,pdmz,pNreps,ptauw,pwb,PATH,norm,OneDOnly)
    return res
    
def get_likelihood(names,state,opstate,sdir=None, psnr=True, Pn=False,pdmz=True, pNreps=True,
                        ptauw=False, pwb=True, PATH=True, norm=True, OneDOnly=False):
    """
    gets a likelihood
    """
    
    surveys, grids = loading.surveys_and_grids(survey_names = names,repeaters=False,
                                                sdir=sdir,init_state=state,zmax=zmax,
                                                dmmax = dmmax, nz=nz, ndm=ndm)
    
    model = opt.loudas_model(opstate)
    final_ll = 0.
    for i,g in enumerate(grids):
        s=surveys[i]
        
        if PATH:
            wrapper = opt.model_wrapper(model,g.zvals)
            lltot,results = it.get_joint_path_zdm_likelihoods(g, s, wrapper, norm=norm, psnr=psnr, Pn=Pn,
                                    pdmz=pdmz, pNreps=pNreps, ptauw=ptauw, pwb=pwb,
                                    return_all=True)
            
            if OneDOnly:
                lltot = np.sum(np.log10(results["zdm_s"]["pxrad"])) # only 1D DM info
        else:
            # probably should include the FRBs with no known z. Oh well...
            lltot = it.calc_likelihoods_2D(g, s, psnr=psnr, Pn=Pn,pdmz=pdmz, pNreps=pNreps, ptauw=ptauw, pwb=pwb, norm=norm)
        
        final_ll += lltot
        #pdm = np.array(results['zdm_s']["pxrad"])
        #llpdm = np.sum(np.log10(pdm))
        #zdm_ll
    return final_ll
    
main()
