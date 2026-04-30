""" 
This script uses CRAFT surveys to illustrate how likelihoods combine
using PATH and zDM simultaneously

We walk through the calculation, generating step-by-step plots,
to illustrate the method.

Output goes in "illustrations/"
"""
import os
import time

from astropy.cosmology import Planck18
from zdm import cosmology as cos
from zdm import figures
from zdm import parameters
from zdm import survey
from zdm import pcosmic
from zdm import iteration as it
from zdm import loading
from zdm import io
from zdm import optical as opt
from zdm import optical_params as op
from zdm import states

import numpy as np
from zdm import survey
from matplotlib import pyplot as plt
import importlib.resources as resources

from astropath import chance

import matplotlib

defaultsize=14
ds=4
font = {'family' : 'Helvetica',
        'weight' : 'normal',
        'size'   : defaultsize}
matplotlib.rc('font', **font)


def main():
    
    # p-values for confidence intervals
    pvalues=np.array([0.6827,0.90,0.9545,0.9973])
    
    # set op directory
    opdir="20210912A/"
    if not os.path.exists(opdir):
        os.mkdir(opdir)
    
    
    # set this to true to calculate p(twau,w). However, this takes quite some time to evaluate - it's slow!
    # That's because of internal arrays that must be built up by the survey
    
    # load states from Hoffman et al 2025
    state = states.load_state("HoffmannHalo25",scat="orig",rep=None)
    #state2 = states.load_state("HoffmannHalo25",scat="updated",rep=None)
    
    
    # add estimation from ptauw
    #state2.scat.Sbackproject=True
    
    if not os.path.exists(opdir):
        os.mkdir(opdir)
    
    # Initialise surveys and grids
    sdir = resources.files('zdm').joinpath('data/Surveys')
    names=['CRAFT_ICS_1300']
    
    
    # setting ptauw to True takes much longer!
    survey_dict = {}
    survey_dict["WMETHOD"] = 3
    
    ndm=1400
    nz=500
    
    # simple method - quick
    ss,gs = loading.surveys_and_grids(survey_names=names,repeaters=False,
                                    init_state=state,sdir=sdir,survey_dict = None,
                                    ndm=ndm, nz=nz) 
    ifrb=11
    zmin=1.31
    zmax=1.33
    zbar=(zmax+zmin)/2.
    dz = (zmax - zmin)
    
    g=gs[0]
    s=ss[0]
    # we now make a plot for the following situations
    
    # 1: p(z) given DM
    # 2: p(z) given unseen
    # 3: p(z) given m
    
    plt.figure()
    plt.xlim(0,2)
    ymax=2.5
    plt.ylim(0,ymax)
    plt.xlabel("$z$")
    plt.ylabel("$P(z)$")
    plt.xticks(np.linspace(0,2,5))
    
    #################3# adds Bera et al prediction ##############
    zmin=1.31
    zmax=1.33
    plt.fill_between([zmin,zmax],[ymax,ymax],color="gray",alpha=0.3)
    plt.text(1.25,1.8,"Bera et al.",rotation=90)
    
    
    ######## naive p(z) based on all FRBs #########
    DMEG = s.DMEGs[ifrb]
    idm1 = np.where(g.dmvals < DMEG)[0][-1]
    idm2 = idm1+1
    kdm2 = (DMEG-g.dmvals[idm1])/(g.dmvals[idm2]-g.dmvals[idm1])
    kdm1 = 1.-kdm2
    pz1 = g.rates[:,idm1]*kdm1 + g.rates[:,idm2]*kdm2
    pz1 /= np.sum(pz1) * (g.zvals[1]-g.zvals[0])
    
    plt.plot(g.zvals,pz1,label="$p(z|DM)$",linestyle=":")
    
    print("\n\nCurve 1: p   min    max")
    intervals = calc_confidence_intervals(g.zvals, pz1,pvalues)
    for i,p in enumerate(pvalues):
        print(p,intervals[i,0],intervals[i,1])
    
    fz = np.interp(zbar,g.zvals,pz1)
    p1 = fz*dz
    
    
    # specific p(z) based on these particular properties
    output = it.calc_likelihoods_1D(g, s, norm=True, psnr=True,dolist=0, Pn=False,
                                    ptauw=False,pwb=True, PATH=True)
    pz2 = output["pzgsnrbwdm"]
    print(pz2.shape)
    pz2 = pz2[:,ifrb]
    pz2 /= (g.zvals[1]-g.zvals[0])
    plt.plot(g.zvals,pz2,label="$p(z|{\\bf x}_{\\rm rad})$",linestyle="-.")
    
    print("\n\nCurve 2: p   min    max")
    intervals = calc_confidence_intervals(g.zvals, pz2,pvalues)
    for i,p in enumerate(pvalues):
        print(p,intervals[i,0],intervals[i,1])
    
    fz = np.interp(zbar,g.zvals,pz2)
    p2 = fz*dz
    
    
    """
    makes a plot of p(z) given mr
    
    Args:
        g: grid object
        s: survey object corresponding to the grid
        ifrb: which FRB in the survey to use
        surveys: list of strigns giving optical names
        means: array of means of P(O|m) of optical images
        widths: array of widths of P(O|m) of optical images
        
    """
    
    
    # these are actually the defaults
    surveys = ["VLT/FORS2"]#,"DESI LIS","PanSTARRS"]
    means = [26.2]#,24,21.8]
    stds = [0.34]#,0.55,0.54]
    
    opstate = op.OpticalState()
    opstate.id.pU_mean = means[0]
    opstate.id.pU_width = stds[0]
    model = opt.marnoch_model(opstate)
    wrapper = opt.model_wrapper(model,g.zvals)
    
    wrapper.init_path_raw_prior_Oi(DMEG,g,pz=pz2)
    
    pzgu = wrapper.get_pz_g_U()
    print(" p(U) is ",wrapper.estimate_unseen_prior())
        
    pzgu /= (g.zvals[1]-g.zvals[0])
    plt.plot(g.zvals, pzgu, label="$p(z|U,{\\bf x}_{\\rm rad})$",linestyle="--")
        
    print("\n\nCurve 3: p   min    max")
    intervals = calc_confidence_intervals(g.zvals, pzgu,pvalues)
    for i,p in enumerate(pvalues):
        print(p,intervals[i,0],intervals[i,1])
        
    fz = np.interp(zbar,g.zvals,pzgu)
    p3 = fz*dz
    
    ####### p(z) from measured FRB host galaxy magnitude ########
    mr = 29.1 
    pz = wrapper.get_pz_g_mr(mr)
    pz /= (g.zvals[1]-g.zvals[0])
    plt.plot(g.zvals, pz, label="$P(z|m=29.1,{\\bf x}_{\\rm rad})$")
    
    print("\n\nCurve 4: p   min    max")
    intervals = calc_confidence_intervals(g.zvals, pz,pvalues)
    for i,p in enumerate(pvalues):
        print(p,intervals[i,0],intervals[i,1])
    
    fz = np.interp(zbar,g.zvals,pz)
    p4 = fz*dz
    
    plt.legend()
    plt.tight_layout()
    plt.savefig(opdir+"pz_20210912A.png")
    plt.close()
    
    
    print("Respective probabilities are ",p1,p2,p3,p4)


def calc_confidence_intervals(x,y,intervals):
    """
    
    Args:
        x (np.ndarray): x values of function
        y (np.ndarray): y-values of function
        intervals (list of floasts): desired intervals (e.g. 0.9 for 90%)
    """
    
    # get cdf
    cy = np.cumsum(y)
    cy /= cy[-1]
    
    # get intervals - assumes two-sides
    nints = len(intervals)
    values = np.zeros([nints,2])
    
    for i,prob in enumerate(intervals):
        p1 = (1.-prob)/2.
        p2 = 1.-p1
        values[i,0]=p1
        values[i,1]=p2
    
    result = np.interp(values.flatten(),cy,x)
    result = result.reshape([nints,2])
    return result
    
    
    
main()
