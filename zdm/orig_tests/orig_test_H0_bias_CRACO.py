"""
Runs likelihood(H0) for the files

CRAFT_CRACO_MC_alpha1_1000.dat
CRAFT_CRACO_MC_alpha1_1000_maxdm.dat
CRAFT_CRACO_MC_alpha1_1000_missing.dat

CRAFT_CRACO_MC_alpha1_gamma_1000.dat
CRAFT_CRACO_MC_alpha1_gamma_1000_maxdm.dat
CRAFT_CRACO_MC_alpha1_gamma_1000_missing.dat

and plots this is "Missing_z_Checks".

This tests whether or not missing DMs, or placing a max in DMEG,
introduces a bias. The result: no it does not.
"""
import pytest

from zdm import io
from zdm.MC_sample import loading
from pkg_resources import resource_filename
import os
import copy
import pickle
import numpy as np
from astropy.cosmology import Planck18

from zdm import cosmology as cos
from zdm import misc_functions
from zdm import parameters
from zdm import survey
from zdm import iteration as it

from IPython import embed
from matplotlib import pyplot as plt
import matplotlib

defaultsize=14
ds=4
font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : defaultsize}
matplotlib.rc('font', **font)

def make_grids(gamma):
    if gamma==0:
        raise ValueError("WARNING: gamma=0 is not currently implemented")
    ############## Load up ##############
    input_dict=io.process_jfile('../../papers/H0_I/Analysis/Cubes/craco_H0_Emax_cube.json')

    # Deconstruct the input_dict
    state_dict, cube_dict, vparam_dict = it.parse_input_dict(input_dict)
    
    surveys = []
    if (state_dict['energy']['luminosity_function'] == 1):
        names = ['CRAFT_CRACO_MC_alpha1_gamma_1000','CRAFT_CRACO_MC_alpha1_gamma_1000_maxdm','CRAFT_CRACO_MC_alpha1_gamma_1000_missing',]
        
        savefile="Missing_z_Checks/missing_ll_gamma_check.npy"
    else:
        names = ['CRAFT_CRACO_MC_alpha1_1000','CRAFT_CRACO_MC_alpha1_1000_maxdm','CRAFT_CRACO_MC_alpha1_1000_missing',]
        savefile="Missing_z_Checks/missing_ll_check.npy"
    
    ############## Initialise survey and grids ##############
    #NOTE: grid will be identical for all three, only bother to update one!
    surveys=[]
    s,grid = loading.survey_and_grid(
        state_dict=state_dict,
        survey_name=names[0],NFRB=1000) #, NFRB=Nsamples
    surveys.append(s)
    
    sdir = os.path.join(resource_filename('zdm', 'craco'), 'MC_Surveys')
    for i,survey_name in enumerate(names):
        if i==0:
            continue
        surveys.append(survey.load_survey(survey_name, grid.state, grid.dmvals,sdir))
    
    nH0=26
    lls=np.zeros([3,nH0])
    
    H0s=np.linspace(60,85,nH0)
    
    trueH0 = grid.state.cosmo.H0
    
    # Let's update H0 (barely) and find the constant for fun too as part of the test
    for ih,H0 in enumerate(H0s):
        
        vparams = {}
        vparams['H0'] = H0
        grid.update(vparams)
        
        lC,llC,lltot=it.minimise_const_only(
            vparams,[grid],[surveys[0]], Verbose=False,
            use_prev_grid=False)
    
        vparams = {}
        vparams['lC'] = lC
        grid.update(vparams)
        
        
        for i,s in enumerate(surveys):
            
            if s.nD==1:
                llsum=it.calc_likelihoods_1D(grid,s,psnr=True,dolist=0,Pn=True)
            elif s.nD==2:
                llsum=it.calc_likelihoods_2D(grid,s,psnr=True,dolist=0,Pn=True)
            elif s.nD==3:
                # mixture of 1 and 2D samples. NEVER calculate Pn twice!
                llsum1=it.calc_likelihoods_1D(grid,s,psnr=True,dolist=0,Pn=True)
                llsum2=it.calc_likelihoods_2D(grid,s,psnr=True,dolist=0,Pn=False)
                llsum = llsum1+llsum2
            lls[i,ih] = llsum
            
    np.save(savefile,lls)
    np.save("Missing_z_Checks/H0list.npy",H0s)

def plot_results(gamma):
    if (gamma == 1):
        lls=np.load("Missing_z_Checks/missing_ll_gamma_check.npy")
        savefile='Missing_z_Checks/lls_gamma.pdf'
    else:
        lls=np.load("Missing_z_Checks/missing_ll_check.npy")
        savefile='Missing_z_Checks/lls.pdf'
    
    H0s=np.load("Missing_z_Checks/H0list.npy")
    
    plt.figure()
    plt.xlabel('$H_0$ [km/s/Mpc]')
    plt.ylabel('$\\log_{10} \\ell(H_0)-\\log_{10} \\ell_{\\rm max}$')
    plt.ylim(-50,0)
    plt.xlim(60,85)
    labels=['All localised','${\\rm DM}_{\\rm EG}^{\\rm max}=1000$','$p({\\rm no}\, z)=\\frac{1}{3}$']
    styles=['-','--',':']
    for i in np.arange(3):
        peak=np.nanmax(lls[i,:])
        plt.plot(H0s[:],lls[i,:]-peak,label=labels[i],linestyle=styles[i],linewidth=3)
    trueH0=67.66
    plt.plot([trueH0,trueH0],[-50,0],linestyle='--',color='grey',label='True $H_0$')
    plt.legend()
    plt.tight_layout()
    plt.savefig(savefile)
    plt.close()

#use 0 for power law, 1 for gamma function
# have not implemented Xavier's version for gamma=0 yet...
for gamma in [1]:
    make_grids(gamma)
    plot_results(gamma)
