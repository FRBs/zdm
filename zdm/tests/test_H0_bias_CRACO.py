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

def make_grids(gamma):

    ############## Initialise parameters ##############
    state = parameters.State()

    
    # Variable parameters
    vparams = {}
    vparams['cosmo'] = {}
    vparams['cosmo']['H0'] = 67.74
    vparams['cosmo']['Omega_lambda'] = 0.685
    vparams['cosmo']['Omega_m'] = 0.315
    vparams['cosmo']['Omega_b'] = 0.044
    
    vparams['FRBdemo'] = {}
    vparams['FRBdemo']['alpha_method'] = 1 #or set to zero for spectral index interpretation
    vparams['FRBdemo']['source_evolution'] = 0
    
    vparams['beam'] = {}
    vparams['beam']['thresh'] = 0
    vparams['beam']['method'] = 2
    
    vparams['width'] = {}
    vparams['width']['logmean'] = 1.70267
    vparams['width']['logsigma'] = 0.899148
    vparams['width']['Wbins'] = 10
    vparams['width']['Wscale'] = 2
    
     # constants of intrinsic width distribution
    vparams['MW']={}
    vparams['MW']['DMhalo']=50
    
    vparams['host']={}
    vparams['energy'] = {}
    
    if vparams['FRBdemo']['alpha_method'] == 0:
        vparams['energy']['lEmin'] = 30
        vparams['energy']['lEmax'] = 41.7
        vparams['energy']['alpha'] = 1.55
        vparams['energy']['gamma'] = -1.09
        vparams['energy']['luminosity_function'] = gamma
        vparams['FRBdemo']['sfr_n'] = 1.67
        vparams['FRBdemo']['lC'] = 3.15
        vparams['host']['lmean'] = 2.11
        vparams['host']['lsigma'] = 0.53
    elif  vparams['FRBdemo']['alpha_method'] == 1:
        vparams['energy']['lEmin'] = 30
        vparams['energy']['lEmax'] = 41.4
        vparams['energy']['alpha'] = 0.65
        vparams['energy']['gamma'] = -1.01
        vparams['energy']['luminosity_function'] = gamma
        vparams['FRBdemo']['sfr_n'] = 0.73
        vparams['FRBdemo']['lC'] = 1 #not best fit, OK for a once-off
        
        vparams['host']['lmean'] = 2.18
        vparams['host']['lsigma'] = 0.48
    
    # Update state
    state.update_param_dict(vparams)

    ############## Initialise cosmology ##############
    cos.set_cosmology(state)
    cos.init_dist_measures()

    # get the grid of p(DM|z). See function for default values.
    # set new to False once this is already initialised
    zDMgrid, zvals,dmvals = misc_functions.get_zdm_grid(
        state, new=True, plot=False, method='analytic')
    
    # generates zdm grids for the specified parameter set
    if state.beam.method == 3: #'Full':
        gprefix='best'
    elif state.beam.method == 2: #'Std':
        gprefix='Std_best'
    else:
        raise ValueError("Bad beam method!")
    
    surveys = []
    if (vparams['energy']['luminosity_function'] == 1):
        names = ['CRAFT_CRACO_MC_alpha1_gamma_1000','CRAFT_CRACO_MC_alpha1_gamma_1000_maxdm','CRAFT_CRACO_MC_alpha1_gamma_1000_missing',]
        gprefix = gprefix+'_gamma_'
        savefile="Missing_z_Checks/missing_ll_gamma_check.npy"
    else:
        names = ['CRAFT_CRACO_MC_alpha1_1000','CRAFT_CRACO_MC_alpha1_1000_maxdm','CRAFT_CRACO_MC_alpha1_1000_missing',]
        savefile="Missing_z_Checks/missing_ll_check.npy"
        
    for survey_name in names:
        surveys.append(survey.load_survey(survey_name, state, dmvals))
    
    # we take advantage of knowing that each survey has an identical grid, so only generate one
    #if state.analysis.NewGrids:
    pklfile='Pickle/'+gprefix+'grids.pkl'
    if os.path.exists(pklfile):
        print("Loading grid ",pklfile)
        with open(pklfile, 'rb') as infile:
            grids=pickle.load(infile)
    else:
        print("Generating new grids, set NewGrids=False to save time later")
        grids=misc_functions.initialise_grids(
            [surveys[0]],zDMgrid, zvals, dmvals, state, wdist=True)#, source_evolution=source_evolution, alpha_method=alpha_method)
        with open(pklfile, 'wb') as output:
            pickle.dump(grids, output, pickle.HIGHEST_PROTOCOL)
    
    grid=grids[0]
    nH0=26
    lls=np.zeros([3,nH0])
    
    H0s=np.linspace(60,85,nH0)
    # vparams = None
    # I have checked that the following yield the same result of lC = 2.53089...
    #   vparams = {}
    #       vparams['lC']=0.1
    #   vparams = {}
    #       vparams['lC']=3
    #   vparams = None
    
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
    plt.xlabel('H_0')
    plt.ylabel('$\\log_{10} \\ell(H_0)$')
    labels=['CRACO 1000','max DMEG 1000','missing 1 in 3']
    for i in np.arange(3):
        peak=np.nanmax(lls[i,:])
        plt.plot(H0s[:],lls[i,:]-peak,label=labels[i])
    plt.legend()
    plt.tight_layout()
    plt.savefig(savefile)
    plt.close()

gamma=1 #use 0 for power law, 1 for gamma function
for gamma in 0,1:
    make_grids(gamma)
    plot_results(gamma)
