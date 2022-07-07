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
from zdm.craco import loading
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

def make_grids(luminosity_function=2,verbose=True):
    
    opdir="FigureA2/"
    savefile=opdir+"bias_ll_g1a1_check_10k.npy"
    if os.path.exists(savefile):
        print("Save file detecting, going directly to plotting...")
        return
    else:
        print("Evaluating likelihoods for three sets of SNR cuts")
        print("This may take O~10 minutes")
        
    if not os.path.exists(opdir):
        os.mkdir(opdir)
    
    if luminosity_function != 2:
        raise ValueError("WARNING: luminosity functions other than 2 not currently implemented")
    
    ############## Load up ##############
    input_dict=io.process_jfile('../Analysis/CRACO/Cubes/craco_H0_Emax_cube.json')
    if verbose:
        print("Loading json file")
    
    # Deconstruct the input_dict
    state_dict, cube_dict, vparam_dict = it.parse_input_dict(input_dict)
    
    state_dict['energy']['luminosity_function'] = luminosity_function
    
    surveys = []
    names = ['CRAFT_ICS_10000_g1a1_all','CRAFT_ICS_10000_g1a1_half','CRAFT_ICS_10000_g1a1_14']
    
    
    ############## Initialise survey and grids ##############
    #NOTE: grid will be identical for all three, only bother to update one!
    surveys=[]
    grids=[]
    nfrbs=[10000,7694,5388]
    for i,name in enumerate(names):
        s,g = loading.survey_and_grid(
            state_dict=state_dict,
            survey_name=names[i],NFRB=nfrbs[i]) #, NFRB=Nsamples
        surveys.append(s)
        grids.append(g)
    
    nH0=26
    lls=np.zeros([3,nH0])
    
    H0s=np.linspace(65,70,nH0)
    
    trueH0 = grids[0].state.cosmo.H0
    
    # Let's update H0 (barely) and find the constant for fun too as part of the test
    for ih,H0 in enumerate(H0s):
        
        vparams = {}
        vparams['H0'] = H0
        
        for i,s in enumerate(surveys):
            
            grids[i].update(vparams)
        
            if s.nD==1:
                llsum=it.calc_likelihoods_1D(grids[i],s,psnr=True,dolist=0,Pn=False)
            elif s.nD==2:
                llsum=it.calc_likelihoods_2D(grids[i],s,psnr=True,dolist=0,Pn=False)
            elif s.nD==3:
                # mixture of 1 and 2D samples. NEVER calculate Pn twice!
                llsum1=it.calc_likelihoods_1D(grids[i],s,psnr=True,dolist=0,Pn=False)
                llsum2=it.calc_likelihoods_2D(grids[i],s,psnr=True,dolist=0,Pn=False)
                llsum = llsum1+llsum2
            lls[i,ih] = llsum
            
    np.save(savefile,lls)
    np.save(opdir+"v2H0list.npy",H0s)

def plot_results(luminosity_function=2,opdir="FigureA2/"):
    
    lls=np.load(opdir+"bias_ll_g1a1_check_10k.npy")
    savefile=opdir+"bias_ll_g1a1_check_10k.pdf"
    
    H0s=np.load(opdir+"v2H0list.npy")
    
    plt.figure()
    plt.xlabel('$H_0$ [km/s/Mpc]')
    plt.ylabel('$\\log_{10} \\ell(H_0)-\\log_{10} \\ell_{\\rm max}$')
    plt.ylim(-5,0)
    plt.xlim(65,70)
    labels=['All SNR>9','Half SNR 9-14','SNR>14']
    styles=['-','--',':']
    for i in np.arange(3):
        peak=np.nanmax(lls[i,:])
        plt.plot(H0s[:],lls[i,:]-peak,label=labels[i],linestyle=styles[i],linewidth=3)
    trueH0=67.66
    plt.plot([trueH0,trueH0],[-50,0],linestyle='--',color='grey',label='True $H_0$')
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(savefile)
    plt.close()

# only luminosity function of 2 (upper incomplete gamma function)
# this is historical
luminosity_function=2
make_grids(luminosity_function)
plot_results(luminosity_function)
