
"""
This generates 1000 MC samples from CRACO.

It's written slightly differently from the standard mc_statistics.py
to make it easier to iterate through different surveys, i.e.
it processes the surveys one at a time.

The main output is the print to screen. This gives the (default 1000)
FRBs and measured redshifts. This has been copied to
CRAFT_CRACO_MC_alpha1_1000.dat (power law)
CRAFT_CRACO_MC_alpha1_gamma_1000.dat (gamma function)

It then also removes the redshift info for all FRBs with DMEG > 1000
and sets z=-1 (z < 0 is code for "unlocalised"), and re-prints the
same list. See
CRAFT_CRACO_MC_alpha1_1000.dat
CRAFT_CRACO_MC_alpha1_gamma_1000.dat

Finally, it also takes one third of all FRBs (irrespective of DM)
and *also* sets z=-1 to simulate FRBs that have not yet been
localised. See
CRAFT_CRACO_MC_alpha1_1000_missing.dat
CRAFT_CRACO_MC_alpha1_gamma_1000_missing.dat


"""


from pkg_resources import resource_filename
import os
import copy
import pickle
import sys
import scipy as sp
import numpy as np
import time
import argparse

from astropy.cosmology import Planck18

from zdm import cosmology as cos
from zdm import misc_functions
from zdm import parameters
from zdm import survey
from zdm import iteration as it
from zdm import beams
from zdm import misc_functions

from IPython import embed

import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm
import matplotlib

from zdm import io

from matplotlib.ticker import NullFormatter

matplotlib.rcParams['image.interpolation'] = None

defaultsize=14
ds=4
font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : defaultsize}
matplotlib.rc('font', **font)

def main(N=100,plots=False):
    "created N*survey.NFRBs mock FRBs for all surveys"
    "Input N : creates N*survey.NFRBs"
    "Output : Sample(list) and Surveys(list)"
    
    ############## Load up ##############
    input_dict=io.process_jfile('../../papers/H0_I/Analysis/Cubes/craco_H0_Emax_cube.json')

    # Deconstruct the input_dict
    state, cube_dict, vparam_dict = it.parse_input_dict(input_dict)
    
    ######## choose which surveys to do an MC for #######
    
    # choose which survey to create an MC for
    names = ['CRAFT/FE', 'CRAFT/ICS', 'CRAFT/ICS892', 'PKS/Mb','CRAFT_CRACO_MC_alpha1_gamma_1000']
    dirnames=['ASKAP_FE','ASKAP_ICS','ASKAP_ICS892','Parkes_Mb','CRACO']
    # select which to perform an MC for
    which=1
    N=10000
    for DMG in [0,100,200,500]:
        do_mc(state,names[which],N,which,DMG)
    
def do_mc(state_dict, survey_name, Nsamples, which_survey,DMG):
 
    ############## Initialise survey and grid ##############
    s,g = survey_and_grid(
        state_dict=state_dict,
        survey_name=survey_name, NFRB=9,DMG=DMG)
          
    ############## Initialise surveys ##############
    #saves everything in this directory
    outdir='DMGtests/'
    if not os.path.isdir(outdir):
        os.mkdir(outdir)
    
    
    samples = []
    
    name = str(which_survey)
    savefile=outdir+'mc_sample_'+name+'_alpha_'+str(g.state.FRBdemo.alpha_method)+str(Nsamples)+'_DMG'+str(DMG)+'.npy'
    
    try:
        sample=np.load(savefile)
        print("Loading ",sample.shape[0]," samples from file ",savefile)
    except:
        print("Generating ",Nsamples," samples from survey/grid ",which_survey)
        sample=g.GenMCSample(Nsamples)
        sample=np.array(sample)
        np.save(savefile,sample)
    print("Shape of samples is ",samples)
    samples.append(sample)
    
    print("########### Set 1: complete ########")
    print("KEY ID 	DM    	DMG     DMEG	Z	SNR	WIDTH")
    for j in np.arange(Nsamples):
        DMEG=sample[j,1]
        DMtot=DMEG+DMG+g.state.MW.DMhalo
        SNRTHRESH=9.5
        SNR=SNRTHRESH*sample[j,4]
        z=sample[j,0]
        w=sample[j,3]
        
        string="FRB "+str(j)+'  {:6.1f}  {:6.1f}   {:6.1f}  {:5.3f}   {:5.1f}  {:5.1f}'.format(DMtot,DMG,DMEG,z,SNR,w)
        #print("FRB ",i,DMtot,SNR,DMEG,w)
        print (string)
    
def survey_and_grid(survey_name:str='CRAFT/CRACO_1_5000',
            state_dict=None,
               alpha_method=1, NFRB:int=100, lum_func:int=0,sdir=None,DMG=None):
    """ Load up a survey and grid for a CRACO mock dataset

    Args:
        cosmo (str, optional): astropy cosmology. Defaults to 'Planck15'.
        survey_name (str, optional):  Defaults to 'CRAFT/CRACO_1_5000'.
        NFRB (int, optional): Number of FRBs to analyze. Defaults to 100.
        lum_func (int, optional): Flag for the luminosity function. 
            0=power-law, 1=gamma.  Defaults to 0.
        state_dict (dict, optional):
            Used to init state instead of alpha_method, lum_func parameters

    Raises:
        IOError: [description]

    Returns:
        tuple: Survey, Grid objects
    """
    # Init state
    from zdm.craco import loading
    state = loading.set_state(alpha_method=alpha_method)

    # Addiitonal updates
    if state_dict is None:
        state_dict = dict(cosmo=dict(fix_Omega_b_h2=True))
        state.energy.luminosity_function = lum_func
    state.update_param_dict(state_dict)
    
    # Cosmology
    cos.set_cosmology(state)
    cos.init_dist_measures()
    
    # get the grid of p(DM|z)
    zDMgrid, zvals,dmvals = misc_functions.get_zdm_grid(
        state, new=True, plot=False, method='analytic',
        datdir=resource_filename('zdm', 'GridData'))
    
    ############## Initialise surveys ##############
    if sdir is not None:
        print("Searching for survey in directory ",sdir)
    else:
        sdir = os.path.join(resource_filename('zdm', 'craco'), 'MC_Surveys')
    isurvey = survey.load_survey(survey_name, state, dmvals,
                                 NFRB=NFRB, sdir=sdir, Nbeams=5)
    # reset survey Galactic DM values
    if DMG is not None:
        pwidths,pprobs=survey.make_widths(isurvey, 
                                    state.width.Wlogmean,
                                    state.width.Wlogsigma,
                                    state.beam.Wbins,
                                    scale=state.beam.Wscale)
        isurvey.DMGs[:]=DMG
        isurvey.init_DMEG(state.MW.DMhalo)
        _ = isurvey.get_efficiency_from_wlist(dmvals,pwidths,pprobs)
        
    # generates zdm grid
    grids = misc_functions.initialise_grids(
        [isurvey], zDMgrid, zvals, dmvals, state, wdist=True)
    print("Initialised grid")

    # Return Survey and Grid
    return isurvey, grids[0]

    
main()
