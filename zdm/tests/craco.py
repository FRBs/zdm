""" Run tests with CRACO FRBs """

######
# first run this to generate surveys and parameter sets, by 
# setting NewSurveys=True NewGrids=True
# Then set these to False and run with command line arguments
# to generate *many* outputs
#####

# It should be possible to remove all the matplotlib calls from this
# but in the current implementation it is not removed.
import argparse
import numpy as np
import os
from pkg_resources import resource_filename
import matplotlib
from matplotlib import pyplot as plt

from astropy.cosmology import Planck15, Planck18

from zdm import survey
from zdm import parameters
from zdm import cosmology as cos
from zdm import misc_functions
from zdm import iteration as it
from zdm import io

from IPython import embed

import pickle


def load_craco(cosmo='Planck15', 
               survey_name='CRAFT/CRACO_1_5000',
               NFRB=100, lum_func=0):

    #psetmins,psetmaxes,nvals=misc_functions.process_pfile(Cube[2])
    #input_dict= io.process_jfile(Cube[2])
    #state_dict, cube_dict, vparam_dict = it.parse_input_dict(input_dict)

    state_dict = dict(FRBdemo=dict(alpha_method=1),
                      cosmo=dict(fix_Omega_b_h2=True))

    ############## Initialise parameters ##############
    state = parameters.State()

    # Clancy values
    state.energy.lEmax=41.4 
    state.energy.alpha=0.65 
    state.energy.gamma=-1.01 
    state.FRBdemo.sfr_n=0.73 
    state.host.lmean=2.18
    state.host.lsigma=0.48

    state.energy.luminosity_function = lum_func

    state.update_param_dict(state_dict)
    
    ############## Initialise cosmology ##############
    if cosmo == 'Planck18':
        state.set_astropy_cosmo(Planck18)
    elif cosmo == 'Planck15':
        state.set_astropy_cosmo(Planck15)
    else:
        raise IOError("Bad astropy cosmology")
        
    cos.set_cosmology(state)
    cos.init_dist_measures()
    
    # get the grid of p(DM|z)
    zDMgrid, zvals,dmvals = misc_functions.get_zdm_grid(
        state, new=True, plot=False, method='analytic',
        datdir=resource_filename('zdm', 'GridData'))
    
    ############## Initialise surveys ##############
    
    
    isurvey = survey.load_survey(survey_name, state, dmvals,
                                 NFRB=NFRB)

    # generates zdm grid
    grids = misc_functions.initialise_grids(
        [isurvey], zDMgrid, zvals, dmvals, state, wdist=True)
    print("Initialised grid")

    # Return
    return isurvey, grids[0]
