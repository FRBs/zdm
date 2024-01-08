""" Load up the Real data """

# It should be possible to remove all the matplotlib calls from this
# but in the current implementation it is not removed.
import numpy as np
import os
from pkg_resources import resource_filename

from astropy.cosmology import Planck18

from zdm import survey
from zdm import parameters
from zdm import cosmology as cos
from zdm import misc_functions

from IPython import embed

def set_state(alpha_method=1, cosmo=Planck18):

    ############## Initialise parameters ##############
    state = parameters.State()

    # Variable parameters
    vparams = {}
    
    vparams['FRBdemo'] = {}
    vparams['FRBdemo']['alpha_method'] = alpha_method
    vparams['FRBdemo']['source_evolution'] = 0
    
    #vparams['beam'] = {}
    #vparams['beam']['Bthresh'] = 0
    #vparams['beam']['Bmethod'] = 2
    
    vparams['width'] = {}
    vparams['width']['Wlogmean'] = 1.70267
    vparams['width']['Wlogsigma'] = 0.899148
    vparams['width']['Wbins'] = 10
    vparams['width']['Wscale'] = 2
    vparams['width']['Wthresh'] = 0.5
    #vparams['width']['Wmethod'] = 2
    
    vparams['scat'] = {}
    vparams['scat']['Slogmean'] = 0.7
    vparams['scat']['Slogsigma'] = 1.9
    vparams['scat']['Sfnorm'] = 600
    vparams['scat']['Sfpower'] = -4.
    
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
        vparams['FRBdemo']['sfr_n'] = 1.67
        vparams['FRBdemo']['lC'] = 3.15
        vparams['host']['lmean'] = 2.11
        vparams['host']['lsigma'] = 0.53
    elif  vparams['FRBdemo']['alpha_method'] == 1:
        vparams['energy']['lEmin'] = 30
        vparams['energy']['lEmax'] = 41.4
        vparams['energy']['alpha'] = 0.65
        vparams['energy']['gamma'] = -1.01
        vparams['FRBdemo']['sfr_n'] = 0.73
        # NOTE: I have not checked what the best-fit value
        # of lC is for alpha method=1
        vparams['FRBdemo']['lC'] = 1 #not best fit, OK for a once-off
        
        vparams['host']['lmean'] = 2.18
        vparams['host']['lsigma'] = 0.48

    # Gamma
    vparams['energy']['luminosity_function'] = 2
        
    state.update_param_dict(vparams)
    state.set_astropy_cosmo(cosmo)

    # Return
    return state


def surveys_and_grids(init_state=None, alpha_method=1, 
                      survey_names=None,
                      add_20220610A=False,
                      nz:int=500, ndm:int=1400,
                      repeaters=False,
                      sdir=None, edir=None): 
    """ Load up a survey and grid for a real dataset

    Args:
        init_state (State, optional):
            Initial state
        survey_name (str, optional):  Defaults to 'CRAFT/CRACO_1_5000'.
        state_dict (dict, optional):
            Used to init state instead of alpha_method, lum_func parameters
        survey_names (list, optional):
            List of surveys to load
        add_20220610A (bool, optional):
            Include this FRB (a bit of a hack)
        nz (int, optional):
            Number of redshift bins
        ndm (int, optional):
            Number of DM bins
        edir (string, optional):
            Directory containing efficiency files if using FRB-specific responses

    Raises:
        IOError: [description]

    Returns:
        tuple: lists of Survey, Grid objects
    """
    # Init state
    if init_state is None:
        state = set_state(alpha_method=alpha_method)
    else:
        state = init_state

    # Cosmology
    cos.set_cosmology(state)
    cos.init_dist_measures()

    # get the grid of p(DM|z)
    zDMgrid, zvals,dmvals = misc_functions.get_zdm_grid(
        state, new=True, plot=False, method='analytic', 
        nz=nz, ndm=ndm,
        datdir=resource_filename('zdm', 'GridData'))
    
    ############## Initialise surveys ##############
    if survey_names is None:
        survey_names = ['CRAFT/FE', 
                    'CRAFT_ICS_1632',
                    'CRAFT_ICS_892', 
                    'CRAFT_ICS_1272',
                    'PKS/Mb']
    if add_20220610A:
        survey_names[3] = 'CRAFT_ICS_w_220610'

    surveys = []
    for survey_name in survey_names:
        # print(f"Initializing {survey_name}")
        s = survey.load_survey(survey_name, 
                               state, dmvals, 
                               sdir=sdir, edir=edir)
        # # Check necessary parameters exist if considering repeaters
        # if repeaters:
        #     s.init_repeaters()

        surveys.append(s)
        
    print("Initialised surveys")

    # generates zdm grid
    grids = misc_functions.initialise_grids(
        surveys, zDMgrid, zvals, dmvals, state, wdist=True, repeaters=repeaters)
    print("Initialised grids")

    # Return Survey and Grid
    return surveys, grids
 
if __name__ == '__main__':
    surveys, grids = surveys_and_grids()
