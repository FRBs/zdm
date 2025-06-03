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
from zdm import figures

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
    vparams['width']['WNbins'] = 10
    #vparams['width']['Wscale'] = 2
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


def load_CHIME(Nbin:int=6, make_plots:bool=False, opdir='CHIME/',\
                Verbose=False,state=None):
    """
    Loads CHIME grids
    Nbins is the number of declination bins to use

    Args:
        Nbin (int, optional): Number of declination bins to use. Defaults to 6.
           30 is allowed too
        make_plots (bool, optional): Whether to make plots. Defaults to False.

    Returns:
        tuple: 
            dmvals (np.ndarray): 1D array of DM values
            zvals (np.ndarray): 1D array of redshift values
            all_rates (np.ndarray): 2D array of rates
            all_singles (np.ndarray): 2D array of single FRB rates
            all_reps (np.ndarray): 2D array of repeating FRB rates
        
    """
    
    if not os.path.exists(opdir) and make_plots:
        os.mkdir(opdir)
    
    # gets the possible states for evaluation
    #pset = st.james_fit()
    #state = st.set_state(pset)
    
    # loads survey data
    sdir = resource_filename('zdm','data/Surveys/CHIME/')
    if Verbose:
        print("Loading CHIME surveys from ",sdir)
    
    # loads beam data
    names=[]
    # Loops through CHIME declination bins
    for ibin in np.arange(Nbin):
        name = "CHIME_decbin_"+str(ibin)+"_of_"+str(Nbin)
        names.append(name)
    
    # loads grids
    ss,rgs = surveys_and_grids(survey_names=names,sdir=sdir,
        init_state=state,repeaters=True)#,init_state=state)
    
    # sums outputs
    for ibin in np.arange(Nbin):
        s = ss[ibin]
        rg = rgs[ibin]
        if Verbose:
            print("Loaded dec bin ",ibin)
        
        # The below is specific to CHIME data. For CRAFT and other
        # FRB surveys, do not use "bmethod=2", and you will have to
        # enter the time per field and Nfields manually.
        # Also, for other surveys, you do not need to iterate over
        # declination bins!
        # rg = rep.repeat_Grid(g,Tfield=s.TOBS,Nfields=1,MC=False,opdir=None,bmethod=2)
        if Verbose:
            print("Loaded repeat grid")
        
        if ibin==0:
            all_rates = rg.rates
            all_singles = rg.exact_singles
            all_reps = rg.exact_reps
        else:
            all_rates = all_rates + rg.rates
            all_singles = all_singles + rg.exact_singles
            all_reps = all_reps + rg.exact_reps
        
    if make_plots:
        figures.plot_grid(all_rates,rg.zvals,rg.dmvals,
                name=opdir+'all_CHIME_FRBs.pdf',norm=3,log=True,
                label='$\\log_{10} p({\\rm DM}_{\\rm EG},z)$ [a.u.]',
                project=False,Aconts=[0.01,0.1,0.5],
                zmax=3.0,DMmax=3000)
        
        figures.plot_grid(all_reps,rg.zvals,rg.dmvals,
                name=opdir+'repeating_CHIME_FRBs.pdf',norm=3,log=True,
                label='$\\log_{10} p({\\rm DM}_{\\rm EG},z)$ [a.u.]',
                project=False,Aconts=[0.01,0.1,0.5],
                zmax=1.0,DMmax=1000)
        
        figures.plot_grid(all_singles,rg.zvals,rg.dmvals,
                name=opdir+'single_CHIME_FRBs.pdf',norm=3,log=True,
                label='$\\log_{10} p({\\rm DM}_{\\rm EG},z)$ [a.u.]',
                project=False,Aconts=[0.01,0.1,0.5],
                zmax=3.0,DMmax=3000)

    return ss,rgs,all_rates, all_singles, all_reps

def surveys_and_grids(init_state=None, alpha_method=1, 
                      survey_names=None,
                      nz:int=500, ndm:int=1400,
                      NFRB=None, repeaters=False,
                      sdir=None, edir=None,
                      rand_DMG=False, discard_empty=False,
                      state_dict=None,
                      survey_dict=None,verbose=False): 
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
        rand_DMG (bool, optional):
            If true, randomise the galactic DM - for MCMC studies
        discard_empty (bool, optional):
            If true, does not calculate empty surveys (mostly for after latitude cuts)
        survey_dict (dict,None): list of survey metadata and values to apply
    Raises:
        IOError: [description]

    Returns:
        tuple: lists of Survey, Grid objects
    """
    # Init state
    if init_state is None:
        # we should be using defaults... by default!
        # if a user wants something else, they can pass it here
        #state = set_state(alpha_method=alpha_method)
        state = parameters.State()
    else:
        state = init_state
    if state_dict is not None:
        state.update_param_dict(state_dict)
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
                    'CRAFT_ICS_1300',
                    'PKS/Mb']

    surveys = []
    for survey_name in survey_names:
        # print(f"Initializing {survey_name}")
        s = survey.load_survey(survey_name, state, dmvals, zvals,
                               NFRB=NFRB, sdir=sdir, edir=edir, 
                               rand_DMG=rand_DMG,survey_dict=survey_dict)
        
        if discard_empty == False or s.NFRB != 0:
            # Check necessary parameters exist if considering repeaters
            if repeaters:
                s.init_repeaters()

            surveys.append(s)
        else:
            print("Skipping empty survey " + s.name)
        
    if verbose:
        print("Initialised surveys")

    # generates zdm grid
    grids = misc_functions.initialise_grids(
        surveys, zDMgrid, zvals, dmvals, state, wdist=True, repeaters=repeaters)
    if verbose:
        print("Initialised grids")

    # Return Survey and Grid
    return surveys, grids
 
if __name__ == '__main__':
    surveys, grids = surveys_and_grids()
