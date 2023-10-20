""" 
This script shows how to use repeating FRB grids.

It produces four outputs in the "Repeaters" directory,
showing zDM for:
- 1: All bursts (single bursts, and bursts from repeating sources)
- 2: FRBs expected as single bursts
- 3: Repeating FRBs (each source counts once)
- 4: Bursts from repeaters (each source counts Nburst times)

We expect 1 = 2+4 (if not, it's a bug!)

"""
import os
from pkg_resources import resource_filename
from zdm import cosmology as cos
from zdm import misc_functions
from zdm import parameters
from zdm import survey
from zdm import pcosmic
from zdm import iteration as it
from zdm.craco import loading
from zdm import io
from zdm import repeat_grid as rep

import numpy as np
from zdm import survey
from matplotlib import pyplot as plt

def main():
    
    # in case you wish to switch to another output directory
    opdir='Repeaters/'
    if not os.path.exists(opdir):
        os.mkdir(opdir)
    
    # standard 1.4 GHz CRAFT data
    name = 'CRAFT_ICS'
    
    state = parameters.State()
    
    # approximate best-fit values from recent analysis
    vparams = {}
    vparams['H0'] = 73
    vparams['lEmax'] = 41.3
    vparams['gamma'] = -0.95
    vparams['alpha'] = 1.03
    vparams['sfr_n'] = 1.15
    vparams['lmean'] = 2.23
    vparams['lsigma'] = 0.57
    vparams['lC'] = 1.963
    
    zvals=[]
    dmvals=[]
    grids=[]
    surveys=[]
    nozlist=[]
    sdir='../data/Surveys/'
    # use loading.survey_and_grid for proper estimates
    # remove loading for width-based estimates
    # the below is hard-coded for a *very* simplified analysis!
    # using loading. gives 5 beams and widths, ignoring that gives a single beam
    s,g = loading.survey_and_grid(survey_name=name,NFRB=None,sdir=sdir) # should be equal to actual number of FRBs, but for this purpose it doesn't matter
    
    # updates survey to have single beam value and weights
    
    # set up new parameters
    g.update(vparams)
    
    
    newC,llC=it.minimise_const_only(None,[g],[s])
    
    #Defines Nfield and Tobs per field as per average parameters (units are days)
    Tfield=20
    Nfields=1 #really more like 20. But these are example parameters now
    
    # adds repeating grid
    rg = rep.repeat_Grid(g,Tfield=Tfield,Nfields=1,MC=True,opdir='Repeaters/')
    
    ############# do 2D plots ##########
    misc_functions.plot_grid_2(g.rates,g.zvals,g.dmvals,
        name=opdir+name+'all_frbs.pdf',norm=3,log=True,label='$\\log_{10} p({\\rm DM}_{\\rm EG},z)$  [a.u.]',
        project=False,FRBDM=s.DMEGs,FRBZ=s.frbs["Z"],Aconts=[0.01,0.1,0.5],zmax=1.5,
        DMmax=1500)
    
    misc_functions.plot_grid_2(rg.exact_singles,g.zvals,g.dmvals,
        name=opdir+name+'single_frbs.pdf',norm=3,log=True,label='$\\log_{10} p({\\rm DM}_{\\rm EG},z)$  [a.u.]',
        project=False,FRBDM=s.DMEGs,FRBZ=s.frbs["Z"],Aconts=[0.01,0.1,0.5],zmax=1.5,
        DMmax=1500)
    
    misc_functions.plot_grid_2(rg.exact_reps,g.zvals,g.dmvals,
        name=opdir+name+'repeating_sources.pdf',norm=3,log=True,label='$\\log_{10} p({\\rm DM}_{\\rm EG},z)$  [a.u.]',
        project=False,FRBDM=s.DMEGs,FRBZ=s.frbs["Z"],Aconts=[0.01,0.1,0.5],zmax=1.5,
        DMmax=1500)
    
    misc_functions.plot_grid_2(rg.exact_rep_bursts,g.zvals,g.dmvals,
        name=opdir+name+'bursts_from_repeaters.pdf',norm=3,log=True,label='$\\log_{10} p({\\rm DM}_{\\rm EG},z)$  [a.u.]',
        project=False,FRBDM=s.DMEGs,FRBZ=s.frbs["Z"],Aconts=[0.01,0.1,0.5],zmax=1.5,
        DMmax=1500)

def survey_and_grid(survey_name:str='CRAFT/CRACO_1_5000',
            init_state=None,
            state_dict=None, iFRB:int=0,
               alpha_method=1, NFRB:int=100, 
               lum_func:int=0,sdir=None):
    """ Load up a survey and grid for a CRACO mock dataset

    Args:
        init_state (State, optional):
            Initial state
        survey_name (str, optional):  Defaults to 'CRAFT/CRACO_1_5000'.
        NFRB (int, optional): Number of FRBs to analyze. Defaults to 100.
        iFRB (int, optional): Starting index for the FRBs.  Defaults to 0
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
    if init_state is None:
        state = loading.set_state(alpha_method=alpha_method)
        # Addiitonal updates
        if state_dict is None:
            state_dict = dict(cosmo=dict(fix_Omega_b_h2=True))
            state.energy.luminosity_function = lum_func
        state.update_param_dict(state_dict)
    else:
        state = init_state
    
    # Cosmology
    cos.set_cosmology(state)
    cos.init_dist_measures()
    
    # get the grid of p(DM|z)
    zDMgrid, zvals,dmvals = misc_functions.get_zdm_grid(
        state, new=True, plot=False, method='analytic',
        datdir=resource_filename('zdm', 'GridData'),
        zlog=False,nz=990,zmax=9.9)

    ############## Initialise surveys ##############
    if sdir is not None:
        print("Searching for survey in directory ",sdir)
    else:
        sdir = os.path.join(resource_filename('zdm', 'craco'), 'MC_Surveys')
    isurvey = load_survey(survey_name, state, dmvals,
                                 NFRB=NFRB, sdir=sdir, Nbeams=5,
                                 iFRB=iFRB)
    
    # generates zdm grid
    grids = misc_functions.initialise_grids(
        [isurvey], zDMgrid, zvals, dmvals, state, wdist=True)
    print("Initialised grid")

    # Return Survey and Grid
    return isurvey, grids[0]


def load_survey(survey_name:str, state:parameters.State, dmvals:np.ndarray,
                sdir:str=None, NFRB:int=None, Nbeams=None, iFRB:int=0):
    """Load a survey

    Args:
        survey_name (str): Name of the survey
            e.g. CRAFT/FE
        state (parameters.State): Parameters for the state
        dmvals (np.ndarray): DM values
        sdir (str, optional): Path to survey files. Defaults to None.
        NFRB (int, optional): Cut the total survey down to a random
            subset [useful for testing]
        iFRB (int, optional): Start grabbing FRBs at this index
            Mainly used for Monte Carlo analysis
            Requires that NFRB be set

    Raises:
        IOError: [description]

    Returns:
        Survey: instance of the class
    """
    print(f"Loading survey: {survey_name}")
    if sdir is None:
        sdir = os.path.join(resource_filename('zdm', 'data'), 'Surveys')

    Nbeams=1
    dfile = 'parkes_mb_class_I_and_II.dat'
    # Do it
    srvy=survey.Survey()
    srvy.name = survey_name
    srvy.process_survey_file(os.path.join(sdir, dfile), NFRB=NFRB, iFRB=iFRB)
    srvy.init_DMEG(state.MW.DMhalo)
    #srvy.init_beam(nbins=1, method=state.beam.Bmethod, plot=False,
    #            thresh=0.9) # tells the survey to use the beam file
    srvy.NBEAMS=1
    srvy.beam_b=np.array([1.])
    srvy.beam_o=np.array([1.])
    #pwidths,pprobs=make_widths(srvy,state)
    pwidths=np.array([1])
    pprobs=np.array([1])
    _ = srvy.get_efficiency_from_wlist(dmvals,pwidths,pprobs)

    return srvy

main()
