"""

This script is a test.

It simply loads in CHIME data and makes sure
everything can be read correctly.

It also demonstrates how to generate "repeat grids"
for a dataset.

"""
from importlib import resources

import numpy as np
from matplotlib import pyplot as plt
import os
import pickle

#import utilities as ute
#import states as st

from zdm.craco import loading
from zdm import repeat_grid as rep
from zdm import misc_functions
from zdm import beams

from IPython import embed

def load(Nbin:int=6, make_plots:bool=False):
    """
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
    
    opdir = 'CHIME/'
    if not os.path.exists(opdir):
        os.mkdir(opdir)
    
    # gets the possible states for evaluation
    #pset = st.james_fit()
    #state = st.set_state(pset)
    
    # loads survey data
    sdir = os.path.join(resources.files('zdm'),
                        'data', 'Surveys','CHIME/')
    
    # loads beam data
    bdir='Nbounds'+str(Nbin)+'/'
    beams.beams_path = os.path.join(resources.files('zdm'),'data',
                                    'BeamData', 'CHIME',bdir)
    #bounds = np.load(beams.beams_path+'bounds.npy')
    #solids = np.load(beams.beams_path+'solids.npy')
    
    
    # Loops through CHIME declination bins
    for ibin in np.arange(Nbin):
        name = "CHIME_decbin_"+str(ibin)+"_of_"+str(Nbin)
        s,g = loading.survey_and_grid(survey_name=name,NFRB=None,sdir=sdir)#,init_state=state)
        print("Loaded dec bin ",ibin)
        
        # The below is specific to CHIME data. For CRAFT and other
        # FRB surveys, do not use "bmethod=2", and you will have to
        # enter the time per field and Nfields manually.
        # Also, for other surveys, you do not need to iterate over
        # declination bins!
        rg = rep.repeat_Grid(g,Tfield=s.TOBS,Nfields=1,MC=False,opdir=None,bmethod=2)
        print("Loaded repeat grid")
        
        if ibin==0:
            all_rates = g.rates
            all_singles = rg.exact_singles
            all_reps = rg.exact_reps
        else:
            all_rates = all_rates + g.rates
            all_singles = all_singles + rg.exact_singles
            all_reps = all_reps + rg.exact_reps
        
    if make_plots:
        misc_functions.plot_grid_2(all_rates,g.zvals,g.dmvals,
                name=opdir+'all_CHIME_FRBs.pdf',norm=3,log=True,
                label='$\\log_{10} p({\\rm DM}_{\\rm EG},z)$ [a.u.]',
                project=False,Aconts=[0.01,0.1,0.5],
                zmax=3.0,DMmax=3000)
        
        misc_functions.plot_grid_2(all_reps,g.zvals,g.dmvals,
                name=opdir+'repeating_CHIME_FRBs.pdf',norm=3,log=True,
                label='$\\log_{10} p({\\rm DM}_{\\rm EG},z)$ [a.u.]',
                project=False,Aconts=[0.01,0.1,0.5],
                zmax=1.0,DMmax=1000)
        
        misc_functions.plot_grid_2(all_singles,g.zvals,g.dmvals,
                name=opdir+'single_CHIME_FRBs.pdf',norm=3,log=True,
                label='$\\log_{10} p({\\rm DM}_{\\rm EG},z)$ [a.u.]',
                project=False,Aconts=[0.01,0.1,0.5],
                zmax=3.0,DMmax=3000)

    return g.dmvals, g.zvals, all_rates, all_singles, all_reps
