"""

This script is a test.

It simply loads in CHIME data and makes sure
everything can be read correctly.

It also demonstrates how to generate "repeat grids"
for a dataset.

"""
from pkg_resources import resource_filename

import numpy as np
from matplotlib import pyplot as plt
import os
import pickle

#import utilities as ute
#import states as st

from zdm.craco import loading
from zdm import cosmology as cos
from zdm import repeat_grid as rep
from zdm import misc_functions
from zdm import survey
from zdm import beams

def main(Nbin=6):
    """
    Nbins is the number of declination bins to use
    """
    
    opdir = 'CHIME/'
    if not os.path.exists(opdir):
        os.mkdir(opdir)
    
    # gets the possible states for evaluation
    #pset = st.james_fit()
    #state = st.set_state(pset)
    
    # loads survey data
    sdir = os.path.join(resource_filename('zdm','data/Surveys/'),'CHIME/')
    
    # loads beam data
    bdir='Nbounds'+str(Nbin)+'/'
    beams.beams_path = os.path.join(resource_filename('zdm','data/BeamData/CHIME/'),bdir)
    bounds = np.load(beams.beams_path+'bounds.npy')
    solids = np.load(beams.beams_path+'solids.npy')
    
    
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

# Nbin = 30 is also implemented
main(Nbin=6)
