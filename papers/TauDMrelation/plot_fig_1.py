""" 
This script creates zdm grids and plots localised FRBs

It can also generate a summed histogram from all CRAFT data

"""
import os

from astropy.cosmology import Planck18
from zdm.zdm import cosmology as cos
from zdm.zdm import figures
from zdm.zdm import parameters
from zdm.zdm import survey
from zdm.zdm import pcosmic
from zdm.zdm import iteration as it
from zdm.zdm import loading
from zdm.zdm import io

import numpy as np
from zdm.zdm import survey
from matplotlib import pyplot as plt
from pkg_resources import resource_filename
import time
import json

def main():

    # in case you wish to switch to another output directory
    opdir = "./"
    if not os.path.exists(opdir):
        os.mkdir(opdir)
    
    # Initialise surveys and grids
    sdir = "/Users/lmasriba/FRBs/zdm/zdm/data/Surveys"
    names = ["CRAFT_ICS_1300"]
    
    # loads state variables
    #with open('state.json') as json_file:
    #    oldstate = json.load(json_file)
      
    
    # essentially turns off DM host and sets all FRB widths to ~0 (or close enough)
    #state_dict = {'lmean': 0.01, 'lsigma': 0.4, 'Wlogmean': -1,'WNbins': 1,
    #    'Wlogsigma': 0.1, 'Slogmean': -2,'Slogsigma': 0.1,'H0': 70,'logF': -0.495,
    #    'alpha': 0.65,'sfr_n': 0.73,'lEmax': 41.4,'lEmin': 30.,'gamma': -1.01,
    #    'alpha_method': 1}
    #state_dict = {'lmean': 0.01, 'lsigma': 0.4, 'Wlogmean': -1,'WNbins': 1,
    #    'Wlogsigma': 0.1, 'Slogmean': -2,'Slogsigma': 0.1}
    #state_dict={'sfr_n': 0.21, 'alpha': 0.11, 'lmean': 0.01, 'lsigma': 0.42, 'lEmax': 41.37, 'Wlogsigma': 0.1, 'Slogmean': -2,'Slogsigma': 0.1, 
    #            'lEmin': 39.47, 'gamma': -1.04, 'H0': 70.23, 'halo_method': 0, 'sigmaDMG': 0.0, 'sigmaHalo': 0.0,
    #            'lC': -7.61, 'min_lat': 0.0}
    state_dict={'sfr_n': 0.21, 'alpha': 0.11, 'lmean': 2.18, 'lsigma': 0.42, 'lEmax': 41.37, 
                'lEmin': 39.47, 'gamma': -1.04, 'H0': 70.23, 'halo_method': 0, 'sigmaDMG': 0.0, 'sigmaHalo': 0.0,
                'lC': -7.61, 'min_lat': 0.0}
    state = parameters.State()
    state.set_astropy_cosmo(Planck18)
    #state.update_params(state_dict)


    
    
    surveys, grids = loading.surveys_and_grids(survey_names = names,
        repeaters=False, sdir=sdir,init_state=state)
    
    # plots it
    g=grids[0]
    
    figures.plot_grid(g.rates,g.zvals,g.dmvals,
            name='figure1.png',norm=3,log=True,
            label='$\\log_{10} p({\\rm DM}_{\\rm cosmic},z)$ [a.u.]',
            ylabel='${\\rm DM}_{\\rm cosmic}$',
            project=False,
            logrange=5,
            zmax=2.5,DMmax=2500,cmap="Oranges",Aconts=[0.01,0.1,0.5],
            cont_clrs=[0.,0.,0.])

main()
