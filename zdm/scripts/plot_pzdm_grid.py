""" 
This script creates p(zdm grids) and plots localised FRBs

It can also generate a summed histogram from all CRAFT data

"""
import os

from zdm import cosmology as cos
from zdm import misc_functions
from zdm import parameters
from zdm import survey
from zdm import pcosmic
from zdm import iteration as it
from zdm import loading
from zdm import io
from pkg_resources import resource_filename
import numpy as np
from zdm import survey
from matplotlib import pyplot as plt

import time

def main():

    # in case you wish to switch to another output directory
    opdir = "PZDM/"
    if not os.path.exists(opdir):
        os.mkdir(opdir)

    # Initialise surveys and grids
    # write one or more FRB survey names here

    # The below is for private, unpublished FRBs. You will NOT see this in the repository!
    # names = ['private_CRAFT_ICS','private_CRAFT_ICS_892','private_CRAFT_ICS_1632']
    
    # Public CRAFT FRBs
    # names = ["CRAFT_ICS_1300", "CRAFT_ICS_892", "CRAFT_ICS_1632"]

    # Examples for other FRB surveys
    # names = ["FAST", "Arecibo", "parkes_mb_class_I_and_II"]
    
    # directory where the survey files are located. The below is the default - you can leave this out
    sdir = os.path.join(resource_filename('zdm', 'data'), 'Surveys')
    
    names = ["CRAFT_ICS_1300"]
    
    # Write True if you want to do repeater grids - see "load_reps.py" to make repeater plots
    repeaters=False
    surveys, grids = loading.surveys_and_grids(survey_names = names, repeaters=False, sdir=sdir,nz=700,ndm=1400)
    
    # plots the zdm grid
    for i,grid in enumerate(grids):
        
        # adds a list of localised FRBs to the plot
        if surveys[i].zlist is not None:
            OK = surveys[i].zlist
            FRBDMs=surveys[i].DMEGs[OK]
            FRBZs=surveys[i].Zs[OK]
        else:
            FRBDMs=None
            FRBZs=None
        
        misc_functions.plot_grid_2(
                    grid.rates,
                    grid.zvals,
                    grid.dmvals,
                    FRBDMs=FRBDMs,
                    FRBZs=FRBZs,
                    name=opdir+names[i]+"_pzdm.png",
                    norm=3,
                    log=True,
                    label='$\\log_{10} p({\\rm DM}_{\\rm IGM} + {\\rm DM}_{\\rm host},z)$ [a.u.]',
                    project=True,
                    ylabel='${\\rm DM}_{\\rm IGM} + {\\rm DM}_{\\rm host}$',
                    zmax=3,DMmax=3000,
                    Aconts=[0.01, 0.1, 0.5]
                    )
    
    

main()
