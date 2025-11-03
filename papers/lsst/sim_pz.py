"""
Simulates p(z) for lSST paper
"""

""" 
This script creates zdm grids for MeerTRAP
                                                                                                                                                         eerTRAPcoherent']
"""
import os

from astropy.cosmology import Planck18
from zdm import cosmology as cos
from zdm import figures
from zdm import parameters
from zdm import survey
from zdm import pcosmic
from zdm import iteration as it
from zdm import loading
from zdm import io
from zdm import optical as opt
from zdm import states

import numpy as np
from zdm import survey
from matplotlib import pyplot as plt
import importlib.resources as resources

def main():
    
    # approximate best-fit values from recent analysis
    # load states from Hoffman et al 2025
    # use b or d for rep
    state = states.load_state("HoffmannEmin25",scat="updated",rep='b')
    opdir="pz/"
    
    # artificially add repeater data - we can't actually know this,
    # because we don't have time per field. Just using one day for now
    survey_dict={}
    survey_dict["TFIELD"] = 24.
    survey_dict["TOBS"] = 24.
    
    survey_dict["NORM_REPS"] = 0
    survey_dict["NORM_SINGLES"] = 0
    survey_dict["NORM_FRB"] = 0
    
    
    if not os.path.exists(opdir):
        os.mkdir(opdir)
    
    # Initialise surveys and grids
    sdir = resources.files('zdm').joinpath('data/Surveys')
    names=['MeerTRAPcoherent']
    names=['CRAFT_CRACO_1300','MeerTRAPcoherent','SKA_mid']
    
    ss,gs = loading.surveys_and_grids(survey_names=names,repeaters=True,init_state=state,
                                        sdir=sdir,survey_dict=survey_dict)
    
    # now use p(z) to estimate optical magnitudes
    
main()
