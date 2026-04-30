""" 
Demonstrates how to set likelihood components for the MC analysis
"""
import os
import time

from zdm import iteration as it
from zdm import loading
from zdm import optical as opt
from zdm import optical_params as op
from zdm import states

import numpy as np

from matplotlib import pyplot as plt
import importlib.resources as resources

def main():
    """
    Calculates likelihoods for fake survey
    """
    
    state = states.load_state("HoffmannHalo25") # old scattering
    #name = "CRAFT_CRACO_900"
    name = "very_short_fake_CRACO_900"
    sdir = resources.files('zdm').joinpath('../papers/CombinedPathzDM/BridgetMC/Surveys/')
    surveys, grids = loading.surveys_and_grids(survey_names = [name],repeaters=False,
                                                sdir=sdir,init_state=state)
    
    s = surveys[0]
    g = grids[0]
    
    opstate = op.OpticalState()
    model = opt.loudas_model(opstate)
    wrapper = opt.model_wrapper(model,g.zvals)
    
    ### sets system variables to point towards fake FRB data
    galdir = resources.files('zdm').joinpath('../papers/CombinedPathzDM/BridgetMC/CandidateFiles')
    frbdir = resources.files('zdm').joinpath('../papers/CombinedPathzDM/BridgetMC/FRBFiles')
    os.environ["ZDM_PATH_FRBDIR"] = str(frbdir)
    os.environ["ZDM_PATH_GALDIR"] = str(galdir)
    
    lltot = it.get_joint_path_zdm_likelihoods(g, s, wrapper, norm=True, psnr=True, Pn=False,
                                    pdmz=True, pNreps=True, ptauw=False, pwb=True,
                                    return_all=False)
    
    
    
    
    
main()
