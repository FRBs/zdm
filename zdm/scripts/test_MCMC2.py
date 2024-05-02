"""
File to test that cubing produces the expected output
"""

from zdm import MCMC
from zdm import MCMC2
from zdm import loading

from zdm import iteration as it

from zdm.misc_functions import *
from astropy.cosmology import Planck18
from pkg_resources import resource_filename

import json

def main():
    # use default parameters
    # Initialise survey and grid 
    # For this purporse, we only need two different surveys
    # the defaults are below - passing these will do nothing
    survey_names = ["DSA",
                    "FAST", 
                    "FAST2", 
                    "CRAFT_class_I_and_II", 
                    "private_CRAFT_ICS_892", 
                    "private_CRAFT_ICS_1300", 
                    "private_CRAFT_ICS_1632",
                    "parkes_mb_class_I_and_II"]
    
    rsurvey_names = ["CHIME/CHIME_decbin_0_of_6", 
                      "CHIME/CHIME_decbin_1_of_6",
                      "CHIME/CHIME_decbin_2_of_6", 
                      "CHIME/CHIME_decbin_3_of_6", 
                      "CHIME/CHIME_decbin_4_of_6", 
                      "CHIME/CHIME_decbin_5_of_6"]

    with open("../data/MCMC/params.json") as f:
        mcmc_dict = json.load(f)

    # Select from dictionary the necessary parameters to be changed
    params = {k: mcmc_dict[k] for k in mcmc_dict['mcmc']['parameter_order']}

    param_dict = {}
    for key,val in params.items():
        param_dict[key] = (val['max'] + val['min'])/2

    param_vals = []     
    for key,val in params.items():
        param_vals.append((val['max'] + val['min'])/2)

    state = parameters.State()
    state.set_astropy_cosmo(Planck18)

    surveys, grids = loading.surveys_and_grids(survey_names=survey_names, init_state=state, alpha_method=0)
    rsurveys, rgrids = loading.surveys_and_grids(survey_names=rsurvey_names, repeaters=True, init_state=state, alpha_method=0)
    ss = surveys + rsurveys
    grids += rgrids
        
    state2 = parameters.State()
    state2.set_astropy_cosmo(Planck18)
    state2.update_params(param_dict)

    surveys2, grids2 = loading.surveys_and_grids(survey_names=survey_names, init_state=state, alpha_method=0)
    rsurveys2, rgrids2 = loading.surveys_and_grids(survey_names=rsurvey_names, repeaters=True, init_state=state, alpha_method=0)
    ss2 = [surveys2, rsurveys2]
        
    grid_params = {}
    grid_params['dmmax'] = 7000.0
    grid_params['ndm'] = 1400
    grid_params['nz'] = 500

    lp = MCMC.calc_log_posterior(param_vals, params, ss, grids)
    lp2 = MCMC2.calc_log_posterior(param_vals, params, ss2, grid_params)

    print(lp, lp2)

main()
