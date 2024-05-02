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
import scipy.stats as st

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

    # newC, llC = it.minimise_const_only(param_dict, grids, surveys)
    # print(newC)
    # for g in grids:
    #     g.state.FRBdemo.lC = newC

    # # calculate all the likelihoods
    # llsum = 0
    # for s, grid in zip(surveys, grids):
    #     llsum += it.get_log_likelihood(grid,s)

    state = parameters.State()
    state.set_astropy_cosmo(Planck18)
    surveys, grids = loading.surveys_and_grids(survey_names=survey_names, repeaters=False, init_state=state, alpha_method=0)

    state2 = parameters.State()
    state2.set_astropy_cosmo(Planck18)
    state2.update_params(param_dict)
    surveys2, grids2 = loading.surveys_and_grids(survey_names=survey_names, repeaters=False, init_state=state2, alpha_method=0)

    grid_params = {}
    grid_params['dmmax'] = 7000.0
    grid_params['ndm'] = 1400
    grid_params['nz'] = 500

    # # generates zdm grid
    # grids2 = initialise_grids(surveys, zDMgrid, zvals, dmvals, state, wdist=True, repeaters=False)

    # newC, llC = it.minimise_const_only(param_dict, grids2, surveys)
    # print(newC)
    # for g in grids2:
    #     g.state.FRBdemo.lC = newC

    # for i,g in enumerate(grids):
    #     diff = g.rates - grids2[i].rates
    #     print(np.max(diff))

    # print(grids[0].state)
    # print(grids2[0].state)

    print("MCMC")
    lp = MCMC.calc_log_posterior(param_vals, params, surveys, grids)
    print("MCMC2")
    lp2 = MCMC2.calc_log_posterior(param_vals, params, [surveys2, []], grid_params)

    print(lp, lp2)

main()
