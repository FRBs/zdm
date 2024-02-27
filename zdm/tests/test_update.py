""" Generate, load, etc. Surveys """
import numpy as np
from pkg_resources import resource_filename

import json

from zdm import loading
from zdm import parameters 
from zdm import MCMC
from zdm import MCMC2
from zdm import iteration as it
from zdm.misc_functions import *
from astropy.cosmology import Planck18

import pytest

import os

def test_update_MCMC():

    survey_names = ["private_CRAFT_ICS_892"]
    rsurvey_names = ["CHIME/CHIME_decbin_0_of_6"]

    # Select from dictionary all variable parameters
    with open(os.path.join(resource_filename('zdm', 'data'), "MCMC/params.json")) as f:
        mcmc_dict = json.load(f)

    params = {k: mcmc_dict[k] for k in mcmc_dict['mcmc_full']['parameter_order']}

    # Set all values to random point in prior
    param_dict = {}
    for key,val in params.items():
        param_dict[key] = np.random.rand() * (val['max'] - val['min']) + val['min']

    param_vals = []     
    for key,val in param_dict.items():
        param_vals.append(val)

    # Make default state, grid, survey
    state = parameters.State()
    state.set_astropy_cosmo(Planck18)

    surveys, grids = loading.surveys_and_grids(survey_names=survey_names, init_state=state, alpha_method=0)
    rsurveys, rgrids = loading.surveys_and_grids(survey_names=rsurvey_names, repeaters=True, init_state=state, alpha_method=0)
    ss = surveys + rsurveys
    grids += rgrids
    
    # Make updated state, survey
    state2 = parameters.State()
    state2.set_astropy_cosmo(Planck18)
    state2.update_params(param_dict)

    surveys2, _ = loading.surveys_and_grids(survey_names=survey_names, init_state=state2, alpha_method=0)
    rsurveys2, _ = loading.surveys_and_grids(survey_names=rsurvey_names, repeaters=True, init_state=state2, alpha_method=0)
    ss2 = [surveys2, rsurveys2]
        
    grid_params = {}
    grid_params['dmmax'] = 7000.0
    grid_params['ndm'] = 1400
    grid_params['nz'] = 500

    # Calc likelihoods
    lp = MCMC.calc_log_posterior(param_vals, params, ss, grids)
    lp2 = MCMC2.calc_log_posterior(param_vals, params, ss2, grid_params) 

    assert np.isclose(lp, lp2)

#==============================================================================

def test_update():

    survey_names = ["private_CRAFT_ICS_892"]
    rsurvey_names = ["CHIME/CHIME_decbin_0_of_6"]

    # Select from dictionary all variable parameters
    with open(os.path.join(resource_filename('zdm', 'data'), "MCMC/params.json")) as f:
        mcmc_dict = json.load(f)

    params = {k: mcmc_dict[k] for k in mcmc_dict['mcmc_full']['parameter_order']}

    # Set all values to random point in prior
    param_dict = {}
    for key,val in params.items():
        param_dict[key] = np.random.rand() * (val['max'] - val['min']) + val['min']

    param_vals = []     
    for key,val in param_dict.items():
        param_vals.append(val)

    # Make default state, grid, survey
    state = parameters.State()
    state.set_astropy_cosmo(Planck18)

    surveys, grids = loading.surveys_and_grids(survey_names=survey_names, init_state=state, alpha_method=0)
    rsurveys, rgrids = loading.surveys_and_grids(survey_names=rsurvey_names, repeaters=True, init_state=state, alpha_method=0)
    surveys += rsurveys
    grids += rgrids
    
    newC, llC = it.minimise_const_only(param_dict, grids, surveys)
    for g in grids:
        g.state.FRBdemo.lC = newC

        if isinstance(g, zdm_repeat_grid.repeat_Grid):
            g.calc_constant()

    # calculate all the likelihoods
    llsum = 0
    for s, grid in zip(surveys, grids):
        llsum += it.get_log_likelihood(grid,s)

    #==========================================================================
    # Make updated state, survey
    state2 = parameters.State()
    state2.set_astropy_cosmo(Planck18)
    state2.update_params(param_dict)

    surveys2, _ = loading.surveys_and_grids(survey_names=survey_names, init_state=state2, alpha_method=0)
    rsurveys2, _ = loading.surveys_and_grids(survey_names=rsurvey_names, repeaters=True, init_state=state2, alpha_method=0)
        
    grid_params = {}
    grid_params['dmmax'] = 7000.0
    grid_params['ndm'] = 1400
    grid_params['nz'] = 500

    # Initialise grids
    grids2 = []
    if len(surveys2) != 0:
        zDMgrid, zvals,dmvals = get_zdm_grid(
            state2, new=True, plot=False, method='analytic', 
            nz=grid_params['nz'], ndm=grid_params['ndm'], dmmax=grid_params['dmmax'],
            datdir=resource_filename('zdm', 'GridData'))

        # generates zdm grid
        grids2 += initialise_grids(surveys2, zDMgrid, zvals, dmvals, state2, wdist=True, repeaters=False)
    
    if len(rsurveys2) != 0:
        zDMgrid, zvals,dmvals = get_zdm_grid(
            state2, new=True, plot=False, method='analytic', 
            nz=grid_params['nz'], ndm=grid_params['ndm'], dmmax=grid_params['dmmax'],
            datdir=resource_filename('zdm', 'GridData'))

        # generates zdm grid
        grids2 += initialise_grids(rsurveys2, zDMgrid, zvals, dmvals, state2, wdist=True, repeaters=True)

    surveys2 += rsurveys2

    # Minimse the constant accross all surveys
    newC, llC = it.minimise_const_only(None, grids2, surveys2)
    for g in grids2:
        g.state.FRBdemo.lC = newC

        if isinstance(g, zdm_repeat_grid.repeat_Grid):
            g.calc_constant()

    # calculate all the likelihoods
    llsum2 = 0
    for s, grid in zip(surveys2, grids2):
        llsum2 += it.get_log_likelihood(grid,s)


    # print(llsum, llsum2)
    assert np.isclose(llsum, llsum2)

    for g1, g2 in zip(grids, grids2):
        # print(np.sum(g1.eff_table), np.sum(g2.eff_table))
        # print(np.sum(g1.rates), np.sum(g2.rates))
        assert np.all(np.isclose(g1.rates, g2.rates))

        if isinstance(g1, zdm_repeat_grid.repeat_Grid):
            # print(np.sum(g1.exact_singles), np.sum(g2.exact_singles))
            # print(np.sum(g1.exact_reps), np.sum(g2.exact_reps))

            assert np.all(np.isclose(g1.exact_singles, g2.exact_singles))
            assert np.all(np.isclose(g1.exact_reps, g2.exact_reps))

    # print("g1", g1.state)
    # print("g2", g2.state)

test_update_MCMC()
