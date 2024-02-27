"""
File: MCMC.py
Author: Jordan Hoffmann
Date: 06/12/23
Purpose: 
    Contains functions used for MCMC runs of the zdm code. MCMC_wrap.py is the 
    main function which does the calling and this holds functions which do the 
    MCMC analysis.
"""

import numpy as np

import zdm.iteration as it
from pkg_resources import resource_filename

import emcee
import scipy.stats as st
import time

from zdm import loading
from zdm import parameters

from astropy.cosmology import Planck18

import multiprocessing as mp

from zdm.misc_functions import *

#==============================================================================

def calc_log_posterior(param_vals, params, surveys_sep, grid_params):
    """
    Calculates the log posterior for a given set of parameters. Assumes uniform
    priors between the minimum and maximum values provided in 'params'.

    Inputs:
        param_vals  =   Array of the parameter values for this step
        params      =   Dictionary of the parameter names, min and max values
        files       =   Object containing survey_names, rep_survey_names, sdir, edir
    
    Outputs:
        llsum       =   Total log likelihood for param_vals which is equivalent
                        to log posterior (un-normalised) due to uniform priors
    """

    # t0 = time.time()
    # Can use likelihoods instead of posteriors because we only use uniform priors which just changes normalisation of posterior 
    # given every value is in the correct range. If any value is not in the correct range, log posterior is -inf
    in_priors = True
    param_dict = {}

    for i, (key,val) in enumerate(params.items()):
        if param_vals[i] < val['min'] or param_vals[i] > val['max']:
            in_priors = False
            break

        param_dict[key] = param_vals[i]

    if in_priors == False:
        llsum = -np.inf
    else:

        # minimise_const_only does the grid updating so we don't need to do it explicitly beforehand
        try:
            # Set state
            state = parameters.State()
            state.set_astropy_cosmo(Planck18) 
            state.update_params(param_dict)
            # state.update_param('alpha_method', 0)
            # state.update_param('luminosity_function', 2)

            surveys = surveys_sep[0] + surveys_sep[1]

            # Recreate grids every time, but not surveys, so must update survey params
            for i,s in enumerate(surveys):
                if 'DMhalo' in param_dict:
                    s.init_DMEG(param_dict['DMhalo'])
                    s.get_efficiency_from_wlist(s.DMlist,s.wlist,s.wplist,model=s.meta['WBIAS']) 

            # Initialise grids
            grids = []
            if len(surveys_sep[0]) != 0:
                zDMgrid, zvals,dmvals = get_zdm_grid(
                    state, new=True, plot=False, method='analytic', 
                    nz=grid_params['nz'], ndm=grid_params['ndm'], dmmax=grid_params['dmmax'],
                    datdir=resource_filename('zdm', 'GridData'))

                # generates zdm grid
                grids += initialise_grids(surveys_sep[0], zDMgrid, zvals, dmvals, state, wdist=True, repeaters=False)
            
            if len(surveys_sep[1]) != 0:
                zDMgrid, zvals,dmvals = get_zdm_grid(
                    state, new=True, plot=False, method='analytic', 
                    nz=grid_params['nz'], ndm=grid_params['ndm'], dmmax=grid_params['dmmax'],
                    datdir=resource_filename('zdm', 'GridData'))

                # generates zdm grid
                grids += initialise_grids(surveys_sep[1], zDMgrid, zvals, dmvals, state, wdist=True, repeaters=True)

            # Minimse the constant accross all surveys
            newC, llC = it.minimise_const_only(None, grids, surveys, update=True)
            # for g in grids:
            #     g.state.FRBdemo.lC = newC

            #     if isinstance(g, zdm_repeat_grid.repeat_Grid):
            #         g.calc_constant()

            # calculate all the likelihoods
            llsum = 0
            for s, grid in zip(surveys, grids):
                llsum += it.get_log_likelihood(grid,s)

        except ValueError as e:
            print("ValueError, setting likelihood to -inf: " + str(e))
            llsum = -np.inf

    if np.isnan(llsum):
        print("llsum was NaN. Setting to -infinity", param_dict)    
        llsum = -np.inf
    
    # print("Posterior calc time: " + str(time.time()-t0) + " seconds", flush=True)

    return llsum

#==============================================================================

def mcmc_runner(logpf, outfile, params, surveys, grid_params, nwalkers=10, nsteps=100, nthreads=1):
    """
    Handles the MCMC running.

    Inputs:
        logpf       =   Log posterior function handle
        outfile     =   Name of the output file (excluding .h5 extension)
        params      =   Dictionary of the parameter names, min and max values
        surveys     =   List of surveys being used
        grids       =   List of grids corresponding to the surveys
        nwalkers    =   Number of walkers
        nsteps      =   Number of steps per walker
        nthreads    =   Number of threads to use for parallelised runs
    
    Outputs:
        posterior_sample    =   Final sample
        outfile.h5          =   HDF5 file containing the sampler
    """
        
    ndim = len(params)
    starting_guesses = []

    # Produce starting guesses for each parameter
    for key,val in params.items():
        starting_guesses.append(st.uniform(loc=val['min'], scale=val['max']-val['min']).rvs(size=[nwalkers]))
        print(key + " priors: " + str(val['min']) + "," + str(val['max']))
    
    starting_guesses = np.array(starting_guesses).T

    backend = emcee.backends.HDFBackend(outfile+'.h5')
    backend.reset(nwalkers, ndim)
    
    start = time.time()
    with mp.Pool() as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, logpf, args=[params, surveys, grid_params], backend=backend, pool=pool)
        sampler.run_mcmc(starting_guesses, nsteps, progress=True)
    end = time.time()
    print("Total time taken: " + str(end - start))
    
    posterior_sample = sampler.get_chain()

    return posterior_sample

#==============================================================================