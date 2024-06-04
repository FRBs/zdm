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

import emcee
import scipy.stats as st
import time

import multiprocessing as mp

from zdm import misc_functions as mf
from zdm import zdm_repeat_grid

#==============================================================================

def calc_log_posterior(param_vals, params, surveys, grids):
    """
    Calculates the log posterior for a given set of parameters. Assumes uniform
    priors between the minimum and maximum values provided in 'params'.

    Inputs:
        param_vals  (np.array)      =   Array of the parameter values for this step
        params      (dictionary)    =   Parameter names, min and max values
        surveys_sep (list)          =   List of surveys
        grids       (list)          =   List of grids corresponding to surveys
    
    Outputs:
        llsum       (double)        =   Total log likelihood for param_vals which is equivalent
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
        # In an MCMC analysis the parameter spaces are sampled throughout and hence with so many parameters
        # it is easy to reach impossible regions of the parameter space. This results in math errors
        # (log(0), log(negative), sqrt(negative), divide 0 etc.) and hence we assume that these math errors
        # correspond to an impossible region of the parameter space and so set ll = -inf
        try:
            newC, llC = it.minimise_const_only(param_dict, grids, surveys)
            for g in grids:
                g.state.FRBdemo.lC = newC

                if isinstance(g, zdm_repeat_grid.repeat_Grid):
                    g.calc_constant()

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

def mcmc_runner(logpf, outfile, params, surveys, grids, nwalkers=10, nsteps=100, nthreads=1):
    """
    Handles the MCMC running.

    Inputs:
        logpf       (function)      =   Log posterior function handle
        outfile     (string)        =   Name of the output file (excluding .h5 extension)
        params      (dictionary)    =   Parameter names, min and max values
        surveys_sep (list)          =   List of surveys
        grids       (list)          =   List of grids corresponding to surveys
        nwalkers    (int)           =   Number of walkers
        nsteps      (int)           =   Number of steps
        nthreads    (int)           =   Number of threads (currently not implemented - uses default)
    
    Outputs:
        posterior_sample    (emcee.EnsembleSampler) =   Final sample
        outfile.h5          (HDF5 file)             =   HDF5 file containing the sampler
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

    # start = time.time()
    with mp.Pool() as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, logpf, args=[params, surveys, grids], backend=backend, pool=pool)
        sampler.run_mcmc(starting_guesses, nsteps, progress=True)
    # end = time.time()
    # print("Total time taken: " + str(end - start))
    
    posterior_sample = sampler.get_chain()

    return posterior_sample

#==============================================================================