"""
File: MCMC2.py
Author: Jordan Hoffmann
Date: 06/12/23
Purpose: 
    Contains functions used for MCMC runs of the zdm code. MCMC_wrap2.py is the 
    main function which does the calling and this holds functions which do the 
    MCMC analysis.

    This function re-initialises the grids on every run while MCMC.py updates the grid.
    Typically this run is more efficient than MCMC.py.
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

from zdm import misc_functions as mf
from zdm import repeat_grid

#==============================================================================

def calc_log_posterior(param_vals, state, params, surveys_sep, grid_params, Pn=True, log_halo=False):
    """
    Calculates the log posterior for a given set of parameters. Assumes uniform
    priors between the minimum and maximum values provided in 'params'.

    Inputs:
        param_vals  (np.array)      =   Array of the parameter values for this step
        state       (params.state)  =   State object to modify
        params      (dictionary)    =   Parameter names, min and max values
        surveys_sep (list)          =   surveys_sep[0] : list of non-repeater surveys
                                        surveys_sep[1] : list of repeater surveys
        grid_params (dictionary)    =   nz, ndm, dmmax
        Pn          (bool)          =   Include Pn or not
        log_halo    (bool)          =   Use a log uniform prior on DMhalo
    
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

    if in_priors is False:
        llsum = -np.inf
    else:
        # minimise_const_only does the grid updating so we don't need to do it explicitly beforehand
        # In an MCMC analysis the parameter spaces are sampled throughout and hence with so many parameters
        # it is easy to reach impossible regions of the parameter space. This results in math errors
        # (log(0), log(negative), sqrt(negative), divide 0 etc.) and hence we assume that these math errors
        # correspond to an impossible region of the parameter space and so set ll = -inf
        try:
            # Set state
            state.update_params(param_dict)

            surveys = surveys_sep[0] + surveys_sep[1]

            # Recreate grids every time, but not surveys, so must update survey params
            for i,s in enumerate(surveys):
                if 'DMhalo' in param_dict:
                    if log_halo:
                        DMhalo = 10**param_dict['DMhalo']
                    else:
                        DMhalo = param_dict['DMhalo']
                    s.init_DMEG(DMhalo)
                    s.get_efficiency_from_wlist(s.DMlist,s.wlist,s.wplist,model=s.meta['WBIAS']) 

            # Initialise grids
            grids = []
            if len(surveys_sep[0]) != 0:
                zDMgrid, zvals,dmvals = mf.get_zdm_grid(
                    state, new=True, plot=False, method='analytic', 
                    nz=grid_params['nz'], ndm=grid_params['ndm'], dmmax=grid_params['dmmax'],
                    datdir=resource_filename('zdm', 'GridData'))

                # generates zdm grid
                grids += mf.initialise_grids(surveys_sep[0], zDMgrid, zvals, dmvals, state, wdist=True, repeaters=False)
            
            if len(surveys_sep[1]) != 0:
                zDMgrid, zvals,dmvals = mf.get_zdm_grid(
                    state, new=True, plot=False, method='analytic', 
                    nz=grid_params['nz'], ndm=grid_params['ndm'], dmmax=grid_params['dmmax'],
                    datdir=resource_filename('zdm', 'GridData'))

                # generates zdm grid
                grids += mf.initialise_grids(surveys_sep[1], zDMgrid, zvals, dmvals, state, wdist=True, repeaters=True)

            # Minimse the constant accross all surveys
            newC, llC = it.minimise_const_only(None, grids, surveys, update=True)
            if Pn:
                for g in grids:
                    g.state.FRBdemo.lC = newC

                if isinstance(g, repeat_grid.repeat_Grid):
                    g.calc_constant()

            # calculate all the likelihoods
            llsum = 0
            for s, grid in zip(surveys, grids):
                llsum += it.get_log_likelihood(grid,s,Pn=Pn)

        except ValueError as e:
            print("ValueError, setting likelihood to -inf: " + str(e))
            llsum = -np.inf

    if np.isnan(llsum):
        print("llsum was NaN. Setting to -infinity", param_dict)    
        llsum = -np.inf
    
    # print("Posterior calc time: " + str(time.time()-t0) + " seconds", flush=True)

    return llsum

#==============================================================================

def mcmc_runner(logpf, outfile, state, params, surveys, grid_params, nwalkers=10, nsteps=100, nthreads=1, Pn=True, log_halo=False):
    """
    Handles the MCMC running.

    Inputs:
        logpf       (function)      =   Log posterior function handle
        outfile     (string)        =   Name of the output file (excluding .h5 extension)
        state       (params.state)  =   State object to modify
        params      (dictionary)    =   Parameter names, min and max values
        surveys     (list)          =   surveys_sep[0] : list of non-repeater surveys
                                        surveys_sep[1] : list of repeater surveys
        grid_params (dictionary)    =   nz, ndm, dmmax
        nwalkers    (int)           =   Number of walkers
        nsteps      (int)           =   Number of steps
        nthreads    (int)           =   Number of threads (currently not implemented - uses default)
        Pn          (bool)          =   Include Pn or not
        log_halo    (bool)          =   Use a log uniform prior on DMhalo
    
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
    
    start = time.time()
    with mp.Pool() as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, logpf, args=[state, params, surveys, grid_params, Pn, log_halo], backend=backend, pool=pool)
        sampler.run_mcmc(starting_guesses, nsteps, progress=True)
    end = time.time()
    print("Total time taken: " + str(end - start))
    
    posterior_sample = sampler.get_chain()

    return posterior_sample

#==============================================================================