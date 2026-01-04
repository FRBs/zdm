"""
MCMC parameter estimation for FRB z-DM analysis.

This module provides functions for running Markov Chain Monte Carlo (MCMC)
parameter estimation using the emcee package. It interfaces with the zdm
likelihood calculations to explore the parameter space and constrain
FRB population and cosmological parameters.

Main Functions
--------------
- `calc_log_posterior`: Compute log-posterior for a parameter vector
- `run_mcmc`: Execute MCMC sampling with emcee
- `get_initial_walkers`: Initialize walker positions

Features
--------
- Uniform priors with configurable bounds
- Optional log/linear priors for specific parameters (DMhalo, host DM)
- Support for multiple surveys and repeater populations
- Grid re-initialization on each evaluation for parameter exploration

Example
-------
>>> from zdm import MCMC
>>> params = {'gamma': {'min': -2.5, 'max': -0.5}, ...}
>>> sampler = MCMC.run_mcmc(state, params, surveys, nwalkers=32, nsteps=1000)
>>> samples = sampler.get_chain(flat=True)

Author: Jordan Hoffmann
Date: 06/12/23
"""

import numpy as np

import zdm.iteration as it
import importlib.resources as resources

import emcee
import scipy.stats as st
import time

from zdm import loading
from zdm import parameters

from astropy.cosmology import Planck18

import multiprocessing as mp

from zdm import misc_functions as mf
from zdm import repeat_grid
import os
#==============================================================================

def calc_log_posterior(param_vals, state, params, surveys_sep, Pn=False, pNreps=True, ptauw=False,
                log_halo=False, lin_host=False, ind_surveys=False, g0info=None):
    """Calculate log-posterior probability for a parameter vector.

    This is the main function called by emcee samplers. It evaluates the
    log-posterior (proportional to log-likelihood for uniform priors) by
    building grids and computing likelihoods for all surveys.

    Parameters
    ----------
    param_vals : ndarray
        Array of parameter values for this MCMC step.
    state : parameters.State
        State object to be updated with new parameter values.
    params : dict
        Dictionary defining parameters to vary. Each key is a parameter name,
        with value dict containing 'min' and 'max' for prior bounds.
    surveys_sep : list
        Two-element list: [non_repeater_surveys, repeater_surveys].
    Pn : bool, optional
        Include Poisson likelihood for total number of FRBs. Default False.
    pNreps : bool, optional
        Include likelihood for number of repeaters. Default True.
    ptauw : bool, optional
        Include p(tau, width) likelihood. Default False.
    log_halo : bool, optional
        Use log-uniform prior on DMhalo. Default False.
    lin_host : bool, optional
        Use linear-uniform prior on host DM mean. Default False.
    ind_surveys : bool, optional
        If True, return list of individual survey likelihoods. Default False.
    g0info : list, optional
        Pre-computed [zDMgrid, zvals, DMvals] for speedup.

    Returns
    -------
    float or tuple
        Log-posterior value. Returns -inf if parameters outside prior bounds.
        If ind_surveys=True, returns (llsum, ll_list) with individual likelihoods.
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

        if lin_host and key == 'lmean':
            param_dict[key] = np.log10(param_vals[i])
        else:
            param_dict[key] = param_vals[i]

    # Initialise list if requesting individual survey likelihoods
    if ind_surveys:
        ll_list = []
    
    if g0info is not None:
        # extract zm grid initial info
        zDMgrid = g0info[0]
        zvals = g0info[1]
        dmvals = g0info[2]
    
    # Check if it is in the priors and do the calculations
    if in_priors is False:
        llsum = -np.inf
    else:
        # minimise_const_only does the grid updating so we don't need to do it explicitly beforehand
        # In an MCMC analysis the parameter spaces are sampled throughout and hence with so many parameters
        # it is easy to reach impossible regions of the parameter space. This results in math errors
        # (log(0), log(negative), sqrt(negative), divide 0 etc.) and hence we assume that these math errors
        # correspond to an impossible region of the parameter space and so set ll = -inf
        #try:
        if True:
            # Set state
            state.update_params(param_dict)

            surveys = surveys_sep[0] + surveys_sep[1]

            # Recreate grids every time, but not surveys, so must update survey params
            for i,s in enumerate(surveys):
                
                
                # updates survey according to DMhalo estimates
                if 'DMhalo' in param_dict:
                    if log_halo:
                        DMhalo = 10**param_dict['DMhalo']
                    else:
                        DMhalo = param_dict['DMhalo']
                    s.init_DMEG(DMhalo)
                    
                if ('Wlogmean' in param_dict or 'Wlogsigma' in param_dict or \
                    'Slogmean'  in param_dict or 'Slogsigma' in param_dict):
                    state.scat.Sbackproject = True
                    s.init_widths(state=state)
                elif 'DMhalo' in param_dict:
                    # this would get re-done within init_widths above, so only do this
                    # if it has *not* been recalculated
                    s.do_efficiencies() #get_efficiency_from_wlist(s.wlist,s.wplist,model=s.meta['WBIAS']) 
            
            # Initialise grids
            grids = []
            
            # gets new zDM grid if F and H0 in the param_dict
            if 'H0' in param_dict or 'logF' in param_dict or g0info is None:
                datdir = resources.files('zdm').joinpath('GridData')
                zDMgrid, zvals,dmvals = mf.get_zdm_grid(
                    state, new=True, plot=False, method='analytic',
                    datdir=datdir)
                g0info = [zDMgrid, zvals,dmvals]
            
            if len(surveys_sep[0]) != 0:
                # generates zdm grid
                grids += mf.initialise_grids(surveys_sep[0], zDMgrid, zvals, dmvals, state, wdist=True, repeaters=False)
            
            if len(surveys_sep[1]) != 0:
                # generates zdm grid
                grids += mf.initialise_grids(surveys_sep[1], zDMgrid, zvals, dmvals, state, wdist=True, repeaters=True)
            
            # Minimse the constant accross all surveys
            if Pn:
                newC, llC = it.minimise_const_only(None, grids, surveys, update=True)
                # for g in grids:
                #     g.state.FRBdemo.lC = newC

                # if isinstance(g, repeat_grid.repeat_Grid):
                #     g.state.rep.RC = g.state.rep.RC / 10**oldC * 10**newC

            # calculate all the likelihoods
            llsum = 0
            for s, grid in zip(surveys, grids):
                ll = it.get_log_likelihood(grid,s,Pn=Pn,pNreps=pNreps,ptauw=ptauw)
                llsum += ll
                if ind_surveys:
                    ll_list.append(ll)

        #except ValueError as e:
        #    print("Error, setting likelihood to -inf: " + str(e))
        #    llsum = -np.inf
        #    ll_list = [-np.inf for _ in range(len(surveys))]

    if np.isnan(llsum):
        print("llsum was NaN. Setting to -infinity", param_dict)    
        llsum = -np.inf
    
    # print("Posterior calc time: " + str(time.time()-t0) + " seconds", flush=True)
    
    if ind_surveys:
        return llsum, ll_list
    else:
        return llsum

#==============================================================================

def mcmc_runner(logpf, outfile, state, params, surveys, nwalkers=10, nsteps=100, nthreads=1, Pn=False,
                pNreps=True, ptauw = False, log_halo=False, lin_host=False, ind_surveys=False, g0info=None,
                reset=False):
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
        pNreps      (bool)          =   Include pNreps or not
        ptauw       (bool)          =   Include ptauw or not
        log_halo    (bool)          =   Use a log uniform prior on DMhalo
        ind_surveys (bool)          =   Return individual survey data
        g0info      (list)          =   List of [zDMgrid, zvals, DMvals] Passed to use as speedup if needed
    
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
    
    # we only reset the backend if specifically requested.
    # This means that walkers will continue from a previous iteration
    backend = emcee.backends.HDFBackend(outfile+'.h5')
    exists = os.path.isfile(outfile+'.h5')
    if reset:
        backend.reset(nwalkers, ndim)
        if exists:
            print("WARNING: output file exists, will be writing new run to old file")
        exists = False # if resetting, ignore that a file exists
    
    start = time.time()
    
    # may or may not be needed
    #os.environ["OMP_NUM_THREADS"] = "1"
    import multiprocessing as mp
    Pool = mp.get_context('fork').Pool
    
    
    
    with Pool() as pool: # could add mp.Pool(ntrheads=5) or Pool = None
        sampler = emcee.EnsembleSampler(nwalkers, ndim, logpf, args=[state, params, surveys, Pn, pNreps,
                                        ptauw, log_halo, lin_host, ind_surveys, g0info], backend=backend, pool=pool)
        if exists:
            # start from last saved position
            sampler.run_mcmc(None, nsteps, progress=True)
        else:
            # start from new random guesses
            sampler.run_mcmc(starting_guesses, nsteps, progress=True)
    end = time.time()
    print("Total time taken: " + str(end - start))
    
    posterior_sample = sampler.get_chain()

    return posterior_sample

#==============================================================================
