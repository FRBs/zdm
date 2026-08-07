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
import time

from zdm import loading
from zdm import parameters

from astropy.cosmology import Planck18

import multiprocessing as mp

from zdm import misc_functions as mf
from zdm import repeat_grid
import os
#==============================================================================

def valid_parameter_combination(param_dict, state):
    """Check joint constraints that cannot be expressed as 1-D priors.

    For broken power-law luminosity functions, the characteristic energies
    are stored in log10 space and must remain strictly ordered.
    Values absent from ``param_dict`` are taken from ``state``, allowing any
    subset of the energy parameters to be sampled.
    """
    luminosity_function = param_dict.get(
        'luminosity_function', state.energy.luminosity_function
    )
    lEmin = param_dict.get('lEmin', state.energy.lEmin)
    lEmax = param_dict.get('lEmax', state.energy.lEmax)
    if luminosity_function == 4:
        lEb = param_dict.get('lEb', state.energy.lEb)
        return bool(lEmin < lEb < lEmax)
    if luminosity_function == 5:
        lEb = param_dict.get('lEb', state.energy.lEb)
        lEb2 = param_dict.get('lEb2', state.energy.lEb2)
        return bool(lEmin < lEb < lEb2 < lEmax)
    return True


def get_initial_walkers(state, params, nwalkers, rng=None, max_attempts=10000):
    """Draw walker positions from the priors, respecting joint constraints."""
    if rng is None:
        rng = np.random.default_rng()

    param_names = list(params)
    ndim = len(param_names)
    walkers = np.empty((nwalkers, ndim), dtype=float)

    for iwalker in range(nwalkers):
        for _ in range(max_attempts):
            candidate = np.array([
                rng.uniform(params[name]['min'], params[name]['max'])
                for name in param_names
            ])
            candidate_dict = dict(zip(param_names, candidate))
            if valid_parameter_combination(candidate_dict, state):
                walkers[iwalker] = candidate
                break
        else:
            raise ValueError(
                "Could not initialize MCMC walkers inside the joint priors. "
                "For luminosity_function=4 or 5, ensure the prior ranges "
                "permit the required ordering of break energies."
            )

    return walkers

#==============================================================================

def calc_log_posterior(param_vals, state, params, surveys_sep, Pn=False, pNreps=True, psnr=True, ptauw=False, pwb=False,
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
    pwb : bool, optional
        Include individual beam likelihoods. Default False.
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

    if in_priors and not valid_parameter_combination(param_dict, state):
        in_priors = False

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
            grid_kwargs = {}
            if g0info is not None:
                # Preserve the resolution selected by MCMC_wrap. Previously,
                # sampling H0/logF silently reverted every worker to the
                # 500 x 1400 default grid, causing both shape errors and large
                # unexpected memory use in low-resolution pilot runs.
                dz = zvals[-1] - zvals[-2]
                ddm = dmvals[-1] - dmvals[-2]
                grid_kwargs = {
                    'nz': zvals.size,
                    'zmax': zvals[-1] + dz / 2,
                    'ndm': dmvals.size,
                    'dmmax': dmvals[-1] + ddm / 2,
                }
            zDMgrid, zvals,dmvals = mf.get_zdm_grid(
                state, new=True, plot=False, method='analytic',
                datdir=datdir, **grid_kwargs)
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

        # calculate all the likelihoods
        llsum = 0
        for s, grid in zip(surveys, grids):
            ll = it.get_log_likelihood(grid,s,Pn=Pn,pNreps=pNreps,psnr=psnr,ptauw=ptauw,pwb=pwb)
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
                pNreps=True, psnr=True, ptauw=False, pwb=False, log_halo=False, lin_host=False, ind_surveys=False, g0info=None,
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
        nthreads    (int)           =   Number of worker processes
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
    # Report priors in sampling order.
    for key,val in params.items():
        print(key + " priors: " + str(val['min']) + "," + str(val['max']))

    # Draw only physically valid initial positions. Broken power-law walkers
    # must satisfy the required ordering of their characteristic energies.
    starting_guesses = get_initial_walkers(state, params, nwalkers)
    
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
    
    if nthreads < 1:
        raise ValueError("nthreads must be at least 1")

    # Prevent numerical libraries from starting extra threads inside each
    # worker process, which can otherwise multiply both CPU and memory use.
    os.environ["OMP_NUM_THREADS"] = "1"
    import multiprocessing as mp
    Pool = mp.get_context('fork').Pool

    def run_sampler(pool):
        sampler = emcee.EnsembleSampler(nwalkers, ndim, logpf, args=[state, params, surveys, Pn, pNreps, psnr,
                                        ptauw, pwb, log_halo, lin_host, ind_surveys, g0info], backend=backend, pool=pool)
        if exists:
            # start from last saved position
            sampler.run_mcmc(None, nsteps, progress=True)
        else:
            # start from new random guesses
            sampler.run_mcmc(starting_guesses, nsteps, progress=True)
        return sampler

    if nthreads == 1:
        # Avoid creating a second Python process for the memory-conservative
        # default mode.
        sampler = run_sampler(None)
    else:
        # Recycling workers periodically releases arrays retained by Python's
        # allocator during repeated six-survey grid construction.
        with Pool(processes=nthreads, maxtasksperchild=10) as pool:
            sampler = run_sampler(pool)
    end = time.time()
    print("Total time taken: " + str(end - start))
    
    posterior_sample = sampler.get_chain()

    return posterior_sample

#==============================================================================
