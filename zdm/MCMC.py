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
from zdm import energetics

from astropy.cosmology import Planck18

import multiprocessing as mp

from zdm import cosmology as cos
from zdm import misc_functions as mf
from zdm import repeat_grid
import os
import cProfile

from zdm import optical_numerics as on
from zdm import optical as opt
from zdm import optical_params as op
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
    if luminosity_function in (4, 6):
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
                "For luminosity_function=4, 5, or 6, ensure the prior ranges "
                "permit the required ordering of break energies."
            )

    return walkers

#==============================================================================

PROFILED_PID = None

def profiled_calc_log_posterior(param_vals, state, params, surveys_sep, Pn=False, Pns=False, Pnr=False,
                pNreps=True, psnr=True, ptauw=False, pwb=False,
                log_halo=False, lin_host=False, ind_surveys=False, g0info=None, nz=500, ndm=1400,
                zmax=5.,dmmax=7000.,
                dopath=False, opstate=None, opt_params=None, opt_model=None):
    
    global PROFILED_PID
    pid = os.getpid()

    # If we haven't chosen a worker yet, choose this one
    if PROFILED_PID is None:
        PROFILED_PID = pid

    if pid == PROFILED_PID:
        profiler_output = f"worker_{pid}.prof"
        return cProfile.runctx(
            "calc_log_posterior(param_vals, state, params, surveys_sep, Pn, Pns, Pnr, "
            "pNreps, psnr, ptauw, pwb, log_halo, lin_host, ind_surveys, g0info, nz, ndm, zmax,dmmax, "
            "dopath, opstate, opt_params, opt_model)", 
            globals(), 
            locals(), 
            profiler_output
        )
    else:
        return calc_log_posterior(param_vals, state, params, surveys_sep, Pn, Pns, Pnr, 
                pNreps, psnr, ptauw, pwb, log_halo, lin_host, ind_surveys, g0info, nz, ndm, zmax,dmmax,
                dopath, opstate, opt_params, opt_model)

def calc_log_posterior(param_vals, state, params, surveys_sep, Pn=False, Pns=False, Pnr=False,
                pNreps=True, psnr=True, ptauw=False, pwb=False,
                log_halo=False, lin_host=False, ind_surveys=False, g0info=None, nz=500, ndm=1400,
                zmax=5.,dmmax=7000.,
                dopath=False, opstate=None, opt_params=None, opt_model=None):
    """Calculate log-posterior probability for a parameter vector.
    
    Alternate version of main function, which aims to initialise only one survet at a time
    
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
    Pns : bool, optional
        Include Poisson likelihood for non-repeating surveys. Default False.
    Pnr : bool, optional
        Include Poisson likelihood for repeating surveys. Default False.
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
    dopath:  bool, optional
        Include PATH host likelihoods according to optical model
    opstate: optical.state, optional
        State object of optical parameters
    opt_params:  dictionary, optional
        Optical parameter names, min and max values
    opt_model: optical.model, optional
        Optical model object, to modify
    
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
    
    # if including path, determine which parameter values are for zDM, which for optical properties
    if dopath:
        Nopt = len(opt_params)
        opt_param_vals = param_vals[-Nopt:]
        param_vals = param_vals[:-Nopt]
    
    # this iterates over zdm parameters
    for i, (key,val) in enumerate(params.items()):
        if param_vals[i] < val['min'] or param_vals[i] > val['max']:
            in_priors = False
            break

        if lin_host and key == 'lmean':
            param_dict[key] = np.log10(param_vals[i])
        else:
            param_dict[key] = param_vals[i]
    
    # performs same check for optical parameters, if used
    if dopath:
        opt_param_dict = {}
        for i, (key,val) in enumerate(opt_params.items()):
            if opt_param_vals[i] < val['min'] or opt_param_vals[i] > val['max']:
                in_priors = False
                break
        
            opt_param_dict[key] = opt_param_vals[i]

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
        # calculate all the likelihoods
        llsum = 0
        
        # minimise_const_only does the grid updating so we don't need to do it explicitly beforehand
        # In an MCMC analysis the parameter spaces are sampled throughout and hence with so many parameters
        # it is easy to reach impossible regions of the parameter space. This results in math errors
        # (log(0), log(negative), sqrt(negative), divide 0 etc.) and hence we assume that these math errors
        # correspond to an impossible region of the parameter space and so set ll = -inf
        #try:
        
        # Set state
        state.update_params(param_dict)
        
        # special updates
        if 'DMhalo' in param_dict:
            if log_halo:
                DMhalo = 10**param_dict['DMhalo']
                state.MW.DMhalo = DMhalo
        
        
        surveys = surveys_sep[0] + surveys_sep[1]
        
        # gets new zDM grid if F and H0 in the param_dict
        if 'H0' in param_dict or 'logF' in param_dict or g0info is None:
            cos.set_cosmology(state)
            cos.init_dist_measures()
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
                datdir=datdir,nz=nz,ndm=ndm,zmax=zmax,dmmax=dmmax)
            g0info = [zDMgrid, zvals,dmvals]
        
        if dopath:
            opstate.update_params(opt_param_dict)
            opt_model = opt.select_model(opstate) #initialise optical model
            # technically, we need one model wrapper per optical survey sensitivity
            # it doesn't need to be associated with a given FRB survey, or grid
        
        # holds expected rates and observed rates. We calculate these likelihoods later
        # so we don't have tohold all the grids in memory
        rs = []
        obs = []
        
        
        # Recreate grids every time, but not surveys, so must update survey params
        for i,s in enumerate(surveys):
            
            # reinitialises survey using updated state variables
            # just generally safest to do this. Noe that this does NOT
            # change 'analysis' variables which govern e.g. which FRBs
            # are or are not included in the sample
            # In theory, we could save time by checking if this needs to be done or not
            # But generally, it does need to be done
            s.reinit(state)
            
            # Initialise grids
            if dopath:
                wrappers = []
            
            
            ident = np.random.randint(0,1000000)
            sident = str(ident)
            
            
            if i < len(surveys_sep[0]):
                # generate normal zdm grid
                grids = mf.initialise_grids([s], zDMgrid, zvals, dmvals, state, wdist=True, repeaters=False)
                g = grids[0]
                if s.TOBS is not None and (Pn or Pns):
                    rs.append(np.sum(g.get_rates())*s.TOBS)
                    obs.append(s.NORM_FRB)
            else:
                # generates repeating zdm grid
                grids = mf.initialise_grids([s], zDMgrid, zvals, dmvals, state, wdist=True, repeaters=True)
                g = grids[0]
                # TOBS is already taken into account in the singles/repeater calculation.
                # But still need Pn or Pnr
                if Pn or Pnr:
                    rs.append(np.sum(g.get_exact_singles()))
                    rs.append(np.sum(g.get_exact_reps()))
                    obs.append(s.NORM_SINGLES)
                    obs.append(s.NORM_REPS)
            
            
            
            if dopath:
                w = opt.model_wrapper(opt_model,g.zvals)
            
            if dopath:
                ll,result = it.get_joint_path_zdm_likelihoods(g, s, w, Pn=False, pNreps=pNreps,
                                                        psnr=psnr,ptauw=ptauw,pwb=pwb,
                                                        return_all=True)
            else:
                ll = it.get_log_likelihood(g,s,Pn=False,pNreps=pNreps,psnr=psnr,ptauw=ptauw,pwb=pwb)
            
            
            debug=False
            if debug:
                # write state and resulting likelihood for later tests
                # generate prefix filename to allow files to be match
                rand = np.random.randint(low=0,high=999999999)
                statefile="Debug/"+str(rand)+".json"
                state.write(statefile)
                # now also write likelihood info
                lfile = "Debug/"+str(rand)+".dat"
                with open(lfile,'w') as out:
                    out.write(str(ll))
                
                # writes optical state
                if opstate is not None:
                    pathfile="Debug/path_"+str(rand)+".json"
                    opstate.write(pathfile)
            
            llsum += ll
            if ind_surveys:
                ll_list.append(ll)
        
        # keep this text here for debug purposes. use it to save states,
        # which can then be loaded to allow likelihood tests
        # generate a random number. This is because there is no way to order or label internal MCMC
        # states. So just generated a random number, and hope for no double-ups
        
        # state.write
        
        # Minimse the constant accross all surveys
        if (Pn or Pns or Pnr) and (len(obs) > 0):
            # dC is change in log constant from current number
            # llc is log probability
            obs = np.array(obs)
            rs = np.array(rs)
            dC, llC = it.minimise_const_only2(obs,rs)
            llsum += llC # adds Pn to llsum

    if np.isnan(llsum):
        print("llsum was NaN. Setting to -infinity", param_dict)    
        llsum = -np.inf
    
    # now clean up memory from energetics. Gammas are floating-point numbers,
    # and will never be re-used
    energetics.reset()
    
    if ind_surveys:
        return llsum, ll_list
    else:
        return llsum

#==============================================================================

def mcmc_runner(logpf, outfile, state, params, surveys, nwalkers=10, nsteps=100, nthreads=1,
                Pn=False, Pns=False, Pnr=False, pNreps=True, psnr=True, ptauw=False, pwb=False, log_halo=False,
                lin_host=False, ind_surveys=False, g0info=None, nz=500, ndm=1400, zmax=5.,dmmax=7000., reset=False,
                dopath=False, opstate=None, opt_params=None):
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
        Pns          (bool)         =   Include Pn for non-repeating surveys or not or not
        Pnr          (bool)         =   Include Pn for repeating surveys or not
        pNreps      (bool)          =   Include pNreps or not
        ptauw       (bool)          =   Include ptauw or not
        log_halo    (bool)          =   Use a log uniform prior on DMhalo
        ind_surveys (bool)          =   Return individual survey data
        g0info      (list)          =   List of [zDMgrid, zvals, DMvals] Passed to use as speedup if needed
        nz          (int)           =   Number of redshift (z) points ot use
        ndm         (int)           =   Number of DM values to use
        dopath      (bool)          =   Include PATH host likelihoods according to optical model
        opstate     (optical.state) =   State object of optical parameters
        opt_params  (dictionary)    =   Optical parameter names, min and max values
        opt_model   (optical.model) =   Optical model object, to modify
        
        
    Outputs:
        posterior_sample    (emcee.EnsembleSampler) =   Final sample
        outfile.h5          (HDF5 file)             =   HDF5 file containing the sampler
    """
        
    # Report zDM priors in sampling order.
    for key, val in params.items():
        print(key + " priors: " + str(val["min"]) + "," + str(val["max"]))

    # Draw physically valid zDM walker positions.
    starting_guesses = get_initial_walkers(state, params, nwalkers)

    if dopath:
        if opt_params is None:
            raise ValueError("opt_params must be supplied when dopath=True")

        optical_guesses = []

        for key, val in opt_params.items():
            print(key + " priors: " + str(val["min"]) + "," + str(val["max"]))

            optical_guesses.append(np.random.uniform(val["min"], val["max"], size=nwalkers))

        optical_guesses = np.asarray(optical_guesses).T

        starting_guesses = np.column_stack([starting_guesses, optical_guesses])

    ndim = starting_guesses.shape[1]
    
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

    allocated_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", nthreads))

    if nthreads > allocated_cpus:
        raise ValueError(f"Requested {nthreads} workers, but SLURM allocated only {allocated_cpus} CPUs.")

    print(f"Using {nthreads} MCMC workers from {allocated_cpus} allocated CPUs")

    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"

    Pool = mp.get_context("fork").Pool

    keys = params.keys()

    if (
        "Wlogmean" in keys
        or "Wlogsigma" in keys
        or "Slogmean" in keys
        or "Slogsigma" in keys
    ):
        state.scat.Sbackproject = True

    cos.set_cosmology(state)
    cos.init_dist_measures()

    def run_sampler(pool):
        sampler = emcee.EnsembleSampler(
            nwalkers,
            ndim,
            logpf,
            args=[state,params,surveys,Pn,Pns,Pnr,pNreps,psnr,ptauw,pwb,log_halo,lin_host,ind_surveys,g0info,nz,ndm,zmax,dmmax,dopath,opstate,opt_params],
            backend=backend,
            pool=pool,
        )

        if exists:
            sampler.run_mcmc(None, nsteps, progress=True)
        else:
            sampler.run_mcmc(starting_guesses, nsteps, progress=True)
            
        return sampler

    if nthreads == 1:
        sampler = run_sampler(None)
    else:
        with Pool(processes=nthreads, maxtasksperchild=10) as pool:
            sampler = run_sampler(pool)
            
    end = time.time()
    print("Total time taken: " + str(end - start))
    
    posterior_sample = sampler.get_chain()

    return posterior_sample

#==============================================================================
