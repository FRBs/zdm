"""
File: MCMC.py
Author: Jordan Hoffmann
Date: 06/12/23
Purpose: 
    Contains functions used for MCMC runs of the zdm code. MCMC_wrap.py is the 
    main function which does the calling and this holds functions which do the 
    MCMC analysis.

    This function re-initialises the grids on every run
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
import os
#==============================================================================

def calc_log_posterior(param_vals, state, params, surveys_sep, Pn=False, pNreps=True, ptauw=False,
                log_halo=False, lin_host=False, ind_surveys=False, g0info=None):
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
        pNreps      (bool)          =   Include p(N repeaters) or not
        ptauw       (bool)          =   Include p(tau,w) or not
        log_halo    (bool)          =   Use a log uniform prior on DMhalo
        lin_host    (bool)          =   Use a linear uniform prior on host mean
        ind_surveys (bool)          =   Return likelihoods for each survey
        g0info      (list)          =   List of [zDMgrid, zvals, DMvals] Passed to use as speedup if needed
    
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
                zDMgrid, zvals,dmvals = mf.get_zdm_grid(
                    state, new=True, plot=False, method='analytic',
                    datdir=resource_filename('zdm', 'GridData'))
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
                pNreps=True, ptauw = False, log_halo=False, lin_host=False, ind_surveys=False, g0info=None):
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

    backend = emcee.backends.HDFBackend(outfile+'.h5')
    backend.reset(nwalkers, ndim)
    
    start = time.time()
    
    # may or may not be needed
    #os.environ["OMP_NUM_THREADS"] = "1"
    import multiprocessing as mp
    Pool = mp.get_context('fork').Pool
    
    
    
    with Pool() as pool: # could add mp.Pool(ntrheads=5) or Pool = None
        
        sampler = emcee.EnsembleSampler(nwalkers, ndim, logpf, args=[state, params, surveys, Pn, pNreps,
                                        ptauw, log_halo, lin_host, ind_surveys, g0info], backend=backend, pool=pool)
        sampler.run_mcmc(starting_guesses, nsteps, progress=True)
    end = time.time()
    print("Total time taken: " + str(end - start))
    
    posterior_sample = sampler.get_chain()

    return posterior_sample

#==============================================================================
