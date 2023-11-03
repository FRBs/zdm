"""
File: MCMC.py
Author: Jordan Hoffmann
Date: 28/09/23
Purpose: 
    Contains functions used for MCMC runs of the zdm code. MCMC_wrap.py is the 
    main function which does the calling and this holds functions which do the 
    MCMC analysis.
"""

import numpy as np

import zdm.iteration as it

import emcee
import scipy.stats as st
import pickle
import time

import multiprocessing as mp

from zdm.misc_functions import *

#==============================================================================

def mcmc_likelihoods(outfile:str, walkers:int, steps:int, params, surveys, grids):
    posterior_sample = mcmc_runner(calc_log_posterior, outfile, params, surveys, grids, nwalkers=walkers,nsteps=steps)

    posterior_dict = {}
    for i,k in enumerate(params):
        posterior_dict[k] = posterior_sample[:,:,i]
    
    with open(outfile+'.pkl', 'wb') as fp:
        pickle.dump(posterior_dict, fp, pickle.HIGHEST_PROTOCOL)
    # np.save(outfile,posterior_sample)

    return

#==============================================================================

def calc_log_posterior(param_vals, params, surveys, grids):

    t0 = time.time()
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
            newC, llC = it.minimise_const_only(param_dict, grids, surveys)
            for g in grids:
                g.state.FRBdemo.lC = newC

            # calculate all the likelihoods
            llsum = 0
            for s, grid in zip(surveys, grids):
                # if DMhalo != None:
                #     s.init_DMEG(DMhalo)
                if 'DMhalo' in param_dict:
                    s.init_DMEG(param_dict['DMhalo'])

                llsum += get_likelihood(grid,s)

        except ValueError as e:
            print("ValueError, setting likelihood to -inf: " + str(e))
            llsum = -np.inf

    if np.isnan(llsum):
        print("llsum was NaN. Setting to -infinity", param_dict)    
        llsum = -np.inf
    
    print("Posterior calc time: " + str(time.time()-t0) + " seconds", flush=True)

    return llsum

#==============================================================================

def mcmc_runner(logpf, outfile, params, surveys, grids, nwalkers=10, nsteps=100):
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
    with mp.Pool(nwalkers) as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, logpf, args=[params, surveys, grids], backend=backend, pool=pool)
        sampler.run_mcmc(starting_guesses, nsteps, progress=True)
    end = time.time()
    print("Total time taken: " + str(end - start))
    
    posterior_sample = sampler.get_chain()

    return posterior_sample

#==============================================================================

def get_likelihood(grid, s):

    if s.nD==1:
        llsum1, lllist, expected = it.calc_likelihoods_1D(grid, s, psnr=True, dolist=1)
        llsum = llsum1

        # print(llsum, lllist)
    elif s.nD==2:
        llsum1, lllist, expected = it.calc_likelihoods_2D(grid, s, psnr=True, dolist=1)
        llsum = llsum1

        # print(llsum, lllist)
    elif s.nD==3:
        llsum1, lllist1, expected1 = it.calc_likelihoods_1D(grid, s, psnr=True, dolist=1)
        llsum2, lllist2, expected2 = it.calc_likelihoods_2D(grid, s, psnr=True, dolist=1, Pn=False)
        llsum = llsum1 + llsum2

        # print(llsum, lllist1, lllist2)
    else:
        print("Implementation is only completed for nD 1-3.")
        exit()

    return llsum

#==============================================================================