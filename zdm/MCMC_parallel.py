######
# first run this to generate surveys and parameter sets, by 
# setting NewSurveys=True NewGrids=True
# Then set these to False and run with command line arguments
# to generate *many* outputs
#####

# It should be possible to remove all the matplotlib calls from this
# but in the current implementation it is not removed.
import argparse
import numpy as np
import os
# import matplotlib

from zdm import survey
# from zdm import parameters
from zdm import cosmology as cos
from zdm.craco import loading

import zdm.iteration as it

import emcee
import scipy.stats as st
import pickle
import json
import time

import multiprocessing as mp

from zdm.misc_functions import *
import argparse
parser = argparse.ArgumentParser()
parser.add_argument(dest='files', nargs='+', help="Survey file names")
parser.add_argument('-i','--initialise', default=None, type=str, help="Prefix used to initialise survey")
parser.add_argument('-p','--pfile', default=None , type=str, help="File defining parameter ranges")
parser.add_argument('-o','--opfile', default=None, type=str, help="Output file for the data")
parser.add_argument('-w', '--walkers', default=20, type=int, help="Number of MCMC walkers")
parser.add_argument('-s', '--steps', default=100, type=int, help="Number of MCMC steps")
args = parser.parse_args()

global params
global surveys 
global grids 

# Check correct flags are specified
if args.pfile is None or args.opfile is None:
    if not (args.pfile is None and args.opfile is None):
        print("All flags (except -i optional) are required unless this is only for initialisation in which case only -i should be specified.")
        exit()

#==============================================================================

def main(args):
    
    global params
    global surveys 
    global grids 
    names=args.files
    prefix=args.initialise


    ############## Initialise cosmology ##############
    # Location for maximisation output
    outdir='mcmc/'

    cos.init_dist_measures()
    state = loading.set_state()
    
    # get the grid of p(DM|z)
    zDMgrid, zvals,dmvals=get_zdm_grid(state,new=True,plot=False,method='analytic',save=True,datdir='MCMCData')
    
    ############## Initialise surveys ##############

    if not os.path.exists('Pickle/'+prefix+'surveys.pkl'):
        # Initialise surveys
        surveys = []
        for name in names:
            filename = 'data/Surveys/' + name
            s=survey.Survey(state, name, filename, dmvals)

            surveys.append(s)
    
        # Initialise grids
        grids=initialise_grids(surveys,zDMgrid, zvals,dmvals,state,wdist=True)

        # Save surveys / grids in pickle format
        if prefix != None:
            if not os.path.exists('Pickle/'):
                os.mkdir('Pickle/')

            # Save surveys
            print("Saving ",'Pickle/'+prefix+'surveys.pkl')
            with open('Pickle/'+prefix+'surveys.pkl', 'wb') as output:
                pickle.dump(surveys, output, pickle.HIGHEST_PROTOCOL)
                pickle.dump(names, output, pickle.HIGHEST_PROTOCOL)

            # Save grids
            print("Saving ",'Pickle/'+prefix+'grids.pkl')
            with open('Pickle/'+prefix+'grids.pkl', 'wb') as output:
                pickle.dump(grids, output, pickle.HIGHEST_PROTOCOL)
    else:
        print("Loading ",'Pickle/'+prefix+'surveys.pkl')
        # Load surveys
        with open('Pickle/'+prefix+'surveys.pkl', 'rb') as infile:
            surveys=pickle.load(infile)
            names=pickle.load(infile)

        # Load grids
        with open('Pickle/'+prefix+'grids.pkl', 'rb') as infile:
            grids=pickle.load(infile)

    print("Initialised grids and surveys ",names)
    
    # If not initialising only, run mcmc
    if args.pfile is not None and args.opfile is not None:
        if not os.path.exists(outdir):
            os.mkdir(outdir)
        
        with open(args.pfile) as f:
            mcmc_dict = json.load(f)

        # Select from dictionary the necessary parameters to be changed
        params = {k: mcmc_dict[k] for k in mcmc_dict['mcmc']['parameter_order']}

        mcmc_likelihoods(outdir + args.opfile, args.walkers, args.steps)
    else:
        print("No parameter or output file provided. Assuming only initialising and no MCMC running is done.")

#==============================================================================

def mcmc_likelihoods(outfile:str, walkers:int, steps:int):

    global params
    global surveys 
    global grids 
    posterior_sample = mcmc_runner(calc_log_posterior, outfile, nwalkers=walkers,nsteps=steps)

    posterior_dict = {}
    for i,k in enumerate(params):
        posterior_dict[k] = posterior_sample[:,:,i]
    
    with open(outfile+'.pkl', 'wb') as fp:
        pickle.dump(posterior_dict, fp, pickle.HIGHEST_PROTOCOL)
    # np.save(outfile,posterior_sample)

    return

#==============================================================================

def calc_log_posterior(param_vals):

    global params
    global surveys 
    global grids 
    t0 = time.time()
    # Can use likelihoods instead of posteriors because we only use uniform priors which just changes normalisation of posterior 
    # given every value is in the correct range. If any value is not in the correct range, log posterior is -inf
    in_priors = True
    param_dict = {}
    uDMG=0.0
    DMhalo=None
    for i, (key,val) in enumerate(params.items()):
        if param_vals[i] < val['min'] or param_vals[i] > val['max']:
            in_priors = False
            break
        # if key == 'uDMG':
        #     uDMG = param_vals[i]
        # if key == 'DMhalo':
        #     DMhalo=val
        # else:
        param_dict[key] = param_vals[i]

    if in_priors == False:
        llsum = -np.inf
    else:

        # for grid in grids:
        #     grid.update(param_dict)

        # minimise_const_only does the grid updating so we don't need to do it explicitly beforehand
        # print(param_dict)
        try:
            it.minimise_const_only(param_dict, grids, surveys)

            # calculate all the likelihoods
            llsum = 0
            for s, grid in zip(surveys, grids):
                # if DMhalo != None:
                #     s.init_DMEG(DMhalo)
                if 'DMhalo' in param_dict:
                    s.init_DMEG(param_dict['DMhalo'])

                llsum += get_likelihood(grid,s)
            
                # if 'uDMG' in param_dict:
                #     # x = np.linspace(st.norm.ppf(0.01), st.norm.ppf(0.99), 10)
                    
                #     n_samps = 100
                #     mus = s.DMGs

                #     DM_ISMs = []
                #     for mu in mus:
                #         DM_ISMs.append(np.random.normal(mu, mu*param_dict['uDMG'], n_samps))

                #     DM_ISMs = np.array(DM_ISMs)
                #     print(DM_ISMs.shape)

                #     for j in range(n_samps):
                #         setattr(s, "DMGs", DM_ISMs[:,j])
                #         llsum += get_likelihood(grid,s) / n_samps

                #     setattr(s, "DMGs", mus)

                # else:
                #     llsum += get_likelihood(grid,s)
        except ValueError as e:
            print("ValueError, setting likelihood to -inf: " + str(e))
            llsum = -np.inf

    if np.isnan(llsum):
        print("llsum was NaN. Setting to -infinity", param_dict)    
        llsum = -np.inf
    
    print("Posterior calc time: " + str(time.time()-t0) + " seconds", flush=True)

    return llsum

#==============================================================================

def mcmc_runner(logpf, outfile, nwalkers=10, nsteps=100):
    global params
    global surveys 
    global grids 
    ndim = len(params)
    starting_guesses = []

    # Produce starting guesses for each parameter
    for key,val in params.items():
        starting_guesses.append(st.uniform(loc=val['min'], scale=val['max']-val['min']).rvs(size=[nwalkers]))
        print(key + " priors: " + str(val['min']) + "," + str(val['max']))
    
    starting_guesses = np.array(starting_guesses).T

    backend = emcee.backends.HDFBackend(outfile+'.h5')
    backend.reset(nwalkers, ndim)

    with mp.Pool(nwalkers) as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, logpf, backend=backend, pool=pool)
        start = time.time()
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

# t = time.process_time()
main(args)
# print("Total execution time: " + str(time.process_time()-t) + " seconds")