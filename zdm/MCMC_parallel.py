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

# matplotlib.rcParams['image.interpolation'] = None
# import pcosmic
# import igm

# defaultsize=14
# ds=4
# font = {'family' : 'normal',
#         'weight' : 'normal',
#         'size'   : defaultsize}
# matplotlib.rc('font', **font)

global params
global surveys 
global grids 

def main(args):
    
    global params
    global surveys 
    global grids 
    names=args.files


    ############## Initialise cosmology ##############
    # Location for maximisation output
    outdir='mcmc/'
    
    #cos.set_cosmology(Omega_m=1.2) setup for cosmology
    cos.init_dist_measures()
    state = loading.set_state()
    
    #parser.add_argument(", help
    # get the grid of p(DM|z)
    zDMgrid, zvals,dmvals=get_zdm_grid(state,new=True,plot=False,method='analytic',save=True,datdir='MCMCData')
    # NOTE: if this is new, we also need new surveys and grids!
    
    ############## Initialise surveys ##############
    
    # # constants of beam method
    # thresh=0
    # method=2
    
    # # constants of intrinsic width distribution
    # Wlogmean=1.70267
    # Wlogsigma=0.899148
    # DMhalo=50
    
    prefix='mcmc'
    # Wbins=5
    # Wscale=3.5
    # Nbeams=[5,5,10]
    
    # Five surveys: we need to distinguish between those with and without a time normalisation
    if args.initialise == True:
        surveys = []
        for name in names:
            filename = 'data/Surveys/' + name
            s=survey.Survey(state, name, filename, dmvals)

            surveys.append(s)
            # pwidths,pprobs=survey.make_widths(s,Wlogmean,Wlogsigma,Wbins,scale=Wscale)
            # efficiencies=s.get_efficiency_from_wlist(dmvals,pwidths,pprobs)
        
        # FE1=survey.Survey()
        # FE1.process_survey_file('Surveys/CRAFT_class_I_and_II.dat')
        # FE1.init_DMEG(DMhalo)
        # FE1.init_beam(nbins=Nbeams[0],method=2,plot=False,thresh=thresh) # tells the survey to use the beam file
        # pwidths,pprobs=survey.make_widths(FE1,Wlogmean,Wlogsigma,Wbins,scale=Wscale)
        # efficiencies=FE1.get_efficiency_from_wlist(dmvals,pwidths,pprobs)
        
        # # load ICS data
        # ICS=survey.survey()
        # ICS.process_survey_file('Surveys/CRAFT_ICS.dat')
        # ICS.init_DMEG(DMhalo)
        # ICS.init_beam(nbins=Nbeams[1],method=2,plot=False,thresh=thresh) # tells the survey to use the beam file
        # pwidths,pprobs=survey.make_widths(ICS,Wlogmean,Wlogsigma,Wbins,scale=Wscale)
        # efficiencies=ICS.get_efficiency_from_wlist(dmvals,pwidths,pprobs)
        
        # # load Parkes data
        # p1=survey.survey()
        # p1.process_survey_file('Surveys/parkes_mb_class_I_and_II.dat')
        # p1.init_DMEG(DMhalo)
        # p1.init_beam(nbins=Nbeams[2],method=2,plot=False,thresh=thresh) # need more bins for Parkes!
        # pwidths,pprobs=survey.make_widths(p1,Wlogmean,Wlogsigma,Wbins,scale=Wscale)
        # efficiencies=p1.get_efficiency_from_wlist(dmvals,pwidths,pprobs)
    
        if not os.path.exists('Pickle/'):
            os.mkdir('Pickle/')

        with open('Pickle/'+prefix+'surveys.pkl', 'wb') as output:
            pickle.dump(surveys, output, pickle.HIGHEST_PROTOCOL)
            pickle.dump(names, output, pickle.HIGHEST_PROTOCOL)
    else:
        print("Loading ",'Pickle/'+prefix+'surveys.pkl')
        with open('Pickle/'+prefix+'surveys.pkl', 'rb') as infile:
            surveys=pickle.load(infile)
            names=pickle.load(infile)
    print("Initialised surveys ",names)
    
    
    # initial parameter values. SHOULD BE LOGSIGMA 0.75! (WAS 0.25!?!?!?)
    # these are meaningless btw - but the program is set up to require
    # a parameter set when first initialising grids
    # lEmin=30.
    # lEmax=42.
    # gamma=-0.7
    # alpha=1.5
    # sfr_n=1.
    # lmean=np.log10(50)
    # lsigma=0.5
    # C=0.
    # pset=[lEmin,lEmax,alpha,gamma,sfr_n,lmean,lsigma,C]
    
    # generates zdm grids for initial parameter set
    # when submitting a job, make sure this is all pre-generated once
    if args.initialise == True:
        grids=initialise_grids(surveys,zDMgrid, zvals,dmvals,state,wdist=True)
        with open('Pickle/'+prefix+'grids.pkl', 'wb') as output:
            pickle.dump(grids, output, pickle.HIGHEST_PROTOCOL)
    else:
        with open('Pickle/'+prefix+'grids.pkl', 'rb') as infile:
            grids=pickle.load(infile)
    print("Initialised grids")
    
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

def calc_log_posterior(param_vals):

    global params
    global surveys 
    global grids 
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

        # for grid in grids:
        #     grid.update(param_dict)

        it.minimise_const_only(param_dict, grids, surveys)

        llsum = 0
        for s, grid in zip(surveys, grids):
            if s.nD==1:
                llsum1, lllist, expected = it.calc_likelihoods_1D(grid, s, psnr=True, dolist=1)
                llsum += llsum1

                print(llsum, lllist)
            elif s.nD==2:
                llsum1, lllist, expected = it.calc_likelihoods_2D(grid, s, psnr=True, dolist=1)
                llsum += llsum1

                print(llsum, lllist)
            elif s.nD==3:
                llsum1, lllist1, expected1 = it.calc_likelihoods_1D(grid, s, psnr=True, dolist=1)
                llsum2, lllist2, expected2 = it.calc_likelihoods_2D(grid, s, psnr=True, dolist=1, Pn=False)
                llsum += llsum1 + llsum2

                print(llsum, lllist1, lllist2)
            else:
                print("Implementation is only completed for nD 1-3.")
                exit()
            

    if np.isnan(llsum):
        print("llsum was NaN. Setting to -infinity", param_dict)    
        llsum = -np.inf
    
    print("Posterior calc time: " + str(time.time()-t0) + " seconds", flush=True)

    return llsum

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

# def commasep(s):
#     if s == None:
#         return None
#     else:
#         return list(map(str, 
#                         s.split(',')))

# test for command-line arguments here
from zdm.misc_functions import *
import argparse
parser = argparse.ArgumentParser()
parser.add_argument(dest='files', nargs='+', help="Survey file names")
parser.add_argument('-i','--initialise', default=False, action='store_true', help="Initialise surveys")
parser.add_argument('-p','--pfile', default=None , type=str, help="File defining parameter ranges")
parser.add_argument('-o','--opfile', default=None, type=str, help="Output file for the data")
parser.add_argument('-w', '--walkers', default=20, type=int, help="Number of MCMC walkers")
parser.add_argument('-s', '--steps', default=100, type=int, help="Number of MCMC steps")
args = parser.parse_args()

# Check correct flags are specified
if args.pfile is None and args.opfile is None:
    if args.pfile is None or args.opfile is None:
        print("All flags (except -i optional) are required unless this is only for initialisation in which case only -i should be specified.")
        exit()

# t = time.process_time()
main(args)
# print("Total execution time: " + str(time.process_time()-t) + " seconds")