"""
Iterates over the Hoffmann et al parameter set to estimate how many
get ruled out by the z=2 scenario.
"""


import emcee
import argparse
import importlib.resources as resources
import os
import json
from zdm import survey
import numpy as np
from zdm import parameters
from astropy.cosmology import Planck18
from zdm import figures
from zdm import misc_functions as mf
from zdm import grid as zdm_grid
import copy
from zdm import pcosmic
from matplotlib import pyplot as plt
import corner


# run with python average_grids.py --survey=MeerTRAPcoherent -b 500 -d "../../zdm/scripts/MCMC/" -o "z2limit/" -i "H0_prior10"
# these are now defaults

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--infile',default="H0_prior10",type=str,help="Input HDF5 file containing MCMC results")
    parser.add_argument('-s', '--survey', default='MeerTRAPcoherent')
    parser.add_argument('-d', '--directory', default="../../zdm/scripts/MCMC/", type=str, help="Directory containing the HDF5 file. Defaults to zdm/mcmc/")
    parser.add_argument('-o', '--opdir', default="z2limit/", type=str, help="Output directory for files")
    parser.add_argument('-n', default=100, type=int, help="Number of parameter sets to calculate")
    parser.add_argument('-b', '--burnin', default=500, type=int, help="Burnin to discard from infile")
    args = parser.parse_args()

    if args.directory == None:
        args.directory = resources.files('zdm').joinpath('mcmc')

    return args

def main():
    # Parse command line arguments
    args = parse_args()
    # Read MCMC output file
    samples, params, config,labels = get_samples(args)
    
    if args.opdir != "":
        opdir = args.opdir+"/"
        if not os.path.exists(opdir):
            os.mkdir(opdir)
        
    else:
        opdir = "./"
    dmmax = 7000.0
    #ndm = 1400
    #nz = 500
    dmmax = 4000
    zmax = 4
    ndm = 400
    nz = 200
    
    load = False
    
    # Set up state
    state = parameters.State()
    state.set_astropy_cosmo(Planck18)
    if config is not None:
        state.update_params(config)
    
    # We explicitly load this to allow the "average grids" routine
    # to more speedily calculate many grids. This only works because we
    # are not changing H0 or F
    grid_vals = mf.get_zdm_grid(
        state, new=True, plot=False, method='analytic', 
        nz=nz, ndm=ndm, dmmax=dmmax, zmax = zmax,
        datdir=resources.files('zdm').joinpath('GridData'))
    
    # Load survey
    # If the survey is not specified, then the default is to use CRAFT_ICS_1300
    s = survey.load_survey(args.survey, state, grid_vals[2])

    # Calculate and average grids
    print("Calculating average grids...")
    if load:
        good_samples = np.load(opdir+"good_samples.npy")
        bad_samples = np.load(opdir+"bad_samples.npy")
    else:
        good_samples,bad_samples = average_grids(samples, params, s, state, grid_vals,opdir)
    
    do_cornerplot(good_samples,labels,opdir+"passing_parameters.png")
    if len(bad_samples) == 0:
        print("Nothing is bad!")
    else:
        print("Bad has length ",len(bad_samples))

def do_cornerplot(samples,labels,savefile):
    """
    Args:
        samples (np.array):
            Array of samples of dimensions nwalkers x nparameters.
            Used to create the cornerplot
        labels (list of strings):
            Text labels for each variable, in order of appearance in samples
        savefile (string): filename for saved cornerplot
            
    """
    fig = plt.figure(figsize=(12,12))

    titles = ['' for i in range(samples.shape[1])] # no titles for the plots
    truths = None
    corner.corner(samples,labels=labels, show_titles=True, titles=titles, 
                  fig=fig,title_kwargs={"fontsize": 15},label_kwargs={"fontsize": 15}, 
                  quantiles=[0.16,0.5,0.84], truths=truths);
    
    plt.savefig(savefile)
    
def get_samples(args):
    # Read in the MCMC results without the burnin
    infile = os.path.join(args.directory, args.infile)
    reader = emcee.backends.HDFBackend(infile + '.h5')
    sample = reader.get_chain(discard=args.burnin, flat=True)

    # Thin the results
    step = len(sample) // args.n
    sample = sample[::step,:]

    # Get the corresponding parameters
    # If there is a corresponding .out file, it will contain all the necessary information that was used during that run,
    # otherwise parameters must be specified manually
    if os.path.exists(infile + '.out'):
        with open(infile + '.out', 'r') as f:
            # Get configuration
            line = f.readline()
            
            while not line.startswith('Config') and line:
                line = f.readline()
            if not line:
                raise ValueError("No 'Config' line found in the .out file.")
            config = json.loads(line[9:].replace("'", '"'))

            # Read down to parameter lines
            while ('priors' not in line) and line:
                line = f.readline()
            
            # Read parameters
            params = []
            while ('priors' in line) and line:
                vals = line.split()
                params.append(vals[0])
                line = f.readline()

    # If there is no .out file, then the parameters must be specified manually
    else:
        params = ["sfr_n", "alpha", "lmean", "lsigma", "lEmax", "lEmin", "gamma", "H0"]
        labels = [r"$n_{\rm sfr}$", r"$\alpha$", r"$\log_{10}\mu_h$", r"$\log_{10}\sigma_h$",
                    r"$\log_{10} E_{\rm max}$",r"$\log_{10}E_{\rm min}$", r"$\gamma$", r"$H_0$"]
        config = None
    
    return sample, params, config, labels

def average_grids(samples, params, s, state, grid_vals, opdir, zFRB = 2.148, pmin=0.01,log_halo=False,verbose=False):

    zDMgrid = grid_vals[0]
    zvals = grid_vals[1]
    dmvals = grid_vals[2]

    av_rates = np.zeros([len(zvals), len(dmvals)])
    rates = []
    
    good_samples = []
    bad_samples = []
    
    # Load a grid for each parameter set
    for i in range(samples.shape[0]):
        if verbose:
            print("Sampling parameter set ",i,". Params: ")
        vals = samples[i,:]
        
        
        dict = {}
        for j in range(len(vals)):
            dict[params[j]] = vals[j]
            if verbose:
                print("     ",params[j],": ",vals[j])
            
        state.update_params(dict)
        if 'DMhalo' in params:
            if log_halo:
                DMhalo = 10**dict['DMhalo']
            else:
                DMhalo = dict['DMhalo']
            s.init_DMEG(DMhalo)
            s.get_efficiency_from_wlist(s.DMlist,s.wlist,s.wplist,model=s.meta['WBIAS']) 
        
        mask = pcosmic.get_dm_mask(
            dmvals, (state.host.lmean, state.host.lsigma), zvals, plot=True
        )
        grid = zdm_grid.Grid(
            s, copy.deepcopy(state), zDMgrid, zvals, dmvals, mask, wdist=True
        )
        
        # we take all grid values where z_grid < z_FRB, but we use the highest one
        # as "including" the FRB, so that we round conservatively
        iz = np.where(grid.zvals < zFRB)[0]
        izmin = iz[-1]
        
        # get redshift distribution
        zdist = np.sum(grid.rates,axis=1)
        zdist = np.cumsum(zdist)
        zdist /= zdist[-1]
        
        # probability of being larger
        pz = 1.-zdist[izmin]
        print("Found a cumulative pz! Prob of being larger is ",pz)
        if pz < pmin:
            # this parameter set rules out the detection, and is incompatible
            bad_samples.append(samples[i])
        else:
            # it's all OK!!! It has enough probability at high z
            good_samples.append(samples[i])
        
        #av_rates += grid.rates
        #rates.append(grid.rates)
    
    # we now have a list of good and bad samples
       
    good = np.array(good_samples)
    bad = np.array(bad_samples)
    np.save(opdir+"good_samples.npy",good)
    np.save(opdir+"bad_samples.npy",bad)
    bad=[]
    #av_rates = av_rates / samples.shape[1]
    return good,bad

main()
