"""
This file shows how to estimate p(z) for an FRB with a given DM,
from a given survey.
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


# run with python est_pz.py --survey=MeerTRAPincoherent -b 500 -d "./" -o "Zest/" -i "H0_prior10" -n 100
# these are now defaults

#python est_pz.py --survey=MeerTRAPincoherent -b 500 -d "./" -o "Zest/" -i "H0_prior10" -n "100" -t "MTPC_1000" -D "1000." -l

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--infile',default="H0_prior10",type=str,help="Input HDF5 file containing MCMC results")
    parser.add_argument('-s', '--survey', default='MeerTRAPcoherent')
    parser.add_argument('-d', '--directory', default="../../zdm/scripts/MCMC/", type=str, help="Directory containing the HDF5 file. Defaults to zdm/mcmc/")
    parser.add_argument('-o', '--opdir', default="z2limit/", type=str, help="Output directory for files")
    parser.add_argument('-n', default=10, type=int, help="Number of parameter sets to calculate")
    parser.add_argument('-b', '--burnin', default=500, type=int, help="Burnin to discard from infile")
    parser.add_argument('-l', '--load', default=False, action='store_true', help="Load previous data?")
    parser.add_argument('-t', '--tag', default="", type=str, help="tag for output")
    parser.add_argument('-D', '--DM', default=2500, type=float, help="DMeg limit")
    
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
    
    load = args.load
    tag = args.tag
    DM = args.DM
    
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
    
    zvals = grid_vals[1]
    dmvals = grid_vals[2]
    
    # Load survey
    # If the survey is not specified, then the default is to use CRAFT_ICS_1300
    s = survey.load_survey(args.survey, state, grid_vals[2])

    # Calculate and average grids
    print("Calculating average grids...")
    if load:
        pdms = np.load(opdir+tag+"pdms.npy")
        zdists = np.load(opdir+tag+"zdists.npy")
    else:
        pdms,zdists = average_grids(samples, params, s, state, grid_vals,opdir,DMFRB=DM)
        np.save(opdir+tag+"pdms.npy",pdms)
        np.save(opdir+tag+"zdists.npy",zdists)
    
    bins = np.linspace(0,0.1,101)
    plt.figure()
    plt.xlabel("$p({\\rm DM}_{\\rm EG} > "+str(DM)+")$")
    plt.ylabel("Counts")
    plt.hist(pdms,bins=bins)
    plt.tight_layout()
    plt.savefig(opdir+tag+"histogram.png")
    plt.close()
    
    
    # p(z) plot
    Nsamples = len(pdms)
    plt.figure()
    avg = np.zeros([zvals.size])
    for i,zdist in enumerate(zdists):
        plt.plot(zvals,zdist,color="grey")
        avg += zdist
    avg /= Nsamples
    plt.plot(zvals,zdist,color="black",linewidth=3)
    plt.xlabel("Redshift z")
    plt.ylabel("p(z|DM_{\\rm EG}="+str(DM)+")")
    plt.tight_layout()
    plt.savefig(opdir+tag+"pz.png")
    plt.close()
    
    # cumulative p(z) plot
    
    Nsamples = len(pdms)
    plt.figure()
    cavg = np.zeros([zvals.size])
    for i,zdist in enumerate(zdists):
        czdist = np.cumsum(zdist)
        czdist /= czdist[-1]
        plt.plot(zvals,czdist,color="grey")
        cavg += czdist
    cavg /= Nsamples
    plt.plot(zvals,cavg,color="black",linewidth=3)
    plt.xlabel("Redshift z")
    plt.ylabel("p(z|DM_{\\rm EG}="+str(DM)+")")
    plt.tight_layout()
    plt.savefig(opdir+tag+"cum_pz.png")
    plt.close()
    
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

def average_grids(samples, params, s, state, grid_vals, opdir, DMFRB = 2500, pmin=0.01,log_halo=False,verbose=False):

    zDMgrid = grid_vals[0]
    zvals = grid_vals[1]
    dmvals = grid_vals[2]

    av_rates = np.zeros([len(zvals), len(dmvals)])
    rates = []
    
    pdms=[]
    zdists=[]
    
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
        iDM = np.where(grid.dmvals < DMFRB)[0][-1]
        
        # get redshift distribution
        zdist = grid.rates[:,iDM]
        zdist /= np.sum(zdist)
        
        # DM distribution
        pdm = np.sum(grid.rates,axis=0)
        
        # probability of that DM or higher
        prob = np.sum(pdm[iDM:])/np.sum(pdm)
        
        pdms.append(prob)
        zdists.append(zdist)

    return pdms,zdists

main()
