"""
This file demonstrates how to use MCMC output to sample random walkers,
and generate zDM grids which include the uncertainty in best-fit parameters.

It requires an MCMC output to sample from - see the directory data/MCMC/
for examples.

Example input:
python average_grids.py H0_prior10 -s MeerTRAPincoherent -n 10 -d "./"
"""


import emcee
import argparse
from pkg_resources import resource_filename
import os
import json
from zdm import survey
import numpy as np
from zdm import parameters
from astropy.cosmology import Planck18
from zdm import misc_functions as mf
from zdm import grid as zdm_grid
import copy
from zdm import pcosmic
from matplotlib import pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(dest='infile',type=str,help="Input HDF5 file containing MCMC results")
    parser.add_argument('-s', '--survey', default='CRAFT_ICS_1300')
    parser.add_argument('-d', '--directory', default=None, type=str, help="Directory containing the HDF5 file. Defaults to zdm/mcmc/")
    parser.add_argument('-o', '--opdir', default="", type=str, help="Output directory for files")
    parser.add_argument('-n', default=10, type=int, help="Number of parameter sets to calculate")
    parser.add_argument('-b', '--burnin', default=500, type=int, help="Burnin to discard from infile")
    args = parser.parse_args()

    if args.directory == None:
        args.directory = resource_filename('zdm', 'mcmc')

    return args

def main():
    # Parse command line arguments
    args = parse_args()
    # Read MCMC output file
    samples, params, config = get_samples(args)
    
    if args.opdir != "":
        opdir = args.opdir+"/"
        if not os.path.exists(opdir):
            os.mkdir(opdir)
        
    else:
        opdir = "./"
    dmmax = 7000.0
    ndm = 1400
    nz = 500

    # Set up state
    state = parameters.State()
    state.set_astropy_cosmo(Planck18)
    state.update_params(config)
    
    # We explicitly load this to allow the "average grids" routine
    # to more speedily calculate many grids
    grid_vals = mf.get_zdm_grid(
        state, new=True, plot=False, method='analytic', 
        nz=nz, ndm=ndm, dmmax=dmmax,
        datdir=resource_filename('zdm', 'GridData'))

    # Load survey
    # If the survey is not specified, then the default is to use CRAFT_ICS_1300
    s = survey.load_survey(args.survey, state, grid_vals[2])

    # Calculate and average grids
    print("Calculating average grids...")
    rates, av_rates = average_grids(samples, params, s, state, grid_vals)
    
    
    
    # Do plotting
    mf.plot_grid_2(
        av_rates,
        grid_vals[1],
        grid_vals[2],
        name=opdir+"averaged_grid.pdf",
        norm=3,
        log=True,
        label='$\\log_{10} p({\\rm DM}_{\\rm IGM} + {\\rm DM}_{\\rm host},z)$ [a.u.]',
        project=True,
        ylabel='${\\rm DM}_{\\rm IGM} + {\\rm DM}_{\\rm host}$',
        zmax=3,DMmax=3000 #,
        #Aconts=[0.01, 0.1, 0.5]
        )
    
    for i, rate in enumerate(rates):
        mf.plot_grid_2(
            rate,
            grid_vals[1],
            grid_vals[2],
            name=opdir+"grid" + str(i) + ".pdf",
            norm=3,
            log=True,
            label='$\\log_{10} p({\\rm DM}_{\\rm IGM} + {\\rm DM}_{\\rm host},z)$ [a.u.]',
            project=True,
            ylabel='${\\rm DM}_{\\rm IGM} + {\\rm DM}_{\\rm host}$',
            zmax=3,DMmax=3000#,
            #Aconts=[0.01, 0.1, 0.5]
            )
        
    
    # do DM porojection
    fig, ax1 = plt.subplots()
    fig, ax2 = plt.subplots()
    
    ax1.set_xlabel('z')
    ax2.set_xlabel('z')
    ax1.set_ylabel('p(z)')
    ax2.set_ylabel('cumulative p(z)')
    
    for i, rate in enumerate(rates):
        pz = np.sum(rate,axis=1)
        cpz = np.cumsum(pz)
        cpz /= cpz[-1]
        ax1.plot(grid_vals[1],pz)
        ax2.plot(grid_vals[1],cpz)
       
    pz = np.sum(av_rates,axis=1)
    ax1.plot(grid_vals[1],pz,label="mean", color="black", linewidth=3)
    
    cpz = np.cumsum(pz)
    cpz /= cpz[-1]
    ax2.plot(grid_vals[1],cpz,label="mean", color="black", linewidth=3)
    
    plt.sca(ax1)
    plt.ylim(0,None)
    plt.legend()
    plt.tight_layout()
    plt.savefig(opdir+"z_projection.pdf")
    plt.close()
    
    plt.sca(ax2)
    plt.ylim(0,1)
    plt.legend()
    plt.tight_layout()
    plt.savefig(opdir+"cumulative_z_projection.pdf")
    plt.close()
        
    # do DM porojection
    fig, ax1 = plt.subplots()
    fig, ax2 = plt.subplots()
    
    for i, rate in enumerate(rates):
        pdm = np.sum(rate,axis=0)
        cpdm = np.cumsum(pdm)
        cpdm /= cpdm[-1]
        ax1.plot(grid_vals[2],pdm)
        ax2.plot(grid_vals[2],cpdm)
    
    pdm = np.sum(av_rates,axis=0)
    ax1.plot(grid_vals[2],pdm,label="mean", color="black", linewidth=3)
    cpdm = np.cumsum(pdm)
    cpdm /= cpdm[-1]
    ax2.plot(grid_vals[2],cpdm,label="mean", color="black", linewidth=3)
    
    plt.sca(ax1)
    plt.ylim(0,None)
    plt.xlabel('DM')
    plt.ylabel('p(DM)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(opdir+"dm_projection.pdf")
    plt.close()
    
    plt.sca(ax2)
    plt.ylim(0,None)
    plt.xlabel('DM')
    plt.ylabel('p(DM)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(opdir+"cumulative_dm_projection.pdf")
    plt.close()
    
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

    return sample, params, config

def average_grids(samples, params, s, state, grid_vals, log_halo=False,verbose=False):

    zDMgrid = grid_vals[0]
    zvals = grid_vals[1]
    dmvals = grid_vals[2]

    av_rates = np.zeros([len(zvals), len(dmvals)])
    rates = []
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
        
        av_rates += grid.rates
        rates.append(grid.rates)
        
    av_rates = av_rates / samples.shape[1]
    return rates, av_rates

main()
