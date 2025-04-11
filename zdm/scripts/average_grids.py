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

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(dest='infile',type=str,help="Input HDF5 file containing MCMC results")
    parser.add_argument('-s', '--survey', default='CRAFT_ICS_1300')
    parser.add_argument('-d', '--directory', default=None, type=str, help="Directory containing the HDF5 file. Defaults to zdm/mcmc/")
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

    dmmax = 7000.0
    ndm = 1400
    nz = 500

    # Set up state
    state = parameters.State()
    state.set_astropy_cosmo(Planck18)
    state.update_params(config)

    grid_vals = mf.get_zdm_grid(
        state, new=True, plot=False, method='analytic', 
        nz=nz, ndm=ndm, dmmax=dmmax,
        datdir=resource_filename('zdm', 'GridData'))

    # Load survey
    # If the survey is not specified, then the default is to use CRAFT_ICS_1300
    s = survey.load_survey(args.survey, state, grid_vals[2])

    # Calculate and average grids
    rates, av_rates = average_grids(samples, params, s, state, grid_vals)

    # Do plotting
    mf.plot_grid_2(
        av_rates,
        grid_vals[1],
        grid_vals[2],
        name="averaged_grid.pdf",
        norm=3,
        log=True,
        label='$\\log_{10} p({\\rm DM}_{\\rm IGM} + {\\rm DM}_{\\rm host},z)$ [a.u.]',
        project=True,
        ylabel='${\\rm DM}_{\\rm IGM} + {\\rm DM}_{\\rm host}$',
        zmax=3,DMmax=3000,
        Aconts=[0.01, 0.1, 0.5]
        )

    for i, rate in enumerate(rates):
        mf.plot_grid_2(
            rate,
            grid_vals[1],
            grid_vals[2],
            name="grid" + str(i) + ".pdf",
            norm=3,
            log=True,
            label='$\\log_{10} p({\\rm DM}_{\\rm IGM} + {\\rm DM}_{\\rm host},z)$ [a.u.]',
            project=True,
            ylabel='${\\rm DM}_{\\rm IGM} + {\\rm DM}_{\\rm host}$',
            zmax=3,DMmax=3000,
            Aconts=[0.01, 0.1, 0.5]
            )

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

def average_grids(samples, params, s, state, grid_vals, log_halo=False):

    zDMgrid = grid_vals[0]
    zvals = grid_vals[1]
    dmvals = grid_vals[2]

    av_rates = np.zeros([len(zvals), len(dmvals)])
    rates = []
    # Load a grid for each parameter set
    for i in range(samples.shape[0]):
        vals = samples[i,:]

        dict = {}
        for j in range(len(vals)):
            dict[params[j]] = vals[j]

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