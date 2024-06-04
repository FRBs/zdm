"""
File: MCMC_wrap.py
Author: Jordan Hoffmann
Date: 28/09/23
Purpose: 
    Wrapper file to run MCMC analysis for zdm code. Handles command line
    parameters and loading of surveys, grids and parameters. Actual MCMC
    analysis functions are in MCMC.py.

    scripts/run_mcmc.slurm contains an example sbatch script.
"""

import argparse
import os

import numpy as np

from astropy.cosmology import Planck18

from zdm import survey
from zdm import cosmology as cos
from zdm import loading
from zdm import MCMC2
from zdm import parameters

import pickle
import json

#==============================================================================

def main():
    """
    Handles the setup for MCMC runs. This involves reading / creating the
    surveys and grids, reading the parameters and prior ranges and then 
    beginning the MCMC run.

    Inputs:
        args = Command line parameters
    
    Outputs:
        None
    """
    
    # Parsing command line
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--files', default=None, nargs='+', type=str, help="Survey file names")
    parser.add_argument('-r', '--rep_surveys', default=None, nargs='+', type=str, help="Surveys to consider repeaters in")
    parser.add_argument('-p','--pfile', default=None , type=str, help="File defining parameter ranges")
    parser.add_argument('-o','--opfile', default=None, type=str, help="Output file for the data")
    parser.add_argument('-w', '--walkers', default=20, type=int, help="Number of MCMC walkers")
    parser.add_argument('-s', '--steps', default=100, type=int, help="Number of MCMC steps")
    parser.add_argument('-n', '--nthreads', default=1, type=int, help="Number of threads")
    parser.add_argument('--sdir', default=None, type=str, help="Directory containing surveys")
    parser.add_argument('--edir', default=None, type=str, help="Directory containing efficiency files")
    parser.add_argument('--outdir', default="", type=str, help="Output directory")
    parser.add_argument('--Pn', default=False, action='store_true', help="Include Pn")
    parser.add_argument('--log_halo', default=False, action='store_true', help="Give a log prior on the halo instead of linear")
    args = parser.parse_args()

    # Check correct flags are specified
    if args.pfile is None or args.opfile is None:
        print("-p and -o flags are required")
        exit()

    # Initialise surveys
    surveys = [[], []]
    state = parameters.State()

    grid_params = {}
    grid_params['dmmax'] = 7000.0
    grid_params['ndm'] = 1400
    grid_params['nz'] = 500
    ddm = grid_params['dmmax'] / grid_params['ndm']
    dmvals = (np.arange(grid_params['ndm']) + 1) * ddm
    
    if args.files is not None:
        for survey_name in args.files:
            s = survey.load_survey(survey_name, state, dmvals, 
                                sdir=args.sdir, edir=args.edir)
            surveys[0].append(s)
    
    if args.rep_surveys is not None:
        for survey_name in args.rep_surveys:
            s = survey.load_survey(survey_name, state, dmvals, 
                                sdir=args.sdir, edir=args.edir)
            surveys[1].append(s)

    # Make output directory
    if args.outdir != "" and not os.path.exists(args.outdir):
        os.mkdir(args.outdir)
        
    with open(args.pfile) as f:
        mcmc_dict = json.load(f)

    # Select from dictionary the necessary parameters to be changed
    params = {k: mcmc_dict[k] for k in mcmc_dict['mcmc']['parameter_order']}

    state = parameters.State()
    state.set_astropy_cosmo(Planck18)
    state.update_params(mcmc_dict['config'])

    print("Config: ", mcmc_dict['config'])

    if args.Pn:
        print("Using Pn")
    if args.log_halo:
        print("Log prior on halo")

    MCMC2.mcmc_runner(MCMC2.calc_log_posterior, os.path.join(args.outdir, args.opfile), state, params, surveys, grid_params, 
                nwalkers=args.walkers, nsteps=args.steps, nthreads=args.nthreads, Pn=args.Pn, log_halo=args.log_halo)

#==============================================================================

main()