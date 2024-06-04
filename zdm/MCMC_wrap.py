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

from zdm import loading
from zdm import MCMC

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
    args = parser.parse_args()

    # Check correct flags are specified
    if args.pfile is None or args.opfile is None:
        print("-p and -o flags are required")
        exit()

    # Initialise surveys and grids
    if args.files is not None:
        surveys, grids = loading.surveys_and_grids(survey_names = args.files, repeaters=False, sdir=args.sdir, edir=args.edir)
    else:
        surveys = []
        grids = []
    
    if args.rep_surveys is not None:
        rep_surveys, rep_grids = loading.surveys_and_grids(survey_names = args.rep_surveys, repeaters=True, sdir=args.sdir, edir=args.edir)
        for s,g in zip(rep_surveys, rep_grids):
            surveys.append(s)
            grids.append(g)

    # Make output directory
    if args.outdir != "" and not os.path.exists(args.outdir):
        os.mkdir(args.outdir)
        
    with open(args.pfile) as f:
        mcmc_dict = json.load(f)

    # Select from dictionary the necessary parameters to be changed
    params = {k: mcmc_dict[k] for k in mcmc_dict['mcmc']['parameter_order']}

    MCMC.mcmc_runner(MCMC.calc_log_posterior, os.path.join(args.outdir, args.opfile), params, surveys, grids, nwalkers=args.walkers, nsteps=args.steps, nthreads=args.nthreads)

#==============================================================================
    
main()