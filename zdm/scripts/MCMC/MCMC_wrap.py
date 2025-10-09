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
from pkg_resources import resource_filename
from zdm import survey
from zdm import cosmology as cos
from zdm import loading
from zdm import MCMC
from zdm import parameters
from zdm import misc_functions as mf

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
    parser.add_argument('--Nz', default=500, type=int, help="Number of z values")
    parser.add_argument('--Ndm', default=1400, type=int, help="Number of DM values")
    parser.add_argument('--zmax', default=5., type=int, help="Maximum z value")
    parser.add_argument('--dmmax', default=7000., type=int, help="Maximum DM value")
    parser.add_argument('--sdir', default=None, type=str, help="Directory containing surveys")
    parser.add_argument('--edir', default=None, type=str, help="Directory containing efficiency files")
    parser.add_argument('--outdir', default="", type=str, help="Output directory")
    parser.add_argument('--Pn', default=False, action='store_true', help="Include Pn")
    parser.add_argument('--pNreps', default=False, action='store_true', help="Include pNreps")
    parser.add_argument('--ptauw', default=False, action='store_true', help="Include p(tau,w)")
    parser.add_argument('--rand', default=False, action='store_true', help="Randomise DMG within uncertainty")
    parser.add_argument('--log_halo', default=False, action='store_true', help="Give a log prior on the halo instead of linear")
    parser.add_argument('--lin_host', default=False, action='store_true', help="Give a linear prior on host mean contribution")
    args = parser.parse_args()

    # Check correct flags are specified
    if args.pfile is None or args.opfile is None:
        print("-p and -o flags are required")
        exit()

    # Select from dictionary the necessary parameters to be changed        
    with open(args.pfile) as f:
        mcmc_dict = json.load(f)

    params = {k: mcmc_dict[k] for k in mcmc_dict["mcmc"]["parameter_order"]}

    state = parameters.State()
    state.set_astropy_cosmo(Planck18)
    state.update_params(mcmc_dict["config"])

    print("Config: ", mcmc_dict["config"])

    if args.Pn:
        print("Using Pn")
    if args.log_halo:
        print("Log prior on halo")
    if args.lin_host:
        print("Linear prior on host")

    # Initialise surveys
    surveys = [[], []]

    zDMgrid, zvals, dmvals = mf.get_zdm_grid(
                    state, new=True, plot=False, method='analytic', 
                    datdir=resource_filename('zdm', 'GridData'),
                    nz=args.Nz,ndm=args.Ndm,zmax=args.zmax,dmmax=args.dmmax)
    
    # pass this to starting iteration
    g0info = [zDMgrid,zvals,dmvals]
    
    # set z-dependent weights in surveys
    if ('Wlogmean' in params or 'Wlogsigma' in params or \
                    'Slogmean'  in params or 'Slogsigma' in params):
        survey_dict = {"WMETHOD": 3}
    else:
        survey_dict = None
    
    if args.files is not None:
        for survey_name in args.files:
            if "CHIME" in survey_name:
                use_dict = None
            else:
                use_dict=survey_dict
            s = survey.load_survey(survey_name, state, dmvals, zvals=zvals, survey_dict=use_dict,
                                sdir=args.sdir, edir=args.edir, rand_DMG=args.rand)
            surveys[0].append(s)
    
    if args.rep_surveys is not None:
        for survey_name in args.rep_surveys:
            if "CHIME" in survey_name:
                use_dict = None
            else:
                use_dict=survey_dict
                
            s = survey.load_survey(survey_name, state, dmvals, zvals=zvals, survey_dict=use_dict,
                                sdir=args.sdir, edir=args.edir, rand_DMG=args.rand)
            surveys[1].append(s)

    # Make output directory
    if args.outdir != "" and not os.path.exists(args.outdir):
        os.mkdir(args.outdir)

    MCMC.mcmc_runner(MCMC.calc_log_posterior, os.path.join(args.outdir, args.opfile), state, params, surveys, 
                         nwalkers=args.walkers, nsteps=args.steps, nthreads=args.nthreads, Pn=args.Pn, pNreps=args.pNreps,
                         ptauw = args.ptauw, log_halo=args.log_halo, lin_host=args.lin_host,g0info=g0info)

#==============================================================================

main()
