"""
Evaluates the likelihood of a slice through H0 for various surveys
"""

import argparse
import numpy as np
import os

from zdm import figures
from zdm import iteration as it

from zdm import parameters
from zdm import repeat_grid as zdm_repeat_grid
from zdm import MCMC
from zdm import survey
from zdm import states
from astropy.cosmology import Planck18

from numpy import random
import matplotlib.pyplot as plt
import time

def main():
    """
    run with:
    python run_H0_slice.py -n 10 --min=50 --max=100 -f CRACO/Smeared CRACO/zFrac CRACO/Spectroscopic CRACO/Smeared_and_zFrac MeerTRAP/Smeared MeerTRAP/zFrac MeerTRAP/Spectroscopic MeerTRAP/Smeared_and_zFrac
    
    """
    t0 = time.time()
    parser = argparse.ArgumentParser()
    #parser.add_argument(dest='param',type=str,help="Parameter to do the slice in")
    parser.add_argument('--min',type=float,help="Min value")
    parser.add_argument('--max',type=float,help="Max value")
    parser.add_argument('-f', '--files', default=None, nargs='+', type=str, help="Survey file names")
    parser.add_argument('-n',dest='n',type=int,default=50,help="Number of values")   
    # parser.add_argument('-r',dest='repeaters',default=False,action='store_true',help="Surveys are repeater surveys")   
    args = parser.parse_args()

    vals = np.linspace(args.min, args.max, args.n)

    # Set state
    state=states.load_state(case="HoffmannHalo25",scat=None,rep=None)
    
    grid_params = {}
    grid_params['dmmax'] = 7000.0
    grid_params['ndm'] = 1400
    grid_params['nz'] = 500
    ddm = grid_params['dmmax'] / grid_params['ndm']
    dmvals = (np.arange(grid_params['ndm']) + 1) * ddm
    
    # Initialise surveys
    surveys = []
    if args.files is not None:
        for survey_name in args.files:
            print("Loading survey ",survey_name)
            s = survey.load_survey(survey_name, state, dmvals,sdir="./")
            surveys.append(s)
    

    # state.update_param('halo_method', 1)
    # state.update_param(args.param, vals[0])
    
    outdir = 'H0/'
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    ll_lists = []
    for val in vals:
        print("val:", val)
        param = {"H0": {'min': -np.inf, 'max': np.inf}}
        ll=0
        ll_list=[]
        sll, sll_list = MCMC.calc_log_posterior([val], state, param,[surveys,[]], grid_params, ind_surveys=True)#,psnr=True)
        
        ll_lists.append(sll_list)
        
    
    ll_lists = np.asarray(ll_lists)
    np.save(outdir+"ll_lists.npy",ll_lists)
    np.save(outdir+"h0vals.npy",vals)

#==============================================================================
"""
Function: commasep
Date: 23/08/2022
Purpose:
    Turn a string of variables seperated by commas into a list

Imports:
    s = String of variables

Exports:
    List conversion of s
"""
def commasep(s):
    return list(map(str, s.split(',')))

#==============================================================================

main()
