import argparse
import numpy as np
import os

from zdm import survey
from zdm import figures
from zdm import iteration as it

from zdm import parameters
from astropy.cosmology import Planck18
from zdm import MCMC

import argparse
import matplotlib.pyplot as plt

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument(dest='param',type=str,help="Parameter to do the slice in")
    parser.add_argument(dest='min',type=float,help="Min value")
    parser.add_argument(dest='max',type=float,help="Max value")
    parser.add_argument('-f', '--files', default=None, nargs='+', type=str, help="Survey file names")
    # parser.add_argument('-r', '--rep_surveys', default=None, nargs='+', type=str, help="Surveys to consider repeaters in")
    parser.add_argument('-n',dest='n',type=int,default=50,help="Number of values")   
    args = parser.parse_args()


    param_dict = {args.param: {'min': args.min, 'max':args.max}}

    state = parameters.State()
    vparams = {
        # 'sfr_n': 0.03867529062400976,
        # 'alpha': 0.7966797154629295,
        # 'lmean': 1.7530952643556483,
        # 'lsigma': 1.085320704456141,
        # 'gamma': -1.0848173305384008,
        # 'H0': 67.20070230478856,
        # 'DMhalo': 100.0
    }
    state.update_params(vparams)

    grid_params = {}
    grid_params['dmmax'] = 7000.0
    grid_params['ndm'] = 1400
    grid_params['nz'] = 500
    ddm = grid_params['dmmax'] / grid_params['ndm']
    dmvals = (np.arange(grid_params['ndm']) + 1) * ddm
    
    surveys = []
    for survey_name in args.files:
        s = survey.load_survey(survey_name, state, dmvals)
        surveys.append(s)
    
    outdir = 'cube/' + args.param
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    vals = np.linspace(args.min, args.max, args.n)

    lls = []
    for val in vals:
        ll = MCMC.calc_log_posterior([val], state, param_dict, [surveys, []], grid_params)
        print(ll, val, flush=True)

        lls.append(ll)

    lls = np.array(lls)
    lls[lls < -1e10] = -np.inf
    plt.figure()
    plt.clf()
    plt.plot(vals, lls)
    # plt.plot(vals, llsum2)
    plt.xlabel(args.param)
    plt.ylabel('log likelihood')
    plt.savefig(os.path.join(outdir, args.param + "_MCMC2.pdf"))

#==============================================================================
"""
Function: plot_grids
Date: 10/01/2024
Purpose:
    Plot grids. Adapted from zdm/scripts/plot_pzdm_grid.py

Imports:
    grids = list of grids
    surveys = list of surveys
    outdir = output directory
    val = parameter value for this grid
"""
def plot_grids(grids, surveys, outdir, val):
    for g,s in zip(grids, surveys):
        zvals=[]
        dmvals=[]
        nozlist=[]
        
        if s.zlist is not None:
            for iFRB in s.zlist:
                zvals.append(s.Zs[iFRB])
                dmvals.append(s.DMEGs[iFRB])
        if s.nozlist is not None:
            for dm in s.DMEGs[s.nozlist]:
                nozlist.append(dm)
        
        frbzvals = np.array(zvals)
        frbdmvals = np.array(dmvals)

        figures.plot_grid(
            g.rates,
            g.zvals,
            g.dmvals,
            name=outdir + s.name + "_" + str(val) + ".pdf",
            norm=3,
            log=True,
            label="$\\log_{10} p({\\rm DM}_{\\rm EG},z)$ [a.u.]",
            project=False,
            FRBDM=frbdmvals,
            FRBZ=frbzvals,
            Aconts=[0.01, 0.1, 0.5],
            zmax=1.5,
            DMmax=3000,
            # DMlines=nozlist,
        )

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
