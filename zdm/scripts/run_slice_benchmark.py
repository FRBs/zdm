import argparse
import numpy as np
import os

from zdm import loading
from zdm import misc_functions
from zdm import iteration as it

from zdm import parameters
from astropy.cosmology import Planck18
from zdm import MCMC2

import argparse
import matplotlib.pyplot as plt

import time

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument(dest='names',type=commasep,help='Survey names')
    parser.add_argument(dest='param',type=str,help="Parameter to do the slice in")
    parser.add_argument(dest='min',type=float,help="Min value")
    parser.add_argument(dest='max',type=float,help="Max value")
    parser.add_argument('-n',dest='n',type=int,default=50,help="Number of values")   
    args = parser.parse_args()

    state = parameters.State()
    state.set_astropy_cosmo(Planck18)

    surveys, grids = loading.surveys_and_grids(survey_names = args.names, repeaters=False, init_state=state)
    
    outdir = 'cube/' + args.param
    if not os.path.exists(outdir):
        os.mkdir(outdir)

    vals = np.linspace(args.min, args.max, args.n)

    sigma = np.linspace(1,6,30)
    sigma[-1] = 7
    fig, axes = plt.subplots(2, 1, sharex=True)
    axes[0].set_ylabel('log likelihood')
    axes[1].set_xlabel('n_sigma')
    axes[1].set_ylabel('time')

    for val in vals:
        lls = []
        ts = []
        for sig in sigma:
            t = 0
            ll  = 0
            for s, g in zip(surveys, grids):
                g.state.update_param('luminosity_function', 0)
                vparams = {}
                vparams[args.param] = val
                g.update(vparams)
                
                t0 = time.time()
                try:
                    ll += it.get_log_likelihood(g, s, sig=sig)
                except ValueError:
                    ll = -np.inf
                t1 = time.time()
                t += t1 - t0

            # plot_grids(grids, surveys, outdir, val)
            ts.append(t)
            lls.append(ll)
        
        lls = np.array(lls)
        lls[lls < -1e10] = -np.inf
        axes[0].plot(sigma, lls, label=str(val))
        axes[1].plot(sigma, ts, label=str(val))

    fig.legend()
    fig.savefig(outdir + "_benchmark.pdf")


    # lls = np.array(lls)
    # lls[lls < -1e10] = -np.inf
    # plt.figure()
    # plt.clf()
    # plt.plot(vals, lls)
    # # plt.plot(vals, llsum2)
    # plt.xlabel(args.param)
    # plt.ylabel('log likelihood')
    # plt.savefig(outdir + "_MCMC2.pdf")

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

        misc_functions.plot_grid_2(
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
