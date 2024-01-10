#WARNING - THIS ROUTINE USES OUTDATED NOTATION AND NEEDS TO BE UPDATED

######
# first run this to generate surveys and parameter sets, by 
# setting NewSurveys=True NewGrids=True
# Then set these to False and run with command line arguments
# to generate *many* outputs
#####

# It should be possible to remove all the matplotlib calls from this
# but in the current implementation it is not removed.
import argparse
import numpy as np
import os

from zdm import survey
from zdm import parameters
from zdm import cosmology as cos
from zdm import loading
from zdm import io
from zdm import MCMC
from zdm import misc_functions

import iteration as it

from misc_functions import *
import argparse

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument(dest='names',type=commasep,help='Survey names')
    parser.add_argument(dest='param',type=str,help="Parameter to do the slice in")
    parser.add_argument(dest='min',type=float,help="Min value")
    parser.add_argument(dest='max',type=float,help="Max value")
    parser.add_argument('-n',dest='n',type=int,default=50,help="Number of values")   
    # parser.add_argument('-o','--opfile',type=str,default="out.pdf",help="Output file for the plot slice")
    args = parser.parse_args()

    surveys, grids = loading.surveys_and_grids(survey_names = args.names, repeaters=False)
    
    outdir = 'cube/'
    if not os.path.exists(outdir):
        os.mkdir(outdir)

    vals = np.linspace(args.min, args.max, args.n)

    lls = []
    params = {args.param: {}}
    params[args.param]['max'] = args.max
    params[args.param]['min'] = args.min
    for val in vals:
        ll = MCMC.calc_log_posterior([val], params, surveys, grids)

        # plot_grids(grids, surveys, outdir, val)

        lls.append(ll)
    
    lls = np.array(lls)
    # np.save(os.path.join(outdir,args.opfile), lls)

    plt.plot(vals, lls)
    plt.xlabel(args.param)
    plt.ylabel('log likelihood')
    plt.savefig(os.path.join(outdir,args.param + ".pdf"))

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
