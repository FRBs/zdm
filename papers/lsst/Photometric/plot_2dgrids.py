"""
Evaluates the likelihood of a slice through H0 for various surveys
"""

import argparse
import numpy as np
import os

from zdm import figures
from zdm import iteration as it
from zdm import loading
from zdm import parameters
from zdm import repeat_grid as zdm_repeat_grid
from zdm import MCMC
from zdm import survey
from zdm import states
from astropy.cosmology import Planck18
import importlib.resources as resources
from numpy import random
import matplotlib.pyplot as plt
import time

def main():
    """
    Plots 2D zDM grids
    """
    # Set state
    state=states.load_state(case="HoffmannHalo25",scat=None,rep=None)
    sdir = resources.files('zdm').joinpath('../papers/lsst/Photometric')
    names = ["Spectroscopic","Smeared","zFrac","Smeared_and_zFrac"]
    xlabels = ["$z_{\\rm spec}$","$z_{\\rm photo}$","$z_{\\rm spec}$","$z_{\\rm photo}$"]
    ss,gs = loading.surveys_and_grids(
        survey_names=names,repeaters=False,init_state=state,sdir=sdir)
    plot_grids(gs,ss,"./",xlabels)
    

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
def plot_grids(grids, surveys, outdir,xlabels):
    for i,g in enumerate(grids):
        s = surveys[i]
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
            name=outdir + s.name + "_zDM.png",
            norm=3,
            log=True,
            label="$\\log_{10} p({\\rm DM}_{\\rm EG},z)$ [a.u.]",
            xlabel=xlabels[i],
            project=False,
            FRBDMs=frbdmvals,
            FRBZs=frbzvals,
            Aconts=[0.01, 0.1, 0.5],
            zmax=3.0,
            DMmax=3000,
            # DMlines=nozlist,
        )

main()
