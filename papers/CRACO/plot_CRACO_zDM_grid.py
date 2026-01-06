""" 
This script creates zdm grids for CRACO observations.

It exists partly to calculate relative rates from surveys

It also calculates expected p(DM) distributions

"""
import os

from astropy.cosmology import Planck18
from zdm import cosmology as cos
from zdm import figures
from zdm import parameters
from zdm import survey
from zdm import pcosmic
from zdm import iteration as it
from zdm import loading
from zdm import io
from zdm import optical as opt
from zdm import states

import numpy as np
from zdm import survey
from matplotlib import pyplot as plt
import importlib.resources as resources

def main():
    
    # in case you wish to switch to another output directory
    name="zDMPlots"
    opdir=name+"/"
    
    # approximate best-fit values from recent analysis
    # best-fit from Jordan et al
    state = states.load_state("HoffmannHalo25",scat="updated",rep=None)
    
    if not os.path.exists(opdir):
        os.mkdir(opdir)
    
    # Initialise surveys and grids
    sdir = resources.files('zdm').joinpath('data/Surveys')
    names=['CRAFT_CRACO_900','CRAFT_CRACO_1300']
    
    ss,gs = loading.surveys_and_grids(
        survey_names=names,repeaters=False,init_state=state,sdir=sdir) # should be equal to actual number of FRBs, but for this purpose it doesn't matter
    
    # set limits for plots - will be LARGE!   
    DMmax=3000
    zmax=2.5
    
    plt.figure()
    ax1 = plt.gca()
    
    plt.figure()
    ax2 = plt.gca()
    
    # ids of FRBs with voltage data and redshift
    voltage = np.array([3, 6, 11, 12, 16],dtype="int")
    
    # chooses the first arbitrarily to extract zvals etc from
    for i,g in enumerate(gs):
        s=ss[i]
        g=gs[i]
        name = names[i]
        
        noz=s.nozlist
        z=s.zlist
        
        if noz is not None:
            DMlines = s.DMEGs[noz]
        else:
            DMlines=None
        
        if z is not None:
            FRBDMs = s.DMEGs[z]
            FRBZs=s.Zs[z]
        else:
            FRBDMs=None
            FRBZs=None
        print(s.DMEGs[voltage])
        print(s.Zs[voltage])
        exit()
        special = [s.DMEGs[voltage],s.Zs[voltage]]
        
        figures.plot_grid(gs[i].get_rates(),g.zvals,g.dmvals,
            name=opdir+name+"_zDM.pdf",norm=3,log=True,
            label='$\\log_{10} p({\\rm DM}_{\\rm cosmic} + {\\rm DM}_{\\rm host},z)$ [a.u.]',
            project=False,ylabel='${\\rm DM}_{\\rm cosmic} + {\\rm DM}_{\\rm host}$',
            zmax=zmax,DMmax=DMmax,Aconts=[0.01,0.1,0.5],
            FRBDMs=FRBDMs,FRBZs=FRBZs,
            DMlines = DMlines,special=special
            )
        
        rate = np.sum(gs[i].rates) * 10**g.state.FRBdemo.lC 
        print("Relative rate for ",name," is ",rate," per day")
    


    
main()
