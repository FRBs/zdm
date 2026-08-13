""" 
This script creates zdm grids for MeerTRAP

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

import matplotlib

defaultsize=14
ds=4
font = {'family' : 'Helvetica',
        'weight' : 'normal',
        'size'   : defaultsize}
matplotlib.rc('font', **font)


def main():
    
    # approximate best-fit values from recent analysis
    # load states from Hoffman et al 2025
    state = states.load_state("HoffmannEmin25",scat="updated",rep=None)
    opdir="DSA"
    
    if not os.path.exists(opdir):
        os.mkdir(opdir)
    
    # Initialise surveys and grids
    sdir = resources.files('zdm').joinpath('data/Surveys')
    name="DSA"
    names=[name]
    
    ss,gs = loading.surveys_and_grids(
        survey_names=names,repeaters=False,init_state=state,sdir=sdir) # should be equal to actual number of FRBs, but for this purpose it doesn't matter
    
    # set limits for plots - will be LARGE!   
    DMmax=2000
    zmax=2.
    
    plt.figure()
    ax1 = plt.gca()
    
    plt.figure()
    ax2 = plt.gca()
    
    s=ss[0]
    g=gs[0]
    name = names[0]
    figures.plot_grid(g.rates,g.zvals,g.dmvals,
        name=opdir+name+"_zDM.pdf",norm=3,log=True,
        label='$\\log_{10} p({\\rm DM}_{\\rm IGM} + {\\rm DM}_{\\rm host},z)$ [a.u.]',
        project=False,ylabel='${\\rm DM}_{\\rm IGM} + {\\rm DM}_{\\rm host}$',
        zmax=zmax,DMmax=DMmax,Aconts=[0.01,0.1,0.5])
    
    pz = np.sum(g.rates,axis=1)
    pz /= np.max(pz)
    ax1.plot(g.zvals,pz,label=name)
    
    cpz = np.cumsum(pz)
    cpz /= cpz[-1]
    
    pdm = np.sum(g.rates,axis=0)
    pdm /= np.max(pdm)
    ax2.plot(g.dmvals,pdm,label=name)
    
    total = np.sum(g.get_rates())
    print(name," detects ",total," per day")
    
    plt.sca(ax1)
    plt.xlabel("z")
    plt.ylabel("p(z)")
    plt.xlim(0.01,3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(opdir+name+"_pz.pdf")
    plt.close()
    
    np.save(opdir+name+"_pz.npy",pz)
    np.save(opdir+name+"zvalues.npy",g.zvals)
    
    np.save(opdir+name+"_pDM.npy",pdm)
    np.save(opdir+name+"DMvalues.npy",g.dmvals)
    
    zDM_norm = g.rates* 10**g.state.FRBdemo.lC
    np.save(opdir+name+"_zDM.npy",zDM_norm)
    
    
    plt.figure()
    plt.plot(g.zvals,cpz)
    plt.xlabel('z')
    plt.xlim(0,zmax)
    plt.ylabel('$p(z_{\\rm FRB} > z)$')
    plt.tight_layout()
    plt.savefig(opdir+name+"_cum_pz.pdf")
    plt.close()
    
    plt.sca(ax2)
    plt.xlabel("DM")
    plt.ylabel("p(DM)")
    plt.xlim(0,DMmax)
    plt.legend()
    plt.tight_layout()
    plt.savefig(opdir+name+"_pdm.pdf")
    plt.close()
    


    
main()
