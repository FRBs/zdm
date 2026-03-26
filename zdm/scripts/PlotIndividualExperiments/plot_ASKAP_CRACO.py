""" 
This script creates zdm grids for ASKAP incoherent sum observations.



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
    name="CRACO"
    opdir=name+"/"
    
    # approximate best-fit values from recent analysis
    # best-fit from Jordan et al
    # plot scat "updated" if better, but takes ages!
    state = states.load_state("HoffmannHalo25",scat="updated",rep=None)
    
    if not os.path.exists(opdir):
        os.mkdir(opdir)
    
    # Initialise surveys and grids
    sdir = resources.files('zdm').joinpath('data/Surveys')
    names=['CRAFT_CRACO_1300','CRAFT_CRACO_900']

    ss,gs = loading.surveys_and_grids(
        survey_names=names,repeaters=False,init_state=state,sdir=sdir) # should be equal to actual number of FRBs, but for this purpose it doesn't matter
    
    # set limits for plots - will be LARGE!   
    DMmax=3000
    zmax=3.
    
    plt.figure()
    ax1 = plt.gca()
    
    plt.figure()
    ax2 = plt.gca()
    
    # chooses the first arbitrarily to extract zvals etc from
    for i,g in enumerate(gs):
        s=ss[i]
        name = names[i]
        
        noz = np.where(s.frbs['Z'].values < 0.)
        
        figures.plot_grid(g.get_rates(),g.zvals,g.dmvals,
            name=opdir+name+"_zDM.pdf",norm=3,log=True,
            label='$\\log_{10} p({\\rm DM}_{\\rm EX},z)$ [a.u.]',
            project=False,ylabel='${\\rm DM}_{\\rm EX}$',
            zmax=zmax,DMmax=DMmax,Aconts=[0.01,0.1,0.5],
            FRBDMs=s.frbs['DMEG'].values,FRBZs=s.frbs['Z'].values,
            DMlines = s.frbs['DMEG'].values[noz])
            
            
            
    exit()
    
    pz = np.sum(mean_rates,axis=1)
    pz /= np.max(pz)
    ax1.plot(g.zvals,pz,label=name)
    
    cpz = np.cumsum(pz)
    cpz /= cpz[-1]
    
    pdm = np.sum(mean_rates,axis=0)
    pdm /= np.max(pdm)
    ax2.plot(g.dmvals,pdm,label=name)
    
    total = np.sum(mean_rates)
    print(name," expected to detect ",total," in ",time," hr")
    
    plt.sca(ax1)
    plt.xlabel("z")
    plt.ylabel("p(z)")
    plt.xlim(0.01,3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(opdir+name+"_pz.pdf")
    plt.close()
    
    np.save(opdir+name+"_pz.npy",pz)
    np.save(opdir+"zvalues.npy",g.zvals)
    
    np.save(opdir+name+"_pDM.npy",pdm)
    np.save(opdir+"DMvalues.npy",g.dmvals)
    
    zDM_norm = g.rates * s.TOBS * 10**g.state.FRBdemo.lC
    np.save(opdir+name+"_zDM.npy",zDM_norm)
    
    
    plt.figure()
    plt.plot(g.zvals,cpz)
    plt.xlabel('z')
    plt.ylabel('$p(z_{\\rm FRB} > z)$')
    plt.tight_layout()
    plt.savefig(opdir+name+"_cum_pz.pdf")
    plt.close()
    
    plt.sca(ax2)
    plt.xlabel("DM")
    plt.ylabel("p(DM)")
    plt.xlim(0,4000)
    plt.legend()
    plt.tight_layout()
    plt.savefig(opdir+name+"_pdm.pdf")
    plt.close()
    


    
main()
