""" 
This script creates a zDM plot for SKA_Mid

It also estimates the raction of SKA bursts that will have
unseen hosts by a VLT-like optical obeservation
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

import numpy as np
from zdm import survey
from matplotlib import pyplot as plt
from pkg_resources import resource_filename

def main():
    
    H0=70
    logF=np.log10(0.32)
    
    # in case you wish to switch to another output directory
    opdir='SKA/'
    if not os.path.exists(opdir):
        os.mkdir(opdir)
    
    # Initialise surveys and grids
    sdir = os.path.join(resource_filename('zdm', 'data'), 'Surveys')
    names=['SKA_mid']
    state = parameters.State()
    # approximate best-fit values from recent analysis
    state.set_astropy_cosmo(Planck18)
    
    # approximate best-fit values from recent analysis
    # best-fit from Jordan et al
    param_dict={'sfr_n': 0.21, 'alpha': 0.11, 'lmean': 2.18, 'lsigma': 0.42, 'lEmax': 41.37, 
                'lEmin': 39.47, 'gamma': -1.04, 'H0': 70.23, 'halo_method': 0, 'sigmaDMG': 0.0, 'sigmaHalo': 0.0,'lC': -7.61}
    
    state.update_params(param_dict)
    
    nozlist=[]
    ss,gs = loading.surveys_and_grids(
        survey_names=names,repeaters=False,init_state=state,sdir=sdir) # should be equal to actual number of FRBs, but for this purpose it doesn't matter
    
    pU = opt.p_unseen(gs[0].zvals,plot=False)
    
    # set limits for plots - will be LARGE!   
    DMmax=5000
    zmax=4.5
    
    plt.figure()
    ax1 = plt.gca()
    
    plt.figure()
    ax2 = plt.gca()
    
    g=gs[0]
    s=ss[0]
    name = names[0]
    figures.plot_grid(g.rates,g.zvals,g.dmvals,
        name=opdir+name+"_zDM.pdf",norm=3,log=True,
        label='$\\log_{10} p({\\rm DM}_{\\rm IGM} + {\\rm DM}_{\\rm host},z)$ [a.u.]',
        project=False,ylabel='${\\rm DM}_{\\rm IGM} + {\\rm DM}_{\\rm host}$',
        zmax=zmax,DMmax=DMmax,Aconts=[0.01,0.1,0.5])
    
    pz = np.sum(g.rates,axis=1)
    ax1.plot(g.zvals,pz/np.max(pz),label=name)
    
    cpz = np.cumsum(pz)
    cpz /= cpz[-1]
    
    pdm = np.sum(g.rates,axis=0)
    ax2.plot(g.dmvals,pdm/np.max(pdm),label=name)
    
    total = np.sum(g.rates) * s.TOBS * 10**g.state.FRBdemo.lC
    print(name," detects ",total," in ",s.TOBS," days")
    
    plt.sca(ax1)
    plt.xlabel("z")
    plt.ylabel("p(z)")
    plt.xlim(0.01,3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(opdir+"SKA_pz.pdf")
    plt.close()
    
    
    pz /= np.max(pz)
    Ntot=np.sum(pz)
    NU=np.sum(pz*pU)
    frac = NU/Ntot
    print(frac," of SKA mid FRBs unseen")
    plt.figure()
    plt.xlim(0.01,5)
    plt.ylim(0,1)
    plt.plot(g.zvals,pz,label='$P_{\\rm FRB}(z)$ SKA FRBs')
    plt.plot(g.zvals,pU,label='$P(U)$')
    plt.plot(g.zvals,pz*pU,label='$P_{\\rm FRB}(z) P(U) $')
    plt.xlabel('z')
    plt.ylabel('$p(z)$')
    plt.legend()
    plt.tight_layout()
    plt.savefig(opdir+"SKA_Punseen.pdf")
    plt.close()
    
    
    plt.figure()
    plt.plot(g.zvals,cpz)
    plt.xlabel('z')
    plt.ylabel('$p(z_{\\rm FRB} > z)$')
    plt.tight_layout()
    plt.savefig(opdir+"SKA_cum_pz.pdf")
    plt.close()
    
    plt.sca(ax2)
    plt.xlabel("DM")
    plt.ylabel("p(DM)")
    plt.xlim(0,4000)
    plt.legend()
    plt.tight_layout()
    plt.savefig(opdir+"SKA_pdm.pdf")
    plt.close()
    


    
main()
