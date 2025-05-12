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

import numpy as np
from zdm import survey
from matplotlib import pyplot as plt
from pkg_resources import resource_filename
def main():
    
    # in case you wish to switch to another output directory
    
    
    
    # approximate best-fit values from recent analysis
    # best-fit from Jordan et al
    if True:
        # approximate best-fit values from recent analysis
        param_dict={'sfr_n': 0.21, 'alpha': 0.11, 'lmean': 2.18, 'lsigma': 0.42, 'lEmax': 41.37, 
                'lEmin': 39.47, 'gamma': -1.04, 'H0': 70.23, 'halo_method': 0, 'sigmaDMG': 0.0, 'sigmaHalo': 0.0,'lC': -7.61}
        opdir='MeerTRAP/'
    else:
        # best fit from James et al
        param_dict={'sfr_n': 1.13, 'alpha': 0.99, 'lmean': 2.27, 'lsigma': 0.55, 'lEmax': 41.26, 
                    'lEmin': 32, 'gamma': -0.95, 'H0': 73, 'halo_method': 0, 'sigmaDMG': 0.0, 'sigmaHalo': 0.0,'lC': -0.76}
        opdir='MeerTRAP2/'
    
    if not os.path.exists(opdir):
        os.mkdir(opdir)
    
    # Initialise surveys and grids
    sdir = os.path.join(resource_filename('zdm', 'data'), 'Surveys')
    names=['MeerTRAPcoherent']
    
    state = parameters.State()
    state.set_astropy_cosmo(Planck18)
    state.update_params(param_dict)
    
    ss,gs = loading.surveys_and_grids(
        survey_names=names,repeaters=False,init_state=state,sdir=sdir) # should be equal to actual number of FRBs, but for this purpose it doesn't matter
    
    # set limits for plots - will be LARGE!   
    DMmax=3000
    zmax=3.
    
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
    
    total = np.sum(g.rates) * s.TOBS * 10**g.state.FRBdemo.lC
    print(name," detects ",total," in 317.5 hr")
    
    plt.sca(ax1)
    plt.xlabel("z")
    plt.ylabel("p(z)")
    plt.xlim(0.01,3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(opdir+"MeerTRAP_pz.pdf")
    plt.close()
    
    np.save(opdir+"MeerTRAP_pz.npy",pz)
    np.save(opdir+"zvalues.npy",g.zvals)
    
    np.save(opdir+"MeerTRAP_pDM.npy",pdm)
    np.save(opdir+"DMvalues.npy",g.dmvals)
    
    zDM_norm = g.rates * s.TOBS * 10**g.state.FRBdemo.lC
    np.save(opdir+"MeerTRAP_zDM.npy",zDM_norm)
    
    
    plt.figure()
    plt.plot(g.zvals,cpz)
    plt.xlabel('z')
    plt.ylabel('$p(z_{\\rm FRB} > z)$')
    plt.tight_layout()
    plt.savefig(opdir+"MeerTRAP_cum_pz.pdf")
    plt.close()
    
    plt.sca(ax2)
    plt.xlabel("DM")
    plt.ylabel("p(DM)")
    plt.xlim(0,4000)
    plt.legend()
    plt.tight_layout()
    plt.savefig(opdir+"MeerTRAP_pdm.pdf")
    plt.close()
    


    
main()
