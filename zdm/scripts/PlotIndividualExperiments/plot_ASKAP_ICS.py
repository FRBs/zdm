""" 
This script creates zdm grids for ASKAP incoherent sum observations.

It steps through different effects, beginning with the
intrinsic zdm, applying various cuts.

It generates the following plots in opdir:

dm_cosmic.pdf: shows only intrinsic p(dm_cosmic|z)
dm_eg.pdf: shows intrinsic p(dm_host + dm_cosmic|z)

pEG_luminosity.pdf: shows p(DM,z) including
    - source evolution
    - FRB luminosity function
    - cosmological volume calculation

pEG_luminosity_eff.pdf: shows p(DM,z) as pEG_luminosity, adding in
    - detection efficiency losses from FRB width, DM smearing etc

pEG_luminosity_beam.pdf: shows p(DM,z) as pEG_luminosity, adding in
    - antenna beamshape

pEG_luminosity_eff_beam.pdf: shows p(DM,z) as pEG_luminosity, adding in
    - detection efficiency losses from FRB width, DM smearing etc
    - telescope beamshape
    (i.e. this is the full calculation of FRB rates)

pEG_luminosity_eff_beam_FRBs.pdf: as above, but shows detected FRBs


"""
import os

from astropy.cosmology import Planck18
from zdm import cosmology as cos
from zdm import misc_functions
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
    name="ASKAP"
    opdir=name+"/"
    
    # approximate best-fit values from recent analysis
    # best-fit from Jordan et al
    if True:
        # approximate best-fit values from recent analysis
        param_dict={'sfr_n': 0.21, 'alpha': 0.11, 'lmean': 2.18, 'lsigma': 0.42, 'lEmax': 41.37, 
                'lEmin': 39.47, 'gamma': -1.04, 'H0': 70.23, 'halo_method': 0, 'sigmaDMG': 0.0, 'sigmaHalo': 0.0,'lC': -7.61}
        
    else:
        # best fit from James et al
        param_dict={'sfr_n': 1.13, 'alpha': 0.99, 'lmean': 2.27, 'lsigma': 0.55, 'lEmax': 41.26, 
                    'lEmin': 32, 'gamma': -0.95, 'H0': 73, 'halo_method': 0, 'sigmaDMG': 0.0, 'sigmaHalo': 0.0,'lC': -0.76}
    
    
    if not os.path.exists(opdir):
        os.mkdir(opdir)
    
    # Initialise surveys and grids
    sdir = os.path.join(resource_filename('zdm', 'data'), 'Surveys')
    names=['CRAFT_ICS_892','CRAFT_ICS_1300','CRAFT_ICS_1632']
    
    state = parameters.State()
    state.set_astropy_cosmo(Planck18)
    state.update_params(param_dict)
    
    ss,gs = loading.surveys_and_grids(
        survey_names=names,repeaters=False,init_state=state,sdir=sdir) # should be equal to actual number of FRBs, but for this purpose it doesn't matter
    
    # set limits for plots - will be LARGE!   
    DMmax=3000
    zmax=3.
    
    # gets sum of rates over three sets of observations
    # weights by constant and TOBS
    time=0
    for i,g in enumerate(gs):
        if i==0:
            mean_rates=g.rates * ss[i].TOBS * 10**g.state.FRBdemo.lC
        else:
            mean_rates += g.rates * ss[i].TOBS * 10**g.state.FRBdemo.lC
        time += ss[i].TOBS
        
    plt.figure()
    ax1 = plt.gca()
    
    plt.figure()
    ax2 = plt.gca()
    
    # chooses the first arbitrarily to extract zvals etc from
    s=ss[0]
    g=gs[0]
    name = names[0]
    misc_functions.plot_grid_2(mean_rates,g.zvals,g.dmvals,
        name=opdir+name+"_zDM.pdf",norm=3,log=True,
        label='$\\log_{10} p({\\rm DM}_{\\rm IGM} + {\\rm DM}_{\\rm host},z)$ [a.u.]',
        project=False,ylabel='${\\rm DM}_{\\rm IGM} + {\\rm DM}_{\\rm host}$',
        zmax=zmax,DMmax=DMmax,Aconts=[0.01,0.1,0.5])
    
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
