""" 
This script is designed to test that MC event generation is working.

It initiatlises a grid, and then generates MC events from it.

It generates a false survey, inserts these MC FRBs into it,
and then evaluates a log-likelihood based on these MC FRBs.

It shows a figure giving those log-likelihoods.

         
"""
import os

from zdm import cosmology as cos
from zdm import figures
from zdm import parameters
from zdm import survey
from zdm import pcosmic
from zdm import iteration as it
from zdm import loading
from zdm import io
from astropy.cosmology import Planck18
import numpy as np
from zdm import survey
from matplotlib import pyplot as plt

np.random.seed()

def main():
    
    # Initialise surveys and grids
    sdir='../data/Surveys/'
    name='parkes_mb_class_I_and_II'
    
    state = parameters.State()
    state.set_astropy_cosmo(Planck18)
    
    # approximate best-fit values from recent analysis
    param_dict={'sfr_n': 0.8808527057055584, 'alpha': 0.7895161131856694, 'lmean': 2.1198711983468064, 'lsigma': 0.44944780033763343, 'lEmax': 41.18671139482926, 
                'lEmin': 38.81049090314043, 'gamma': -1.1558450520609953, 'H0': 54.6887137195215, 'halo_method': 0, 'sigmaDMG': 0.0, 'sigmaHalo': 0.0}
    state.update_params(param_dict)
    names=["CRAFT_ICS_1300"]
    surveys, grids = loading.surveys_and_grids(survey_names = names, repeaters=False,
        init_state=state)
    
    s=surveys[0]
    g=grids[0]
    
    figures.plot_grid(g.rates,g.zvals,g.dmvals,
            name="TESTrates.pdf",norm=3,log=True,
            label='$\\log_{10} p({\\rm DM}_{\\rm IGM} + {\\rm DM}_{\\rm host},z)$ [a.u.]',
            project=False,ylabel='${\\rm DM}_{\\rm IGM} + {\\rm DM}_{\\rm host}$',
            zmax=3.,DMmax=3000.)
    
    ############ sets up the histograms #########
    
    # OK, the zvals and DM vals represent FRBs found in bins *centred* at those values
    dz = g.zvals[1]-g.zvals[0]
    dDM = g.dmvals[1] - g.dmvals[0]
    NZ = g.zvals.size
    NDM = g.dmvals.size
    zbins = np.linspace(g.zvals[0] - dz/2., g.zvals[-1] + dz/2.,NZ+1)
    DMbins = np.linspace(g.dmvals[0] - dDM/2., g.dmvals[-1] + dDM/2.,NDM+1)
    
    ############ generates the MCMC #########
    # array values are DM, z, B, w, and SNR
    NMC = 400
    frbs = g.GenMCSample(NMC)
    frbs = np.array(frbs)
    
    MCs = frbs[:,3]
    MCDM = frbs[:,1]
    MCz = frbs[:,0]
    
    
    # updates survey with critical FRB info only. This is a hack!
    s.zlist = np.arange(NMC)
    s.nozlist = np.arange(NMC)
    s.DMEGs = MCDM
    s.Zs = MCz
    s.Ss = MCs
    s.DMs = MCDM + 80.
    s.DMhalos = np.full([NMC],50.)
    s.DMGs = np.full([NMC],30.)
    s.NORM_FRB = NMC
    
    ##### evaluate 2D likelihoods #####
    ll,list,nfrb,allll = it.calc_likelihoods_1D(g,s,dolist=2,psnr=True)
    
    plt.figure()
    plt.scatter(MCz,allll)
    plt.xlabel('z')
    plt.ylabel('log likelihood [DM=1]')
    
    plt.show()
    
main()
