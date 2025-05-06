""" 
This script is designed to test that MC event generation is working.

It initiatlises a grid, and then generates MC events from it.

It saves these to two 1D histograms in z and DM, and a 2D
histogram in zDM space (other MC info is currently discarded).

It then generates plots to compare the MC generated events to
the grid they were generated from.

Because generating sufficient MC takes quite some time,
it saves all data, and adds to it with each iteration.
Thus it can be run with 10,100,1000 etc MC events,
and then statistics can be accumulated.

The MC events are saved in the following npy files:
- dmhist.npy        A 1D histogram of dispersion measure   
- totalcount.npy    Single integer: number of MC events generated
- zdmhist.npy       A 2D histogram in zDM space
- zhist.npy         A 1D histogram of redshift
Note that for this to make sense, the dimensions of the original
grid should stay constant.

Plots get generated at each iteration. These are:
-   pz.pdf      Compares expected and generated p(z) distributions
-   pzerr.pdf   Statistical error in the above in units of sigma
-   pDM.pdf     Compares expected and generated p(DM) distributions
-   pDMerr.pdf  Statistical error in the above in units of sigma
-   MCzDM.pdf   Generated 2D histogram of MC generated events
-   rel_zDM_err.pdf Relative error in the above (units of sigma)
-   grid_expectation.pdf   Grid expected zDM distribution


         
"""
import os

from zdm import cosmology as cos
from zdm import misc_functions
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
                'lEmin': 39.81049090314043, 'gamma': -1.1558450520609953, 'H0': 54.6887137195215, 'halo_method': 0, 'sigmaDMG': 0.0, 'sigmaHalo': 0.0}
    state.update_params(param_dict)
    names=["CRAFT_ICS_1300"]
    surveys, grids = loading.surveys_and_grids(survey_names = names, repeaters=False,
        init_state=state)
    
    s=surveys[0]
    g=grids[0]
    
    ############ sets up the histograms #########
    
    # OK, the zvals and DM vals represent FRBs found in bins *centred* at those values
    dz = g.zvals[1]-g.zvals[0]
    dDM = g.dmvals[1] - g.dmvals[0]
    NZ = g.zvals.size
    NDM = g.dmvals.size
    zbins = np.linspace(g.zvals[0] - dz/2., g.zvals[-1] + dz/2.,NZ+1)
    DMbins = np.linspace(g.dmvals[0] - dDM/2., g.dmvals[-1] + dDM/2.,NDM+1)
    
    zDMhistfile = "zdmhist.npy"
    zhistfile = "dmhist.npy"
    DMhistfile =  "zhist.npy"
    countfile = "totalcount.npy"
    
    if os.path.exists(zDMhistfile):
        zDMhist = np.load(zDMhistfile)
        DMhist = np.load(DMhistfile)
        zhist = np.load(zhistfile)
        NFRB = np.load(countfile)
    else:
        zDMhist = np.zeros([NZ,NDM])
        zhist = np.zeros([NZ])
        DMhist = np.zeros([NDM])
        NFRB=0
    
    ############ generates the MCMC #########
    # array values are DM, z, B, w, and SNR
    NMC = 400
    frbs = g.GenMCSample(NMC)
    frbs = np.array(frbs)
    
    zs = frbs[:,0]
    DMs = frbs[:,1]
    snrs = frbs[:,3]
    bs = frbs[:,2]
    ws = frbs[:,4]
    
    # check it lives in the right bins
    lowz = np.where(zs < zbins[0])[0]
    highz = np.where(zs > zbins[-1])[0]
    lowDM = np.where(DMs < DMbins[0])[0]
    highDM = np.where(DMs > DMbins[-1])[0]
    print("FRBs out of range: ",len(lowz),len(highz),len(lowDM),len(highDM))
    if len(lowz)>0:
        print(zs[lowz])
    if len(highz)>0:
        print(zs[highz])
    if len(lowDM)>0:
        print(DMs[lowDM])
    if len(highDM)>0:
        print(DMs[highDM])
    ############ histogram the data #########
    
    tempzDMhist,xb,yb = np.histogram2d(zs,DMs,bins=[zbins,DMbins])
    tempzhist,xb = np.histogram(zs,bins=zbins)
    tempDMhist,yb = np.histogram(DMs,bins=DMbins)
    
    zDMhist += tempzDMhist
    zhist += tempzhist
    DMhist += tempDMhist
    
    np.save(zDMhistfile,zDMhist)
    np.save(zhistfile,zhist)
    np.save(DMhistfile,DMhist)
    
    NFRB += NMC
    np.save(countfile,NFRB)
    
    print("MC done, total number of simulated FRBs: ",NFRB)
    ############Normalisation and Errors ########
    # z-only plot
    
    
    # gets expectation
    pz = np.sum(g.rates,axis=1)
    pdm = np.sum(g.rates,axis=0)
    pz /= np.sum(pz)
    pdm /= np.sum(pdm)
    pzdm = g.rates / np.sum(g.rates)
    
    zDMhisterr = zDMhist **0.5
    norm = np.sum(zDMhist)
    zDMhisterr /= norm
    zDMhist /= norm
    
    zhisterr = zhist **0.5
    norm = np.sum(zhist)
    zhisterr /= norm
    zhist /= norm
    
    DMhisterr = DMhist **0.5
    norm = np.sum(DMhist)
    DMhisterr /= norm
    DMhist /= norm
    
    # z-only plot
    
    plt.figure()
    plt.plot(g.zvals,pz,linewidth=3,linestyle="-",color="black",label="expectation")
    plt.plot(g.zvals,zhist,linestyle = "-",linewidth=2,label="MC")
    plt.plot(g.zvals,zhist-zhisterr,linestyle = ":",linewidth=1,label="err",color=plt.gca().lines[-1].get_color())
    plt.plot(g.zvals,zhist+zhisterr,linestyle = ":",linewidth=1,color=plt.gca().lines[-1].get_color())
    plt.xlabel("z")
    plt.ylabel("p(z) dz")
    plt.xlim(0,2)
    plt.legend()
    plt.tight_layout()
    plt.savefig("pz.pdf")
    plt.close()
    
    plt.figure()
    altzhisterr = (pz*NFRB)**0.5/NFRB
    altrel_error = (zhist - pz)/altzhisterr
    rel_error = (zhist - pz)/zhisterr
    plt.plot(g.zvals,rel_error,linewidth=1,linestyle="-",marker='o',markerfacecolor='red')
    plt.xlabel("z")
    plt.xlim(0,2)
    plt.ylabel("$\\Delta p(z) / \\sigma p(z)$")
    plt.tight_layout()
    plt.savefig("pzerr.pdf")
    plt.close()
    
    
    # DM-only plot
    plt.figure()
    plt.plot(g.dmvals,pdm,linewidth=3,linestyle="-",color="black",label="expectation")
    plt.plot(g.dmvals,DMhist,linestyle = "-",linewidth=2,label="MC")
    plt.plot(g.dmvals,DMhist-DMhisterr,linestyle = ":",linewidth=1,label="err",color=plt.gca().lines[-1].get_color())
    plt.plot(g.dmvals,DMhist+DMhisterr,linestyle = ":",linewidth=1,color=plt.gca().lines[-1].get_color())
    plt.xlabel("DM")
    plt.ylabel("p(DM) dDM")
    plt.legend()
    plt.tight_layout()
    plt.savefig("pDM.pdf")
    plt.close()
    
    
    plt.figure()
    altDMhisterr = (pdm*NFRB)**0.5/NFRB
    altrel_error = (DMhist - pdm)/altDMhisterr
    rel_error = (DMhist - pdm)/DMhisterr
    plt.plot(g.dmvals,rel_error,linewidth=1,linestyle="-",marker='o',markerfacecolor='red')
    plt.xlabel("DM")
    plt.ylabel("$\\Delta p({\\rm DM}) / \\sigma p({\\rm DM})$")
    plt.tight_layout()
    plt.savefig("pDMerr.pdf")
    plt.close()
    
    
    
    # zDM 2D plot
    # plots the p(DMEG (host + cosmic)|z) grid
    misc_functions.plot_grid_2(zDMhist,g.zvals,g.dmvals,
        name='MCzDM.pdf',norm=3,log=False,
        label='$\\log_{10} p({\\rm DM}_{\\rm IGM} + {\\rm DM}_{\\rm host}|z)$ [a.u.]',
        project=False)
    
    misc_functions.plot_grid_2(g.rates,g.zvals,g.dmvals,
        name='grid_expectation.pdf',norm=3,log=False,
        label='$\\log_{10} p({\\rm DM}_{\\rm IGM} + {\\rm DM}_{\\rm host}|z)$ [a.u.]',
        project=False)
    
    expectation = g.rates / np.sum(g.rates)
    zDMerr = (expectation * NFRB)**0.5 / NFRB
    rel_err = (zDMhist - expectation)/zDMhisterr
    
    misc_functions.plot_grid_2(rel_err,g.zvals,g.dmvals,
        name='rel_zDM_err.pdf',norm=3,log=False,
        label='$\\sigma$ deviation',
        project=False)
    
main()
