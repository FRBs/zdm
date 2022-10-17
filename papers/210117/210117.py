""" 
This script creates pots for the 210117 paper

"""
import os

from zdm import analyze_cube as ac
from zdm import cosmology as cos
from zdm import misc_functions
from zdm import parameters
from zdm import survey
from zdm import pcosmic
from zdm import iteration as it
from zdm.craco import loading
from zdm import io

import numpy as np
from zdm import survey
from matplotlib import pyplot as plt
import matplotlib

defaultsize=14
ds=4
font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : defaultsize}
matplotlib.rc('font', **font)


def main():
    
    
    
    from astropy.cosmology import Planck18
    
    doE=False
    if doE:
        convert_energy()
    
    # in case you wish to switch to another output directory
    opdir = "210117/"
    if not os.path.exists(opdir):
        os.mkdir(opdir)
    
    load=False
    
    # The below is for private, unpublished FRBs. You will NOT see this in the repository!
    names = 'CRAFT_ICS'
    sdir = "../../zdm/data/Surveys/"
    
    #labels=["lEmax","alpha","gamma","sfr_n","lmean","lsigma"]
    
    # approximate best-fit values from recent analysis
    vparams = {}
    vparams["H0"] = 73
    vparams["lEmax"] = 41.26
    vparams["gamma"] = -0.95
    vparams["alpha"] = 0.99
    vparams["sfr_n"] = 1.13
    vparams["lmean"] = 2.27
    vparams["lsigma"] = 0.55
    
    opfile=opdir+"pzgdm.pdf"
    plot_expectations(names,sdir,vparams,opfile,dmhost=False)
    
    
def plot_expectations(name,sdir,vparams,opfile,intermediate=False,sumit=True,dmhost=False):
    # Initialise surveys and grids
    # if True, this generates a summed histogram of all the surveys, weighted by
    # the observation time
    #sumit = True
    DMEG210117=730-34.4-50
    Z210117=0.214
    
    zvals = []
    dmvals = []
    nozlist = []
    #state = set_special_state()
    #,init_state=state
    s, g = loading.survey_and_grid(
        survey_name=name, NFRB=None, sdir=sdir, lum_func=2, nz=1000,ndm=2000
        )  # should be equal to actual number of FRBs, but for this purpose it doesn't matter
    
    # set up new parameters
    g.update(vparams)
    # gets cumulative rate distribution
    
    
    
    # does plot of p(DM|z)
    ddm=g.dmvals[1]-g.dmvals[0]
    print("ddm is ",ddm)
    dz=g.zvals[1]-g.zvals[0]
    idm=int(DMEG210117/ddm)
    iz=int(Z210117/dz)
    pzgdm = g.rates[:,idm]/np.sum(g.rates[:,idm])/dz
    
    logmean=g.state.host.lmean
    
    pdmgz=g.rates[iz,:]
    dmigm=g.grid[iz,:]
    intrinsic=g.smear_grid[iz,:]
    
    peak=np.argmax(pdmgz)
    #print(peak)
    mdm=g.dmvals[peak]
    #print("Peak dm is ",)
    
    ymax=0.004
    plt.figure()
    plt.xlim(0,1500)
    plt.ylim(0,ymax)
    
    plt.plot(g.dmvals,pdmgz/np.sum(pdmgz)/ddm,label='$p({\\rm DM}_{\\rm EG}|z)$')
    plt.plot(g.dmvals,intrinsic/np.sum(intrinsic)/ddm,label='intrinsic (no bias)',linestyle='--')
    plt.plot(g.dmvals,dmigm/np.sum(dmigm)/ddm,label='$p({\\rm DM}_{\\rm IGM}|z)$',linestyle='-.')
    plt.plot([mdm,mdm],[0.,ymax],color='black',linestyle=':',label='${\\rm DM}_{\\rm EG} = 182$ pc cm$^{-3}$')
    #plt.text(0.15,0.1,"peak DM$_{\\rm EG}$ = 182",rotation=90)
    plt.yticks(np.linspace(0,0.004,5))
    plt.legend(loc='upper right')
    
    plt.ylabel('$p({\\rm DM}| z)$')
    plt.xlabel('${\\rm DM}$')
    plt.tight_layout()
    plt.savefig(opfile)
    plt.close()
    
        
    return 


main()
