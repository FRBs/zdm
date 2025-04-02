""" 
This script creates a redshifty comparison figure of MeerTRAP,
ASKAP/CRACO (estimates), DSA, and CHIME


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

import matplotlib


defaultsize=14
ds=4
font = {'family' : 'Helvetica',
        'weight' : 'normal',
        'size'   : defaultsize}
matplotlib.rc('font', **font)


def main():
    
    # in case you wish to switch to another output directory
    
    opdir='Inzcomparison/'
    
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
    names=["MeerTRAPincoherent","DSA","CRAFT_ICS_1300"]
    
    state = parameters.State()
    state.set_astropy_cosmo(Planck18)
    state.update_params(param_dict)
    
    ss,gs = loading.surveys_and_grids(
        survey_names=names,repeaters=False,init_state=state,sdir=sdir) # should be equal to actual number of FRBs, but for this purpose it doesn't matter
    
    
    if False:
        #### plots p(z|DM) ####
        DMfrb = 2398
        iDM = np.where(DMfrb < gs[0].dmvals)[0][0]
        pzgdm = gs[0].rates[:,iDM]
        
        pzgdm /= np.sum(pzgdm)
        plt.figure()
        plt.xlabel("z")
        plt.ylabel("p(z|DMEG = 2398)")
        plt.plot(gs[0].zvals,pzgdm)
        plt.tight_layout()
        plt.savefig(opdir+"pzgdm.png")
        plt.close()
        
        np.save("pzgdm.npy",pzgdm)
        np.save("zvals.npy",gs[0].zvals)
    
    # set limits for plots - will be LARGE!   
    
    ######### Loads public FRBs #######
    
    if False:
        from frb.galaxies import utils as frb_gal_u
        
        # Load up the hosts
        host_tbl, _ = frb_gal_u.build_table_of_hosts(attrs=['redshift'])
        
        # Cut
        host_tbl = host_tbl[host_tbl['P_Ox'] > POx_min]
        
        # DMs
        DM_FRB = units.Quantity([frb.DM for frb in host_tbl.FRBobj.values])
        DM_ISM = units.Quantity([frb.DMISM for frb in host_tbl.FRBobj.values])
        DM_EG = DM_FRB - DM_ISM - DM_MWhalo
    
    
    ########### Get CHIME info ###########
    
    # defines CHIME grids to load
    NDECBINS=6
    cnames=[]
    for i in np.arange(NDECBINS):
        cname="CHIME_decbin_"+str(i)+"_of_6"
        cnames.append(cname)
    survey_dir = os.path.join(resource_filename('zdm', 'data'), 'Surveys/CHIME/')
    css,cgs = loading.surveys_and_grids(survey_names=cnames, init_state=state, rand_DMG=False,sdir = survey_dir, repeaters=True)
    
    # compiles sums over all six declination bins
    crates = cgs[0].rates * 10**cgs[0].state.FRBdemo.lC * css[0].TOBS
    creps = cgs[0].exact_reps * cgs[0].state.rep.RC
    csingles = cgs[0].exact_singles * cgs[0].state.rep.RC
    
    for i,g in enumerate(cgs):
        s = css[i]
        if i ==0:
            continue
        else:
            crates += g.rates * 10**g.state.FRBdemo.lC * s.TOBS
            creps += g.exact_reps * g.state.rep.RC
            csingles += g.exact_singles * g.state.rep.RC
    
    
    ###### plots MeerTRAP zDM figure ###########
    s=ss[0]
    g=gs[0]
    name = names[0]
    misc_functions.plot_grid_2(g.rates,g.zvals,g.dmvals,
        name=opdir+name+"_zDM.pdf",norm=3,log=True,
        label='$\\log_{10} p({\\rm DM}_{\\rm IGM} + {\\rm DM}_{\\rm host},z)$ [a.u.]',
        project=False,ylabel='${\\rm DM}_{\\rm IGM} + {\\rm DM}_{\\rm host}$',
        zmax=5,DMmax=5000,Aconts=[0.01],othergrids=[gs[1].rates,crates,gs[2].rates],
        othernames = ["MeerKAT","DSA","CHIME","ASKAP"])
        #0.01, 0.1,0.5
    
    
    ############ Plots z projection ##########
    plt.figure()
    
    names = ["MeerTRAP coherent", "DSA 110", "ASKAP ICS"]
    styles=["-","--","-."]
    for i,g in enumerate(gs):
        s=ss[i]
        
        pz = np.sum(g.rates,axis=1)
        pz /= np.max(pz)
        plt.plot(g.zvals,pz,label=names[i],linestyle=styles[i],linewidth=2)
    
    # adds CHIME
    pz = np.sum(crates,axis=1)
    pz /= np.max(pz)
    plt.plot(g.zvals,pz,label="CHIME",linestyle=":",linewidth=2)
    
    plt.xlabel("z")
    plt.ylabel("p(z)")
    plt.xlim(0.,3)
    plt.ylim(0,1)
    plt.legend()
    plt.tight_layout()
    plt.savefig(opdir+"pz_comparison.pdf")
    plt.close()
    

    
main()
