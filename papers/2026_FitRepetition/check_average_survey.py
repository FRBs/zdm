""" 
This script compares predictions for an averaged ASKAP survey
to three frequency-dependent surveys.

It does this by creating zDM grids and projections onto p(z),
p(DM), and NFRB



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
    opname="Average_ASKAP"
    opdir=opname+"/"
    
    # approximate best-fit values from recent analysis
    # best-fit from Jordan et al
    if True:
        # approximate best-fit values from recent analysis 0.11
        param_dict={'sfr_n': 0.21, 'alpha': 0.11, 'lmean': 2.18, 'lsigma': 0.42, 'lEmax': 41.37, 
                'lEmin': 39.47, 'gamma': -1.04, 'H0': 70.23, 'halo_method': 0, 'sigmaDMG': 0.0, 'sigmaHalo': 0.0,'lC': -7.61}
        
    else:
        # best fit from James et al
        param_dict={'sfr_n': 1.13, 'alpha': 0.99, 'lmean': 2.27, 'lsigma': 0.55, 'lEmax': 41.26, 
                    'lEmin': 32, 'gamma': -0.95, 'H0': 73, 'halo_method': 0, 'sigmaDMG': 0.0, 'sigmaHalo': 0.0,'lC': -0.76}
    
    
    if not os.path.exists(opdir):
        os.mkdir(opdir)
    
    
    state = parameters.State()
    state.set_astropy_cosmo(Planck18)
    state.update_params(param_dict)
    
    ############## Load ASKAP survey properties ###########
    
    sdir = os.path.join(resource_filename('zdm', 'data'), 'Surveys')
    
    # We now loads the ASKAP averages
    names=['CRAFT_average_ICS']
    
    ss,gs = loading.surveys_and_grids(
        survey_names=names,repeaters=False,init_state=state,sdir=sdir) # should be equal to actual number of FRBs, but for this purpose it doesn't matter
    savg = ss[0]
    gavg = gs[0]
    
    
    # with repeaters
    ss,gs = loading.surveys_and_grids(
        survey_names=names,repeaters=True,init_state=state,sdir=sdir) # should be equal to actual number of FRBs, but for this purpose it doesn't matter
    repavg = gs[0]
    
    # Initialise surveys and grids
    names=['CRAFT_ICS_892','CRAFT_ICS_1300','CRAFT_ICS_1632']
    
    ss,gs = loading.surveys_and_grids(
        survey_names=names,repeaters=False,init_state=state,sdir=sdir) # should be equal to actual number of FRBs, but for this purpose it doesn't matter
    
    
    # gets sum of rates over three sets of observations
    # weights by constant and TOBS
    time=0
    mean_rates = 0
    for i,g in enumerate(gs):
        this_rate=g.rates * ss[i].TOBS * 10**g.state.FRBdemo.lC
        
        setDMzero = np.where(g.dmvals + g.ddm/2. > ss[i].max_dm)[0]
        print("survey ",i," max dm is ",ss[i].max_dm)
        print(setDMzero)
        this_rate[:,setDMzero]=0.
        
        mean_rates += this_rate
        
    ############ Generate comparison plots ######## 
    
    # set limits for plots
    DMmax=3000
    zmax=1.5
    
    plt.figure()
    ax1 = plt.gca()
    
    plt.figure()
    ax2 = plt.gca()
    
    # chooses the first arbitrarily to extract zvals etc from
    s=ss[0]
    g=gs[0]
    name = names[0]
    misc_functions.plot_grid_2(mean_rates,g.zvals,g.dmvals,
        name=opdir+"three_freq_zDM.pdf",norm=3,log=True,
        label='$\\log_{10} p({\\rm DM}_{\\rm IGM} + {\\rm DM}_{\\rm host},z)$ [a.u.]',
        project=False,ylabel='${\\rm DM}_{\\rm IGM} + {\\rm DM}_{\\rm host}$',
        zmax=zmax,DMmax=DMmax,Aconts=[0.01,0.1,0.5])
    
    gavg_rates = gavg.rates * savg.TOBS * 10**g.state.FRBdemo.lC
    repavg_rates = repavg.exact_singles * repavg.Rc * repavg.Nfields
    misc_functions.plot_grid_2(gavg_rates,g.zvals,g.dmvals,
        name=opdir+"Average_zDM.pdf",norm=3,log=True,
        label='$\\log_{10} p({\\rm DM}_{\\rm IGM} + {\\rm DM}_{\\rm host},z)$ [a.u.]',
        project=False,ylabel='${\\rm DM}_{\\rm IGM} + {\\rm DM}_{\\rm host}$',
        zmax=zmax,DMmax=DMmax,Aconts=[0.01,0.1,0.5])
    
    ###### gets the average p(z) distribution for the different surveys ######
    
    # average survey
    pz = np.sum(mean_rates,axis=1)
    #pz /= np.max(pz)
    
    # sum of other surveys
    avg_pz = np.sum(gavg_rates,axis=1)
    
    # repeaters
    rep_pz = np.sum(repavg_rates,axis=1)
    
    #avg_pz /= np.max(pz)
    ax1.plot(g.zvals,pz,label="Sum")
    ax1.plot(g.zvals,avg_pz,label="Average",linestyle="--")
    ax1.plot(g.zvals,rep_pz,label="(inc. repeaters)",linestyle=":")
    
    # cumulative distribution
    cpz = np.cumsum(pz)
    cpz /= cpz[-1]
    
    pdm = np.sum(mean_rates,axis=0)
    #pdm /= np.max(pdm)
    avg_pdm = np.sum(gavg_rates,axis=0)
    #avg_pdm /= np.max(pdm)
    
    ax2.plot(g.dmvals,pdm,label="Sum")
    ax2.plot(g.dmvals,avg_pdm,label="Average")
    
    total = np.sum(mean_rates)
    print(name," expected to detect ",total," in ",time," hr")
    
    plt.sca(ax1)
    plt.xlabel("z")
    plt.ylabel("p(z)")
    plt.xlim(0.01,1.5)
    plt.ylim(0,0.8)
    plt.legend()
    plt.tight_layout()
    plt.savefig(opdir+opname+"_pz.pdf")
    plt.close()
    
    plt.sca(ax2)
    plt.xlabel("DM")
    plt.ylabel("p(DM)")
    plt.xlim(0,4000)
    plt.legend()
    plt.tight_layout()
    plt.savefig(opdir+opname+"_pdm.pdf")
    plt.close()
    


    
main()
