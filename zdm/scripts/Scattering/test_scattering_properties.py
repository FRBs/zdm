""" 
This script is intended to illustrate the different
ways in which scattering can be implemented in zDM

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
from pkg_resources import resource_filename
import numpy as np
from zdm import survey
from matplotlib import pyplot as plt

import matplotlib

defaultsize=14
ds=4
font = {'family' : 'Helvetica',
        'weight' : 'normal',
        'size'   : defaultsize}
matplotlib.rc('font', **font)

def main():

    # in case you wish to switch to another output directory
    opdir = "Plots/"
    if not os.path.exists(opdir):
        os.mkdir(opdir)
    
    # directory where the survey files are located. The below is the default - 
    # you can leave this out, or change it for a different survey file location.
    sdir = os.path.join(resource_filename('zdm', 'data'), 'Surveys')
    
    survey_name = "CRAFT_average_ICS"
    
    # make this into a list to initialise multiple surveys art once
    names = [survey_name]
    
    repeaters=True
    # sets plotting limits
    zmax = 2.
    dmmax = 2000
    
    # initialise surveys and grids with different state values
    imethods = np.arange(4)
    methnames = ["const","width", "scat","zscat"]
    zdists=[]
    dmdists=[]
    
    if repeaters:
        zdists_s=[]
        dmdists_s=[]
        zdists_r=[]
        dmdists_r=[]
        zdists_b=[]
        dmdists_b=[]
    
    for i,Method in enumerate(imethods):
        
        survey_dict = {"WMETHOD": Method}
        # Write True if you want to do repeater grids - see "plot_repeaters.py" to make repeater plots
        surveys, grids = loading.surveys_and_grids(survey_names = names,\
                        repeaters=repeaters, sdir=sdir,nz=70,ndm=140,
                        survey_dict = survey_dict)
        g=grids[0]
        s=surveys[0]
        
        llsum = it.calc_likelihoods_1D(g,s,pNreps=False)
        print("For scattering method ",methnames[i],", 1D likelihoods ",llsum)
        
        llsum = it.calc_likelihoods_2D(g,s,pNreps=False)
        print("For scattering method ",methnames[i],", 2D likelihoods are ",llsum)
        
        # extracts weights from survey and plots as function of z
        if i==3:
            weights = s.wplist
            widths = s.wlist[:,0]
            nw,nz = weights.shape
            plt.figure()
            plt.xlabel('z')
            plt.ylabel('weight')
            for iw in np.arange(nw):
                plt.plot(g.zvals,weights[iw,:],label="width = "+str(widths[iw])[0:5])
            total = np.sum(weights,axis=0)
            plt.plot(g.zvals,total,label="total",color="black")
            plt.yscale("log")
            plt.legend()
            plt.tight_layout()
            plt.savefig(opdir+"z_dependent_weights.png")
            plt.close()
            
        figures.plot_grid(
                g.rates,
                g.zvals,
                g.dmvals,
                name=opdir+"pzdm_"+methnames[i]+".png",
                norm=3,
                log=True,
                label='$\\log_{10} p({\\rm DM}_{\\rm cosmic} + {\\rm DM}_{\\rm host},z)$ [a.u.]',
                project=True,
                ylabel='${\\rm DM}_{\\rm cosmic} + {\\rm DM}_{\\rm host}$',
                zmax=zmax,DMmax=dmmax,
                Aconts=[0.01, 0.1, 0.5]
                )
        zdists.append(np.sum(g.rates,axis=1))
        dmdists.append(np.sum(g.rates,axis=0))
        
        if repeaters:
            zdists_s.append(np.sum(g.exact_singles,axis=1))
            dmdists_s.append(np.sum(g.exact_singles,axis=0))
            
            zdists_r.append(np.sum(g.exact_reps,axis=1))
            dmdists_r.append(np.sum(g.exact_reps,axis=0))
            
            zdists_b.append(np.sum(g.exact_rep_bursts,axis=1))
            dmdists_b.append(np.sum(g.exact_rep_bursts,axis=0))
            
    
    # comparison of z distributions
    plt.figure()
    plt.xlabel('z')
    plt.ylabel('p(z) [arb units]')
    plt.xlim(0,zmax)
    plt.ylim(0,None)
    for i,Method in enumerate(imethods):
        plt.plot(g.zvals,zdists[i],label=methnames[i])
    plt.legend()
    plt.tight_layout()
    plt.savefig(opdir+"pdm_comparison.png",)
    plt.close()
    
    plt.figure()
    plt.xlabel('DM')
    plt.xlim(0,dmmax)
    plt.ylim(0,None)
    plt.ylabel('p(DM) [arb units]')
    for i,Method in enumerate(imethods):
        plt.plot(g.dmvals,dmdists[i],label=methnames[i])
    plt.legend()
    plt.tight_layout()
    plt.savefig(opdir+"pz_comparison.png",)
    plt.close()
    
    if not repeaters:
        exit()
    
    ### singles ###
    # comparison of z distributions
    plt.figure()
    plt.xlabel('z')
    plt.xlim(0,zmax)
    plt.ylim(0,None)
    plt.ylabel('p(z) [arb units]')
    for i,Method in enumerate(imethods):
        plt.plot(g.zvals,zdists_s[i],label=methnames[i])
    plt.legend()
    plt.tight_layout()
    plt.savefig(opdir+"pdm_singles_comparison.png",)
    plt.close()
    
    plt.figure()
    plt.xlabel('DM')
    plt.xlim(0,dmmax)
    plt.ylim(0,None)
    plt.ylabel('p(DM) [arb units]')
    for i,Method in enumerate(imethods):
        plt.plot(g.dmvals,dmdists_s[i],label=methnames[i])
    plt.legend()
    plt.tight_layout()
    plt.savefig(opdir+"pz_singles_comparison.png",)
    plt.close()
    
    
    ### repeaters ###
    # comparison of z distributions
    plt.figure()
    plt.xlabel('z')
    plt.xlim(0,zmax)
    plt.ylim(0,None)
    plt.ylabel('p(z) [arb units]')
    for i,Method in enumerate(imethods):
        plt.plot(g.zvals,zdists_r[i],label=methnames[i])
    plt.legend()
    plt.tight_layout()
    plt.savefig(opdir+"pdm_repeater_comparison.png",)
    plt.close()
    
    plt.figure()
    plt.xlabel('DM')
    plt.xlim(0,dmmax)
    plt.ylim(0,None)
    plt.ylabel('p(DM) [arb units]')
    for i,Method in enumerate(imethods):
        plt.plot(g.dmvals,dmdists_r[i],label=methnames[i])
    plt.legend()
    plt.tight_layout()
    plt.savefig(opdir+"pz_repeater_comparison.png",)
    plt.close()
    
    ### bursts ###
    # comparison of z distributions
    plt.figure()
    plt.xlabel('z')
    plt.xlim(0,zmax)
    plt.ylim(0,None)
    plt.ylabel('p(z) [arb units]')
    for i,Method in enumerate(imethods):
        plt.plot(g.zvals,zdists_b[i],label=methnames[i])
    plt.legend()
    plt.tight_layout()
    plt.savefig(opdir+"pdm_bursts_comparison.png",)
    plt.close()
    
    plt.figure()
    plt.xlabel('DM')
    plt.xlim(0,dmmax)
    plt.ylim(0,None)
    plt.ylabel('p(DM) [arb units]')
    for i,Method in enumerate(imethods):
        plt.plot(g.dmvals,dmdists_b[i],label=methnames[i])
    plt.legend()
    plt.tight_layout()
    plt.savefig(opdir+"pz_bursts_comparison.png",)
    plt.close()
    
    
    
    
    
main()
