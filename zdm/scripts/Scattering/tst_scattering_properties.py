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
    
    repeaters=False
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
        state_dict = {}
        state_dict["scat"] = {}
        state_dict["scat"]["Sbackproject"] = True # turns on backprojection of tau and width for our model
        
        # Write True if you want to do repeater grids - see "plot_repeaters.py" to make repeater plots
        surveys, grids = loading.surveys_and_grids(survey_names = names,\
                        repeaters=repeaters, sdir=sdir,nz=70,ndm=140,
                        survey_dict = survey_dict, state_dict = state_dict)
        g=grids[0]
        s=surveys[0]
        
        llsum = it.calc_likelihoods_1D(g,s,pNreps=False)
        print("For scattering method ",methnames[i],", 1D likelihoods ",llsum)
        
        llsum = it.calc_likelihoods_2D(g,s,pNreps=False)
        print("For scattering method ",methnames[i],", 2D likelihoods are ",llsum)
        
        if i==2:
            # gets constant weights from plot
            const_weights = s.wplist
        
        # extracts weights from survey and plots as function of z
        if i==3:
            weights = s.wplist # 2D
            widths = s.wlist # widths are now 1D, they don't vary
            nw,nz = weights.shape
            plt.figure()
            plt.xlabel('z')
            plt.ylabel('weight')
            for iw in np.arange(nw):
                plt.plot(g.zvals,weights[iw,:],label="width = "+str(widths[iw])[0:5])
                plt.plot([g.zvals[0],g.zvals[-1]],[const_weights[iw],const_weights[iw]],
                        color=plt.gca().lines[-1].get_color(),linestyle=":")
            total = np.sum(weights,axis=0)
            plt.plot(g.zvals,total,label="total",color="black")
            plt.legend(fontsize=6)
            plt.tight_layout()
            plt.savefig(opdir+"z_dependent_weights_lin.png")
            
            plt.yscale("log")
            plt.tight_layout()
            plt.savefig(opdir+"z_dependent_weights_log.png")
            plt.close()
            
            
            # plots back-projected probabilities
            plt.figure()
            ax1 = plt.gca()
            plt.figure()
            ax2 = plt.gca()
            
            # gets zvalues correponding to 0.1,0.5,1,2
            izs = []
            for z in [ 0.1,0.5,1,2]:
                # gets index of the above redshift values
                iz = np.where(g.zvals>z)[0][0]
                izs.append(iz)
            styles=["-","--","-.",":"]
            for iw in np.arange(s.NWbins):
                label=str(s.wlist[iw])[0:5]
                for j,iz in enumerate(izs):
                    if j==0:
                        ax1.plot(s.internal_logwvals,s.ptaus[iz,:,iw],label=label,
                                linestyle = styles[j])
                        ax2.plot(s.internal_logwvals,s.pws[iz,:,iw],label=label,
                                linestyle = styles[j])
                    else:
                        ax1.plot(s.internal_logwvals,s.ptaus[iz,:,iw],label=label,
                                linestyle = styles[j],color=plt.gca().lines[-1].get_color())
                        ax2.plot(s.internal_logwvals,s.pws[iz,:,iw],label=label,
                                linestyle = styles[j],color=plt.gca().lines[-1].get_color())
                    label=None
            plt.sca(ax1)
            plt.xlabel("Natural log of scattering width (observed)")
            plt.ylabel("Probability given total width and redshift")
            plt.legend(fontsize=6)
            plt.tight_layout()
            plt.savefig(opdir+"z_dependent_ptau.png")
            plt.close()
            
            
            plt.sca(ax2)
            plt.xlabel("Natural log of intrinsic width (observed)")
            plt.ylabel("Probability given total width and redshift")
            plt.legend(fontsize=6)
            plt.tight_layout()
            plt.savefig(opdir+"z_dependent_pw.png")
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
    ymax=0.
    for i,Method in enumerate(imethods):
        themax = np.max(zdists[i])
        if ymax < themax:
            ymax = themax
        plt.plot(g.zvals,zdists[i],label=methnames[i])
    plt.ylim(0,ymax)
    plt.legend()
    plt.tight_layout()
    plt.savefig(opdir+"pz_comparison.png",)
    plt.close()
    
    plt.figure()
    plt.xlabel('DM')
    plt.xlim(0,dmmax)
    plt.ylabel('p(DM) [arb units]')
    ymax = 0.
    for i,Method in enumerate(imethods):
        themax = np.max(dmdists[i])
        if ymax < themax:
            ymax = themax
        plt.plot(g.dmvals,dmdists[i],label=methnames[i])
    
    plt.ylim(0,ymax)
    plt.legend()
    plt.tight_layout()
    plt.savefig(opdir+"pzdm_comparison.png",)
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
