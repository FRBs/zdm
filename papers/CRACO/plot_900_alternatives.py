""" 
This script  plots the resulting redshift and dm distributions
from various alternative hypothertical CRACO setups,
in order to estimate the effects on the system

This one is limited to 900 MHz observations

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
from zdm import states
import matplotlib

defaultsize=14
ds=4
font = {'family' : 'Helvetica',
        'weight' : 'normal',
        'size'   : defaultsize}
matplotlib.rc('font', **font)


import numpy as np
from zdm import survey
from matplotlib import pyplot as plt
import importlib.resources as resources

def main():
    """
    Main function - it loads in various surveys, and calculates their rates.
    """
    
    # common features - zdm state, directoruy for surveys, dm and z range, etc
    state = states.load_state("HoffmannRepeaters26Pn",scat="updated",rep=None)
    sdir = resources.files('zdm').joinpath('../papers/CRACO/Surveys')
    opdir="TestSurveys/"
    nz=400
    zmax=4
    ndm=500
    dmmax=5000
    #do_dm_width(state,sdir,opdir,nz,zmax,ndm,dmmax)
    do_comparison_plots(state,sdir,opdir,nz,zmax,ndm,dmmax)
    
def do_dm_width(state,sdir,opdir,nz,zmax,ndm,dmmax):
    """
    see descriptions above
    """
    ###############################################################
    ############### PART 1: DM and width efficiency   #############
    ###############################################################
    
    dm_ratios=[]
    w_ratios=[]
    itsamps = [2,4,8,16,64]
    for itsamp in itsamps:
        name1 = 'CRACO_900_itsamp_'+str(itsamp)
        name2 = name1+"_nodm"
        name3 = name1+"_icsw"
        names=[name1,name2,name3]
        ss,gs = loading.surveys_and_grids(survey_names=names,repeaters=False,
                                    init_state=state,sdir=sdir,
                                    zmax=zmax,nz=nz,dmmax=dmmax,ndm=ndm)
        R1 = np.sum(gs[0].get_rates())
        R2 = np.sum(gs[1].get_rates())
        R3 = np.sum(gs[2].get_rates())
        dm_ratios.append(R1/R2)
        w_ratios.append(R1/R3)
    np.save("900_nodm.npy",dm_ratios)
    np.save("900_icsw.npy",w_ratios)
    for i,itsamp in enumerate(itsamps):
        print("For itsamp ",itsamp," at 900 MHz, DM efficiency is ",dm_ratios[i]," width eff is ",w_ratios[i])

def do_comparison_plots(state,sdir,opdir,nz,zmax,ndm,dmmax):
    """
    see descriptions above
    """    
    ###############################################################
    ####################### PART 2: Plotting   ####################
    ###############################################################
    
    # in case you wish to switch to another output directory
    
    
    if not os.path.exists(opdir):
        os.mkdir(opdir)
    
    # Initialise surveys and grids
    
    names=['CRACO_900_itsamp_8','CRACO_900_itsamp_8_nodm','CRACO_900_itsamp_2',\
        'CRACO_900_itsamp_2_nodm','CRACO_900_ics_like']
    
    
    #names=['CRAFT_CRACO_900','CRAFT_CRACO_900_alldm','CRAFT_CRACO_900_icsdm',
    #        'CRAFT_CRACO_900_1.28','CRAFT_CRACO_900_alldm_1.28',
    #        'CRAFT_CRACO_900_3ms','CRAFT_CRACO_900_3ms_alldm','CRAFT_CRACO_900_3ms_at_1.28_alldm']
    labels = ["900 MHz: 13.8 ms","           all DMs","900 MHz: 3.5 ms","           all DMs","ICS-like"]
    linestyles=["-","-.","--",":","-","-.","--",":"]
    
    
    ss,gs = loading.surveys_and_grids(survey_names=names,repeaters=False,
                                    init_state=state,sdir=sdir,
                                    zmax=zmax,nz=nz,dmmax=dmmax,ndm=ndm) 
    
    
    ##### prints total relative rates #####
    for i,n in enumerate(names):
        print("Total rate for survey ",n," is ",np.sum(gs[i].rates)/np.sum(gs[0].rates))
    
    
    
    ######### plots total DM and z distribution #######
    # set limits for plots - will be LARGE!   
    DMmax=4000
    zmax=4.
    
    plt.figure()
    ax1 = plt.gca()
    plt.xlabel("redshift $z$")
    plt.ylabel("p(z) [a.u.]")
    plt.xlim(0.01,3)
    plt.ylim(0,1)
    #plt.ylim(0,80)
    
    plt.figure()
    ax2 = plt.gca()
    plt.xlabel("DM pc cm$^{-3}$")
    plt.ylabel("p(DM) [a.u.]")
    plt.xlim(0,3000)
    plt.ylim(0,1)
    #plt.ylim(0,0.0009)
    
    zvals = gs[0].zvals
    dz = zvals[1]-zvals[0]
    dmvals = gs[0].dmvals
    ddm = dmvals[1]-dmvals[0]
    
    pzs=[]
    pdms=[]
    allrates=[]
    # chooses the first arbitrarily to extract zvals etc from
    for i,g in enumerate(gs):
        
        s=ss[i]
        g=gs[i]
        name = names[i]
        figures.plot_grid(gs[i].rates,g.zvals,g.dmvals,
            name=opdir+name+"_zDM.pdf",norm=3,log=True,
            label='$\\log_{10} p({\\rm DM}_{\\rm IGM} + {\\rm DM}_{\\rm host},z)$ [a.u.]',
            project=False,ylabel='${\\rm DM}_{\\rm IGM} + {\\rm DM}_{\\rm host}$',
            zmax=zmax,DMmax=DMmax,Aconts=[0.01,0.1,0.5])
        
        rates = gs[i].get_rates() #gs[i].rates * 10**g.state.FRBdemo.lC 
        rate = np.sum(rates)
        allrates.append(rate)
        pz = np.sum(rates,axis=1)
        pz /= dz
        
        pdm = np.sum(rates,axis=0)
        pdm /= ddm
        
        pzs.append(pz)
        pdms.append(pdm)
    
    for i,g in enumerate(gs):
        pz = pzs[i]/np.max(pzs)
        pdm = pdms[i]/np.max(pdms)
        
        print("Relative rate for ",names[i]," is ",allrates[i]/allrates[0]," per day")
        
        plt.sca(ax1)
        plt.plot(zvals,pz,label=labels[i],linestyle=linestyles[i])
        
                     
        plt.sca(ax2)
        plt.plot(dmvals,pdm,label=labels[i],linestyle=linestyles[i])
        
    
    
    plt.sca(ax1)
    plt.legend()
    plt.tight_layout()
    plt.savefig("Plots/CRACO900_dmcomparison.png")
    plt.close()
    
    plt.sca(ax2)
    plt.legend()
    plt.tight_layout()
    plt.savefig("Plots/CRACO900_zcomparison.png")
    plt.close()
    
def plot_efficiencies(gs,ss):
    """
    Does some efficiency plots
    
    Args:
        gs [list of grids]: grid objects to plot
        ss [ list of surveys]: survey objects to plot
    """
    ###### plots efficiencies ######
    plt.figure()
    for i,s in enumerate(ss):
        
        for j in np.arange(s.NWbins):
            if j==0:
                plt.plot(s.dmvals,s.efficiencies[j,:],linestyle=linestyles[i],label=labels[i])
            else:
                plt.plot(s.dmvals,s.efficiencies[j,:],linestyle=linestyles[i],color=plt.gca().lines[-1].get_color())
    plt.xlabel("DM")
    plt.ylabel("Efficiency")
    plt.legend()
    plt.tight_layout()
    plt.savefig("Plots/efficiency.png")
    plt.close()
    
    
    ##### Plots an example of the threshold ######
    plt.figure()
    for i,g in enumerate(gs):
        print("Survey weights are ",ss[i].wlist,ss[i].wplist)
        for j in np.arange(g.nthresh):
            if j==0:
                plt.plot(g.dmvals,g.thresholds[j,10,:],linestyle=linestyles[i],label=labels[i],linewidth=0.2)
            else:
                plt.plot(g.dmvals,g.thresholds[j,10,:],linestyle=linestyles[i],color=plt.gca().lines[-1].get_color(),linewidth=j)
    plt.xlabel("DM")
    plt.ylabel("Threshold (erg)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("Plots/g_thresholds.png")
    plt.close()
     
    
def check_FE(state):
    """
    Checks FRB rate compared to Fly's Eye rate, which is the most reliable and consistent
    
    Args:
        state: zDM state object, to be used for Fly's Eye rate calculation
    
    """
    ###### Checks normalisation ######
    ss,gs = loading.surveys_and_grids(
        survey_names=["CRAFT_class_I_and_II"],repeaters=False,init_state=state) # should be equal to actual number of FRBs, but for this purpose it doesn't matter
    
    rate = np.sum(gs[0].rates) * 10**gs[0].state.FRBdemo.lC  * ss[0].TOBS
    print("Expected number for Fly's Eys is ",rate," per day")
    print("c.f. actual number: ",ss[0].NORM_FRB)
    

    
main()
