""" 
This script creates zdm grids for ASKAP incoherent sum observations.

It exists partly to calculate relative rates from surveys

For CHIME 1.28, it's 2.54
For updated, it's 1.88

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
    
    # in case you wish to switch to another output directory
    
    opdir="ImprovedSurveys/"
    
    # approximate best-fit values from recent analysis
    # best-fit from Jordan et al
    state = states.load_state("HoffmannHalo25",scat="updated",rep=None)
    
    if not os.path.exists(opdir):
        os.mkdir(opdir)
    
    # Initialise surveys and grids
    sdir = resources.files('zdm').joinpath('../papers/CRACO/ImprovedSurveys')
    names=['CRAFT_CRACO_1300',
            'CRAFT_CRACO_1300_imp2.1_div2','CRAFT_CRACO_1300_imp2.1_div4','CRAFT_CRACO_1300_imp2.1_div8'
            ]
    labels = ["CRACO 1300 MHz",
               "2.1 $t_{\\rm rs}=6.8$\\,ms","2.1 $t_{\\rm rs}=3.4$\\,ms","2.1 $t_{\\rm rs}=1.7$\\,ms"
               ]
    linestyles=["-","-.","--",":"]
    nz=400
    zmax=4
    ndm=500
    dmmax=5000
    
    ss,gs = loading.surveys_and_grids(survey_names=names,repeaters=False,
                                    init_state=state,sdir=sdir,
                                    zmax=zmax,nz=nz,dmmax=dmmax,ndm=ndm) 
    
    
    ##### prints total relative rates #####
    for i,n in enumerate(names):
        print("Total rate for survey ",labels[i]," is ",np.sum(gs[i].rates)/np.sum(gs[0].rates))
    
    
    
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
        #figures.plot_grid(gs[i].rates,g.zvals,g.dmvals,
        #    name=opdir+name+"_zDM.pdf",norm=3,log=True,
        #    label='$\\log_{10} p({\\rm DM}_{\\rm IGM} + {\\rm DM}_{\\rm host},z)$ [a.u.]',
        #    project=False,ylabel='${\\rm DM}_{\\rm IGM} + {\\rm DM}_{\\rm host}$',
        #    zmax=zmax,DMmax=DMmax,Aconts=[0.01,0.1,0.5])
        
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
        pz = pzs[i]/np.max(pzs[1])
        pdm = pdms[i]/np.max(pdms[1])
        
        print("Relative rate for ",names[i]," is ",allrates[i]/allrates[0]," per day")
        
        plt.sca(ax1)
        plt.plot(zvals,pz,label=labels[i],linestyle=linestyles[i])
        
        plt.sca(ax2)
        plt.plot(dmvals,pdm,label=labels[i],linestyle=linestyles[i])
    
        
               
    plt.sca(ax1)
    plt.legend(fontsize=6)
    plt.tight_layout()
    plt.savefig("ImprovedSurveys/improved_1300_zs.png")
    plt.close()
    
    plt.sca(ax2)
    plt.legend(fontsize=6)
    plt.tight_layout()
    plt.savefig("ImprovedSurveys/improved_1300_dms.png")
    plt.close()
    
def plot_efficiencies(gs,ss):
    """
    Does some efficiency plots
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
    """
    ###### Checks normalisation ######
    ss,gs = loading.surveys_and_grids(
        survey_names=["CRAFT_class_I_and_II"],repeaters=False,init_state=state) # should be equal to actual number of FRBs, but for this purpose it doesn't matter
    
    rate = np.sum(gs[0].rates) * 10**gs[0].state.FRBdemo.lC  * ss[0].TOBS
    print("Expected number for Fly's Eys is ",rate," per day")
    print("c.f. actual number: ",ss[0].NORM_FRB)
    

    
main()
