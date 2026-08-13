""" 
This script creates plots of p(z) and p(dm) for different SKA configs

"""
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
    """
    Plots outputs of simulations
    
    """
    
    
    ####### Loop over input files #########
    # these set the frequencies in MHz and bandwidths in MHz
    names = ["SKA_mid","SKA_mid","SKA_low"]
    
    datadir = "sys_outputs/"
    plotdir = "sysplotdir/"
    
    
    plt.figure()
    plt.xlabel("z")
    plt.ylabel("N(z) per year")
    plt.xlim(0,5)
    
    for i,tel in enumerate(["Band1", "Band2", "Low"]):
        # sets frequency and bandwidth for each instrument
        for telconfig in ["AA4","AAstar"]:
            label = tel+"_"+telconfig
            print("\n\n\n DOING ",label," ########")
            zvals,plow,pmid,phigh = make_plots(label,datadir=datadir)
            if telconfig == "AA4":
                if i==2:
                    plotlabel="SKA Low "+telconfig
                else:
                    plotlabel="SKA Mid "+tel+" "+telconfig
                plt.fill_between(zvals,plow,phigh,linestyle=":",linewidth=1,label=plotlabel,alpha=0.5)
                plt.plot(zvals,pmid,linestyle="-",linewidth=2)
    plt.ylim(1,1e4)
    plt.legend()
    plt.yscale("log")
    plt.tight_layout()
    plt.savefig("all_pz.png")
    plt.close()
    
def make_plots(label,datadir="sys_outputs/",plotdir="sysplotdir/"):
    """
    
    Args:
        label (string): string label identifying the band and config
            of the SKA data to load, and tag to apply to the
            output files
        
    """
    
    # load redshift and dm values
    zvals = np.load(datadir+"zvals.npy")
    dmvals = np.load(datadir+"dmvals.npy")
    
    # load survey-specific outputs
    Ns = np.load(datadir+label+"_sys_N.npy")
    meanN = np.sum(Ns)/Ns.size
    print("Mean annual event rate for ",label," is ",meanN)
    
    pzs = np.load(datadir+label+"_sys_pz.npy")
    pdms = np.load(datadir+label+"_sys_pdm.npy")
    
    plow,pmid,phigh = make_pz_plots(zvals,pzs,plotdir+label)
    make_pdm_plots(dmvals,pdms,plotdir+label)

    return zvals,plow,pmid,phigh
    
def make_pz_plots(zvals,pzs,label):
    """
    Make plots of p(z) for each systematic simulation
    """
    
    Nparams,NZ = pzs.shape
    
    # this scales from the "per z bin" to "per z",
    # i.e. to make the units N per year per dz
    scale = 1./(zvals[1]-zvals[0])
    
    mean = np.sum(pzs,axis=0)/Nparams
    
    # total estimates
    Ntots = np.sum(pzs,axis=1)
    Nordered = np.sort(Ntots)
    Nbar = np.sum(Ntots)/Nparams
    sigma1 = Nordered[15]
    sigma2 = Nordered[83]
    print("Range for Ntotal is ",sigma1-Nbar,Nbar,sigma2-Nbar)
    
    # constructs intervals - does this on a per-z basis
    # first sorts over the axis of different simulations
    zordered = np.sort(pzs,axis=0)
    pzlow = zordered[15,:]
    pzhigh = zordered[83,:]
    
    # make un-normalised plots
    plt.figure()
    plt.xlabel("z")
    plt.ylabel("N(z) per year")
    plt.xlim(0,5)
    
    themax = np.max(pzs)
    plt.ylim(0,themax*scale)
    for i in np.arange(Nparams):
        plt.plot(zvals,pzs[i,:]*scale,color="grey",linestyle="-")
    
    plt.plot(zvals,mean*scale,color="black",linestyle="-",linewidth=2,label="Simulation mean")
    
    plt.legend()
    plt.tight_layout()
    plt.savefig(label+"_pz.png")
    plt.close()
    
    # prints some summary statistics in z-space
    # calculates mean z
    zbar = zvals * mean / np.sum(mean)
    last=0.
    tot = np.sum(mean)
    for z in np.arange(5)+1:
        OK = np.where(zvals < z)
        Nthis = np.sum(mean[OK])
        N = Nthis -last
        print("FRBs from ",z-1," to ",z,": ",N/tot," %")
        last = Nthis
    
    return pzlow*scale,mean*scale,pzhigh*scale
    
def make_pdm_plots(dmvals,pdms,label):
    """
    Make plots of p(DM) for each systematic simulation
    """
    
    Nparams,NDM = pdms.shape
    
    # this scales from the "per z bin" to "per z",
    # i.e. to make the units N per year per dz
    scale = 1./(dmvals[1]-dmvals[0])
    
    mean = np.sum(pdms,axis=0)/Nparams
    
    # make un-normalised plots
    plt.figure()
    plt.xlabel("DM [pc cm$^{-3}$]")
    plt.ylabel("N(DM) per year")
    plt.xlim(0,5000)
    
    themax = np.max(pdms)
    plt.ylim(0,themax*scale)
    
    for i in np.arange(Nparams):
        plt.plot(dmvals,pdms[i,:]*scale,color="grey",linestyle="-")
    
    plt.plot(dmvals,mean*scale,color="black",linestyle="-",linewidth=2,label="Simulation mean")
    
    plt.legend()
    plt.tight_layout()
    plt.savefig(label+"_pdm.png")
    plt.close()   

main()
