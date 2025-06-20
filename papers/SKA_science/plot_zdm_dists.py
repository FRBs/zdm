""" 
This script creates plots of p(z) and p(dm) for different SKA configs

"""
import numpy as np
from matplotlib import pyplot as plt

def main():
    """
    Plots outputs of simulations
    
    """
    
    
    ####### Loop over input files #########
    # these set the frequencies in MHz and bandwidths in MHz
    names = ["SKA_mid","SKA_mid","SKA_low"]
    
    for i,tel in enumerate(["Band1", "Band2", "Low"]):
        # sets frequency and bandwidth for each instrument
        for telconfig in ["AA4","AAstar"]:
            label = tel+"_"+telconfig
            make_plots(label)
            
    
def make_plots(label):
    """
    
    Args:
        label (string): string label identifying the band and config
            of the SKA data to load, and tag to apply to the
            output files
        
    """
    
    # load redshift and dm values
    zvals = np.load("zvals.npy")
    dmvals = np.load("dmvals.npy")
    
    # load survey-specific outputs
    Ns = np.load(label+"_sys_N.npy")
    meanN = np.sum(Ns)/Ns.size
    print("Mean annual evenbt rate for ",label," is ",meanN)
    
    pzs = np.load(label+"_sys_pz.npy")
    pdms = np.load(label+"_sys_pdm.npy")
    
    make_pz_plots(zvals,pzs,label)
    make_pdm_plots(dmvals,pdms,label)


def make_pz_plots(zvals,pzs,label):
    """
    Make plots of p(z) for each systematic simulation
    """
    
    Nparams,NZ = pzs.shape
    
    # this scales from the "per z bin" to "per z",
    # i.e. to make the units N per year per dz
    scale = 1./(zvals[1]-zvals[0])
    
    mean = np.sum(pzs,axis=0)/Nparams
    
    # make un-normalised plots
    plt.figure()
    plt.xlabel("z")
    plt.ylabel("N(z) per year")
    plt.xlim(0,5)
    
    for i in np.arange(Nparams):
        plt.plot(zvals,pzs[i,:],color="grey",linestyle="-")
    
    plt.plot(zvals,mean,color="black",linestyle="-",linewidth=2,label="Simulation mean")
    
    plt.legend()
    plt.tight_layout()
    plt.savefig(label+"_pz.png")
    plt.close()
    
 
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
    
    for i in np.arange(Nparams):
        plt.plot(dmvals,pdms[i,:],color="grey",linestyle="-")
    
    plt.plot(dmvals,mean,color="black",linestyle="-",linewidth=2,label="Simulation mean")
    
    plt.legend()
    plt.tight_layout()
    plt.savefig(label+"_pdm.png")
    plt.close()   

main()
