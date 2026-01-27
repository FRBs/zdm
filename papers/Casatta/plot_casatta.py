
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
    plots casatta simulation results
    """
    df = read_casatta_params()
    nsims,ncols = df.shape
    
    dailys = np.load("dailys.npy")
    pzs = np.load("pzs.npy")
    pdms = np.load("pdms.npy")
    zvals = np.load("zvals.npy")
    dmvals = np.load("dmvals.npy")
    threshs = np.load("threshs.npy")
    
    # daily FRB rate for each config
    plt.figure()
    
    plt.yscale('log')
    plt.scatter(np.arange(nsims),dailys)
    plt.xticks(np.arange(nsims),df["Array_name"],rotation=90,fontsize=6)
    
    plt.ylabel("Daily FRB rate")
    plt.tight_layout()
    plt.savefig("frb_rate_per_configuration.png")
    plt.close()
    
    plt.figure()
    plt.yscale("log")
    
    
    # multiplies by z-bin width
    dz = zvals[1]-zvals[0]
    plt.xlabel("z")
    plt.ylabel("p(z) [FRBs / day / z]")
    plt.ylim(1e-3,1e5)
    for isim in np.arange(nsims):
        plt.plot(zvals,pzs[isim,:]/dz,label=df["Array_name"][isim])
    plt.legend(fontsize=4)
    plt.tight_layout()
    plt.savefig("all_pz.png")
    plt.close()
    
    
    plt.figure()
    plt.yscale("log")
    plt.ylim(1e-6,1e2)
    # multiplies by DM width
    ddm = dmvals[1]-dmvals[0]
    
    plt.xlabel("DM [pc cm$^{-3}$]")
    plt.ylabel("p(DM) [FRBs /day /pc cm$^{-3}$]")
    for isim in np.arange(nsims):
        plt.plot(dmvals,pdms[isim,:]/ddm,label=df["Array_name"][isim])
    plt.legend(fontsize=4)
    plt.tight_layout()
    plt.savefig("all_pdm.png")
    plt.close()
    
    
    # compares estimates from nominal figure of merit
    FOM = threshs**-1.5 * df["FWHM_deg"]**2
    
    plt.figure()
    plt.xlabel("FOM [FWHM$^2$ (Jy ms)$^{-1.5}$]")
    plt.ylabel("Daily rate")
    plt.xscale("log")
    plt.yscale("log")
    plt.scatter(dailys,FOM)
    \
    plt.plot([1e-5,1e4],[0.05,5e7],color="black",label="1-1 line",linestyle="--")
    plt.legend()
    plt.tight_layout()
    plt.savefig("FOM.png")
    plt.close()
    
    
def read_casatta_params(infile="CASATTA MFAA SKA2 FRB estimates.csv"):
    """
    Reads in casatta parameters
    """
    
    import pandas as pd
    df = pd.read_csv(infile)
    
    return df

    
main()
    
    
