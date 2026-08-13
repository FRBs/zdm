
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
    
    # this factor did NOT multiply pzs and pdms because I'm an idiot and forgot
    # daily rates are correct
    RENORM = 0.177
    
    df = read_casatta_params()
    nsims,ncols = df.shape
    
    
    # selects which casatta plot to use
    OK1=np.where(df["FrequencyMHz"]==600)
    OK2=np.where(df["FWHM_deg"]==120)
    OK=np.intersect1d(OK1,OK2)
    
    
    dailys = np.load("dailys.npy")[OK]
    pzs = np.load("pzs.npy")[OK,:]
    pdms = np.load("pdms.npy")[OK,:]
    pzs *= RENORM
    pdms *= RENORM
    zvals = np.load("zvals.npy")
    dmvals = np.load("dmvals.npy")
    threshs = np.load("threshs.npy")[OK]
    
    
    # compares estimates from nominal figure of merit
    FOM = threshs**-1.5 * df["FWHM_deg"][OK]**2
    
    plt.figure()
    plt.xlabel("FOM [FWHM$^2$ (Jy ms)$^{-1.5}$]")
    plt.ylabel("Daily rate")
    plt.xscale("log")
    plt.yscale("log")
    plt.scatter(FOM,dailys,label="CASATTA 600 MHz, FWHM=120 deg")
    plt.ylim(1e-2,1e8)
    plt.xlim(1,1e8)
    
    plt.plot([1e-2,1e8],[3e-4,3e6],color="black",label="1-1 line",linestyle="--")
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig("FOM_ATNF.png")
    plt.close()
    
    
def read_casatta_params(infile="CASATTA MFAA SKA2 FRB estimates.csv"):
    """
    Reads in casatta parameters
    """
    
    import pandas as pd
    df = pd.read_csv(infile)
    
    return df

    
main()
    
    
