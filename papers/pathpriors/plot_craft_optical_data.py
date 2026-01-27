"""
This file generates plots of the CRAFT host galaxy candidates
"""


#standard Python imports
import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from importlib import resources
from scipy.integrate import quad

# imports from the "FRB" series
from zdm import optical as opt
from frb.frb import FRB
from astropath import chance

import matplotlib

defaultsize=14
ds=4
font = {'family' : 'Helvetica',
        'weight' : 'normal',
        'size'   : defaultsize}
matplotlib.rc('font', **font)



def main():
    """
    Main function
    Contains outer loop to iterate over parameters
    
    """
    
    # gets this
    frblist = opt.frblist
    
    maglist=[]
    anglist=[]
    
    
    for i,frb in enumerate(frblist):
        my_frb = FRB.by_name(frb)
            # reads in galaxy info
        ppath = os.path.join(resources.files('frb'), 'data', 'Galaxies', 'PATH')
        pfile = os.path.join(ppath, f'{my_frb.frb_name}_PATH.csv')
        ptbl = pd.read_csv(pfile)
        candidates = ptbl[['ang_size', 'mag', 'ra', 'dec', 'separation']]
        
        maglist = maglist + list(candidates['mag'].values)
        anglist = anglist + list(candidates['ang_size'].values)
    
    
    ngs = len(maglist)
    weights = np.full([ngs],1.)
    
    # gets most likely ics hosts
    z,mr,w = read_craft()
    weights = np.append(weights,-w)
    tempmaglist = np.append(maglist,mr)
    
    
    
    # plots histograms
    NM=21
    mags = np.linspace(10,30,NM)
    mids = (mags[1:] + mags[:-1])/2.
    #pmags = np.zeros([NM-1])
    #for i,mag in enumerate(mids):
    #    pmags[i] = chance.driver_sigma(mag)
    pmags = int_driver(mags)
    
    plt.figure()
    plt.hist(maglist,label="optical images",bins=mags)
    plt.hist(tempmaglist,label="'host' subtracted",weights=weights,bins=mags)
    plt.xlabel("Apparent magnitude, $m$")
    plt.ylabel("Number of galaxies")
    
    # arbitrary normalisation
    pmags = 2.*pmags*80/pmags[12]
    
    plt.plot(mids,pmags,label="Driver et al. 2016",linewidth=3,linestyle="--")
    plt.legend()
    plt.xlim(14,28)
    plt.xticks(np.linspace(14,28,8))
    plt.ylim(0,100)
    plt.tight_layout()
    plt.savefig("Figures/mag_hist.png")
    plt.close()
    
    # gets the ratio
    h,b = np.histogram(maglist,bins=mags)
    ratio = h/pmags
    
    plt.figure()
    plt.plot(mids,ratio)
    plt.xlabel("Apparent magnitude, $m$")
    plt.ylabel("ratio: CRAFT/Driver et al")
    plt.yscale("log")
    plt.tight_layout()
    plt.savefig("Figures/ratio.png")
    plt.close()
    
    plt.figure()
    plt.scatter(maglist,anglist)
    plt.xlabel("Apparent magnitude, $m$")
    plt.ylabel("Angular size, $\\theta$ [']")
    plt.tight_layout()
    plt.savefig("Figures/ang_mag.png")
    plt.close()


def int_driver(bins):
    """
    Integrates the driver et al formula over the magnitude bins
    
    Args:
        bins: bins in r-band magnitude
    """
    
    nbins = len(bins)
    integral = np.zeros([nbins-1])
    for i in np.arange(nbins-1):
        result = quad(chance.driver_sigma,bins[i],bins[i+1])
        integral[i] = result[0]
    return integral
    
def read_craft():
    """
    CRAFT ICS data
    """
    
    DF = pd.read_csv("../lsst/Data/craft_ics_hosts.csv")
    
    z = np.array(DF["z"])
    mr = np.array(DF["mr"])
    nfrb = len(mr)
    w = np.full([nfrb],1.) # artificial, but all are highy confidence
    return z,mr,w
    
main()
