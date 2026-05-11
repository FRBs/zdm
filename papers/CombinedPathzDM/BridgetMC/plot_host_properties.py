"""
This file generates plots based on the generated FRB host properties
"""

import os
import pandas as pd
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
    
    """
    opdir="Hosts/"
    if not os.path.exists(opdir):
        os.mkdir(opdir)
        
    frbs = pd.read_csv("craco_900_mc_sample.csv")
    hosts = pd.read_csv("craco_assigned_galaxies.csv")
    
    plot_host_properties(frbs,hosts,opdir)
    
    #get_true_hosts(frbs,hosts,opdir)
    get_true_hosts(frbs,hosts,opdir+"arcmin_",indir="WideCandidateFiles/",NMAX=7000)
    
def get_true_hosts(frbs,hosts,opdir,indir="CandidateFiles/",NMAX=10000):
    """
    Runs through candidate files, getting which FRBs have a
    host, and which do not
    """
    NFRB = len(frbs)
    if NMAX > NFRB:
        NMAX = NFRB
    Nhosts = len(hosts)
    zfrb = np.zeros([NMAX])
    mlist = []
    notmlist = []
    hmlist = []
    ####### Writes out candidate files ######
    for i in np.arange(NMAX):
        fname = indir+'FRB'+str(i)+"_PATH.csv"
        
        if not os.path.exists(fname):
            zfrb[i] = -1
            continue
        
        cands = pd.read_csv(fname)
        # searches for FRB with known z
        zmatch = np.where(cands["z"] > -1)[0]
        if len(zmatch) == 0:
            zfrb[i] = -1
        else:
            zmatch = zmatch[0]
            zfrb[i] = cands["z"][zmatch]
        
        
        for j,mag in enumerate(cands["mag"]):
            mlist.append(mag)
            if cands["z"][j] == -1:
                notmlist.append(mag)
            else:
                hmlist.append(mag)
    # we can now looks at the properties of FRBs with
    # and without the true host
    mlist = np.array(mlist)[:NMAX]
    notmlist = np.array(notmlist)[:NMAX]
    hmlist = np.array(hmlist)[:NMAX]
    
    
    MISSING = np.where(zfrb == -1)[0]
    NMISSING = len(MISSING)
    print("Missing hosts for ",NMISSING," frbs")
    
    
    # histogram of redshifts for FRBs with known and unknown hosts
    
    bins = np.linspace(0,2,21)
    plt.figure()
    plt.hist(frbs["z"][:NMAX],label="True redshift",bins=bins,histtype='step')
    plt.hist(zfrb,label="Observed hosts",bins=bins,histtype='step',linestyle="--")
    plt.xlabel("FRB redshift, z")
    plt.ylabel("Counts")
    plt.legend()
    plt.yscale("log")
    plt.tight_layout()
    plt.savefig(opdir+"zhist.png")
    plt.close()
    
    
    # histogram of magnitudes as above
    bins = np.linspace(10,30,21)
    plt.figure()
    plt.hist(frbs["m_r"][:NMAX],label="All hosts",histtype='step',ls="-",bins=bins)
    plt.hist(hmlist,label="Observed hosts",histtype='step',ls="--",bins=bins)
    plt.hist(notmlist,label="Field galaxies",histtype='step',ls=":",bins=bins)
    plt.legend(loc="upper left")
    plt.yscale("log")
    plt.tight_layout()
    plt.savefig(opdir+"maghist.png")
    plt.close()
    
    print("Minimum and maximum magnitudes are ",np.min(hmlist),np.max(hmlist))
    
    
        
def plot_host_properties(frbs,hosts,opdir):
    """
    Makes plots comparing host and assigned host magnitudes
    """
    
    print(frbs.keys())
    
    print(hosts.keys())
    m1 = frbs["m_r"][hosts["FRB_ID"]]    
    
    print("Number of assigned hosts is ",len(m1))
    
    plt.figure()
    plt.scatter(m1,hosts["mag"])
    plt.xlabel("Simulated host magnitude")
    plt.ylabel("Assigned catalogue host magnitude")
    plt.tight_layout()
    plt.savefig(opdir+"host_assigned_scatter.png")
    plt.close()
    
    

main()
