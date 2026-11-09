"""
Function to test delay-time distributions
"""

import numpy as np
from matplotlib import pyplot as plt
from zdm import cosmology as cos
import time

def run_test():
    
    zs = np.linspace(0.,5.,40)
    alpha=1. # dimensionless slope
    pl_tmin=0.001 # in Gyr
    pl_tmax = 1.5
    
    plt.figure()
    
    # tests DTD evolution
    for i,pl_tmax in enumerate(np.linspace(0.2,1.0,5)):
        print("calling with tmin, tmax = ",pl_tmin,pl_tmax)
        phi = cos.powerlaw_DTD_evolution(zs,alpha,pl_tmin,pl_tmax)
        if i==0:
            themax = np.max(phi)
        plt.plot(zs,phi,label="tmax is "+str(pl_tmax)[0:3])
    sfrd = 0.015 * (((1+zs)**2.7)/(1+((1+zs)/2.9)**5.6))
    
    
    phi4 = cos.sfr_evolution(zs,1.)
    plt.plot(zs,phi4,label="SFR")
    
    phi2 = cos.GC_evolution(zs)
    plt.plot(zs,phi2,label="GC_GTD")
    
    phi3 = cos.GC_and_CCSNe_evolution(zs,0.5)
    plt.plot(zs,phi3,label="50% mixture")
    
    phi5 = cos.lognormal_DTD_evolution(zs,0.5,0.5)
    plt.plot(zs,phi5,label="lognormal DTD 0.5 0.5")
    
    #plt.plot(zs,sfrd/np.max(sfrd)*themax,label="SFRD")
    #plt.plot(zs,phi2,label="orig slow")
    #plt.plot(zs,(phi-phi2)*100,label="diff x 100")
    plt.xlabel("z")
    plt.ylabel("$\\Phi(z)$")
    plt.legend()
    plt.tight_layout()
    plt.savefig("source_evolution.pdf")
    plt.close()
