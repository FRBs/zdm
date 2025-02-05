"""
Compares FAST and PARKES multibeam profiles

Parkes is scaled to FAST according to simple scaling
relations.

Note that the Parkes profile is originally taken
from a full beam simulation.
"""
import numpy as np
from matplotlib import pyplot as plt

def main():
    
    
    
    FASTb = np.load("FAST_log_bins.npy")
    FASTh = np.load("FAST_log_hist.npy")
    
    Parkesb = np.load("../data/BeamData/parkes_mb_log_bins.npy")
    Parkesh = np.load("../data/BeamData/parkes_mb_log_hist.npy")
    
    lPb = np.log10(Parkesb)
    lPb = lPb[:-1]
    lPb += (lPb[1]-lPb[0])/2.
    Parkesb = 10**lPb
    
    lFb = np.log10(FASTb)
    lFb = lFb[:-1]
    lFb += (lFb[1]-lFb[0])/2.
    FASTb = 10**lFb
    
    total = np.sum(FASTh)
    print("Total FAST is ",total)
    # scaling of Parkes hist
    Parkesh *= 19/13 * (64/300)**2.
    
    plt.figure()
    
    plt.plot(Parkesb,Parkesh,label="Scaled Parkes")
    plt.plot(FASTb,FASTh,label="Gaussian FAST sim")
    plt.xlabel("B")
    plt.ylabel("$\\Omega(B)$")
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    plt.tight_layout()
    plt.savefig("parkes_fast_comparison.png")
    plt.close()
    
main()
    
