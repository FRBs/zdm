
import numpy as np
from matplotlib import pyplot as plt
import importlib.resources as resources
import os
import matplotlib
import pandas as pd

defaultsize=14
ds=4
font = {'family' : 'Helvetica',
        'weight' : 'normal',
        'size'   : defaultsize}
matplotlib.rc('font', **font)

def main():
    
    # loads in data
    #h1=np.load("FinalBeams/CRACO_900_log_hist.npy")
    #b1=np.load("FinalBeams/CRACO_900_log_bins.npy")
    #h2=np.load("FinalBeams/CRACO_1300_log_hist.npy")
    #b2=np.load("FinalBeams/CRACO_1300_log_bins.npy")
    
    indir = os.path.join(resources.files('zdm'), 'data','BeamData')
    
    beams = ["CRACO_900","CRACO_1300"]#,"ASKAP_892","ASKAP_1300"]
    labels=["CRACO 900","CRACO 1300"]#,"ICS 892","ICS 1300"]
    linestyles=["-","--",":","-."]
        
    plt.figure()
    for i,beam in enumerate(beams):
        bfile=beam+"_bins.npy"
        hfile=beam+"_hist.npy"
        
        bfile = os.path.join(indir,bfile)
        hfile = os.path.join(indir,hfile)
        h=np.load(hfile)
        b=np.load(bfile)
        
        print("Total solid for ",labels[i]," is ",np.sum(h))
        
        # divides hist file by log-scaling in b
        bwidth = np.log10(b[1]/b[0])
        h /= bwidth
        # get bin centres
        b=b[:-1] * bwidth**0.5
        
        
        plt.plot(b,h,label=labels[i])
        
        
        
    plt.xlabel("$B$")
    plt.ylabel("$\\Omega(B)$")
    plt.legend()
    plt.tight_layout()
    
    # plots data
    plt.savefig("craco_beams.png")
    
    ###### Adds primary beams #####
    
    # adds plots of primary beam response
    # just knows that the b values are identical to previous
    h = np.load("PrimaryBeams/CRACO_900_hist.npy")
    h /= bwidth
    plt.plot(b,h,label="Primary 900")
    
    h = np.load("PrimaryBeams/CRACO_1300_hist.npy")
    h /= bwidth
    plt.plot(b,h,label="Primary 1300")
    plt.legend()
    plt.tight_layout()
    plt.savefig("primary_askap_beams.png")
    plt.close()
    
    
    ##### plots all components #####
    configs = pd.read_csv("configs.csv")# np.loadtxt("configs.dat",dtype="str")
    nconfigs = len(configs)
    
    b = np.load("BeamHistograms/craco_histogram_bins.npy")
    bwidth = np.log10(b[1]/b[0])
    # get bin centres
    b=b[:-1] * bwidth**0.5
    
    plt.figure()
    for i in np.arange(nconfigs):
        
        fp = configs["footprint"][i]
        pitch = configs["pitch"][i]
        fpitch = float(pitch)
        spitch = str(pitch)
        fbar = configs["fbar"][i]
        tobs = configs["Ttot"][i]
        teff = configs["Teff"][i]
        freq = float(fbar)
        sfreq = str(fbar)
        
        if freq > 1050:
            continue
        if fp=="closepack36":
            footprint="closepack"
        elif fp == "square_6x6":
            footprint="square"
            continue
        else:
            print("Unrecognised footprint ",fp)
            exit()
        
        gsize = 10.
        gpix = 2560
        basename = f"BeamHistograms/hist_craco_{footprint}_p{fpitch:.2f}_f{freq:.1f}MHz_f{gsize:.1f}d_npix{gpix}_.npy"
        
        if os.path.exists(basename):
            h = np.load(basename)
        
        else:
            print("Cannot find ",basename)
            exit()
        
        plt.plot(b,h/bwidth,label=footprint+" "+sfreq[0:4]+"MHz " + spitch)
    
    
    plt.xlabel("$B$")
    plt.ylabel("$\\Omega(B)$")
    plt.legend(fontsize=8,loc="upper left")
    plt.tight_layout()
    
    # plots data
    plt.savefig("closepack_lowf_component_beams.png")
    plt.close()
    
    
main()
