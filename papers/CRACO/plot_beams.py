
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
    """
    Loads in beam data and plots this
    """
    # loads in data
    indir = os.path.join(resources.files('zdm'), 'data','BeamData')
    
    beams = ["CRACO_900","CRACO_1300"]#,"ASKAP_892","ASKAP_1300"]
    labels=["CRACO 900","CRACO 1300"]#,"ICS 892","ICS 1300"]
    linestyles=["-","--",":","-."]
        
    Senses = []
    bfiles = []
    
    plt.figure()
    for i,beam in enumerate(beams):
        bfile=beam+"_bins.npy"
        hfile=beam+"_hist.npy"
        
        bfile = os.path.join(indir,bfile)
        hfile = os.path.join(indir,hfile)
        h=np.load(hfile)
        b=np.load(bfile)
        
        
        # divides hist file by log-scaling in b
        bwidth = b[1]/b[0]
        lbwidth = np.log10(bwidth)
        b=b[:-1] * bwidth**0.5
        
        # need to sum before normalisation
        Sens = np.sum(h*b**1.5)
        Senses.append(Sens)
        bfiles.append(bfile)
        
        h /= lbwidth
        # get bin centres
        
        if i == 0:
            plt.plot(b,h,label=labels[i],linestyle=linestyles[i])
        else:
            plt.plot(b,h,label=labels[i],linestyle=linestyles[i],color=plt.gca().lines[-1].get_color())
        
        
    plt.xlabel("$B$")
    plt.ylabel("$\\Omega(B)$")
    plt.legend()
    plt.tight_layout()
    
    # plots data
    plt.savefig("Plots/craco_beams.png")
    
    
    ###### Adds primary beams #####
    
    # adds plots of primary beam response
    # just knows that the b values are identical to previous
    bfile="PrimaryBeams/CRACO_900_hist.npy"
    h = np.load(bfile)
    
    
    # need to sum before normalisation
    Sens = np.sum(h*b**1.5)
    Senses.append(Sens)
    bfiles.append(bfile)
    
    
    h /= lbwidth
    plt.plot(b,h,label="Primary 900",linestyle="-")
    
    bfile="PrimaryBeams/CRACO_1300_hist.npy"
    h = np.load(bfile)
    
    # need to sum before normalisation
    Sens = np.sum(h*b**1.5)
    Senses.append(Sens)
    bfiles.append(bfile)
    
    h /= lbwidth
    plt.plot(b,h,label="Primary 1300",linestyle="--",color=plt.gca().lines[-1].get_color())
    plt.legend()
    plt.tight_layout()
    plt.savefig("Plots/primary_askap_beams.png")
    
    ###### Adds primary beams #####
    
    # adds plots of primary beam response
    # just knows that the b values are identical to previous
    bfile=indir+"/ASKAP_892_hist.npy"
    h = np.load(bfile)
    b = np.load(indir+"/ASKAP_892_bins.npy")
    
    
    bwidth = b[1]/b[0]
    b = b[:-1]*bwidth**0.5
    
    # need to sum before normalisation
    Sens = np.sum(h*b**1.5)
    Senses.append(Sens)
    bfiles.append(bfile)
    
    lbwidth = np.log10(bwidth)
    h /= lbwidth
    
    plt.plot(b,h,label="ICS 900",linestyle="-")
    
    bfile=indir+"/ASKAP_1300_hist.npy"
    h = np.load(bfile)
    b = np.load(indir+"/ASKAP_1300_bins.npy")
    bwidth = b[1]/b[0]
    b = b[:-1]*bwidth**0.5
    
    # need to sum before normalisation
    Sens = np.sum(h*b**1.5)
    Senses.append(Sens)
    bfiles.append(bfile)
    
    lbwidth = np.log10(bwidth)
    h /= lbwidth
    
    plt.plot(b,h,label="ICS 1300",linestyle="--",color=plt.gca().lines[-1].get_color())
    plt.ylim(0,0.07)
    plt.xlim(0,1)
    plt.legend()
    plt.tight_layout()
    plt.savefig("Plots/comparison_askap_beams.png")
    
    
    plt.close()
    
    # prints relative sensitivitiesa compared to 
    for i in np.arange(3):
        
        print("Sensitivity of beam file ",i," compared to ICS: ",bfiles[2*i],Senses[2*i]/Senses[4])
        print("Sensitivity of beam file ",i," compared to ICS: ",bfiles[2*i+1],Senses[2*i+1]/Senses[5])
    
    
    ##### plots all components #####
    configs = pd.read_csv("Logs/configs.csv")# np.loadtxt("configs.dat",dtype="str")
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
    plt.savefig("Plots/closepack_lowf_component_beams.png")
    plt.close()
    
    
main()
