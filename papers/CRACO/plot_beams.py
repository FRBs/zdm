"""
Script to plot inverse beamshapes \Omega(B) vs B

"""
import numpy as np
from matplotlib import pyplot as plt
import importlib.resources as resources
import os
import matplotlib
import pandas as pd

defaultsize=14

font = {'family' : 'Helvetica',
        'weight' : 'normal',
        'size'   : defaultsize}
matplotlib.rc('font', **font)

def main():
    """
    Loads in beam data and plots it
    """
    
    prefix=""
    #prefix="3ms_"
    
    # loads in data
    bdir1 = os.path.join(resources.files('zdm'), 'data','BeamData')
    bdir2 = os.path.join(resources.files('zdm'), '../papers/CRACO','FinalBeams/')
    bdir3 = os.path.join(resources.files('zdm'), '../papers/CRACO','PrimaryBeams/')
    hdir =  os.path.join(resources.files('zdm'), '../papers/CRACO','BeamHistograms/')
    
    #beams = ["CRACO_900","CRACO_1300","3ms_CRACO_900","3ms_CRACO_1300"]#,"ASKAP_892","ASKAP_1300"]
    
    itsamps = [1,2,4,8,16,64]
    tlabels=["1.7ms","3.4ms","6.9ms","13.8ms","27.6ms","110ms"]
    
    beams=[]
    labels=[]
    
    for i,itsamp in enumerate(itsamps):
        beam = str(itsamp)+"_CRACO_900_hist"
        beams.append(beam)
        
        label = tlabels[i]+" 900MHz"
        labels.append(label)
        
        beam = str(itsamp)+"_CRACO_1300_hist"
        beams.append(beam)
        
        label = tlabels[i]+" 1300MHz"
        labels.append(label)
    
    
    #labels=["CRACO 900 13.8ms","CRACO 1300 13.8ms","CRACO 900 3.4ms","CRACO 1300 3.4ms"]#,"ICS 892","ICS 1300"]
    linestyles=["-","--",":","-."]
        
    Senses = []
    slabels = []
    #bfiles = []
    #hfiles=[]
    
    # for inclusive plot
    plt.figure()
    ax1 = plt.gca()
    
    # for specific plot
    plt.figure()
    ax2 = plt.gca()
    iplot=[6,7]
    niplot=0
    
    for i,beam in enumerate(beams):
        bfile=hdir+"craco_histogram_bins.npy"
        #hfile=bdir+beam+"_hist.npy"
        hfile=bdir2+beam+".npy"
        
        # nor all combinations of times and frequencies exist
        if not os.path.exists(hfile):
            print("File ",hfile," DNE, continuing...")
            continue
        
        #bfile = os.path.join(indir,bfile)
        #hfile = os.path.join(indir,hfile)
        h=np.load(hfile)
        b=np.load(bfile)
        
        
        # divides hist file by log-scaling in b. Gets bin centres for plotting
        bwidth = b[1]/b[0]
        lbwidth = np.log10(bwidth)
        b=b[:-1] * bwidth**0.5
        
        
        # need to sum before normalisation
        Sens = np.sum(h*b**1.5)
        Senses.append(Sens)
        slabels.append("imaged+"+labels[i])
        #bfiles.append(bfile)
        #hfiles.append(hfile)
        
        h /= lbwidth
        # get bin centres
        
        plt.sca(ax1)
        if i == 0:
            plt.plot(b,h,label=labels[i],linestyle=linestyles[i%4])
        else:
            plt.plot(b,h,label=labels[i],linestyle=linestyles[i%4])#,color=plt.gca().lines[-1].get_color())
        
        plt.sca(ax2)
        if i in iplot:
            plt.plot(b,h,label="Imaged "+labels[i],linestyle=linestyles[niplot%4])
            niplot += 1
    
    plt.sca(ax1)  
    plt.xlabel("$B$")
    plt.ylabel("$\\Omega(B)$")
    plt.legend(fontsize=10,loc="upper left")
    plt.ylim(0,0.035)
    plt.tight_layout()
    
    # plots data
    plt.savefig("Plots/"+prefix+"craco_beams.png")
    
    
    ##### does restricted paper version of plot ######
    plt.sca(ax2)  
    plt.xlabel("$B$")
    plt.ylabel("$\\Omega(B)$")
    plt.legend(fontsize=10,loc="upper left")
    plt.ylim(0,0.035)
    plt.tight_layout()
    
    # plots data
    plt.savefig("Plots/"+prefix+"paper_craco_beams.png")
    # do not close this, since we are doing  more plotting later
    
    
    ###### Adds primary beams #####
    
    # new figure for primaries only
    plt.figure()
    ax3 = plt.gca()
    #niplot=0
    
    for i,beam in enumerate(beams):
        bfile=hdir+"craco_histogram_bins.npy"
        #hfile=bdir+beam+"_hist.npy"
        hfile=bdir3+beam+".npy"
        
        # nor all combinations of times and frequencies exist
        if not os.path.exists(hfile):
            print("File ",hfile," DNE, continuing...")
            continue
        
        h = np.load(hfile)
        
        # need to sum before normalisation
        Sens = np.sum(h*b**1.5)
        Senses.append(Sens)
        slabels.append("primary+"+labels[i])
        #bfiles.append(bfile)
    
        h /= lbwidth
        plt.sca(ax1)
        plt.plot(b,h,label="primary "+labels[i],linestyle=linestyles[i%4],color=plt.gca().lines[i].get_color())
        
        plt.sca(ax3)
        plt.plot(b,h,label="primary "+labels[i],linestyle=linestyles[i%4])
    
        plt.sca(ax2)
        if i in iplot:
            plt.plot(b,h,linestyle=linestyles[niplot%4],linewidth=2,label="Primary "+labels[i])
            niplot += 1
    
    plt.sca(ax1)
    plt.legend(fontsize=6,loc="upper left")
    plt.ylim(0,0.07)
    plt.tight_layout()
    plt.savefig("Plots/"+prefix+"primary_askap_beams.png")
    
    plt.sca(ax3) 
    plt.xlabel("$B$")
    plt.ylabel("$\\Omega(B)$")
    plt.ylim(0,0.07)
    plt.legend(fontsize=8,loc="upper left")
    plt.tight_layout()
    plt.savefig("Plots/"+prefix+"only_primary_askap_beams.png")
    
    plt.sca(ax2)
    plt.ylim(0,0.07)
    #plt.legend(fontsize=10,loc="upper left")
    plt.tight_layout()
    plt.savefig("Plots/"+prefix+"paper_primary_askap_beams.png")
    
    
    # do not close this - anpother version, showing primary beams for ICS configs
    
    
    ###### Adds primary beams from ICS mode #####
    
    stdfiles = ["ASKAP_892","ASKAP_1300"]
    alabels = ["ICS 900 MHz","ICS 1300 MHz"]
    plt.sca(ax2)
    
    for i,std in enumerate(stdfiles):
        bfile = bdir1+"/"+std+"_hist.npy"
        h = np.load(bfile)
        
        bfile = bdir1+"/"+std+"_bins.npy"
        b = np.load(bfile)
    
        bwidth = b[1]/b[0]
        b = b[:-1]*bwidth**0.5
        
        # need to sum before normalisation
        Sens = np.sum(h*b**1.5)
        Senses.append(Sens)
        slabels.append(alabels[i])
        #bfiles.append(bfile)
        
        lbwidth = np.log10(bwidth)
        h /= lbwidth
        
        
        plt.plot(b,h,label=alabels[i],linestyle=linestyles[i],color="black")
    
    #plt.ylim(0,0.07)
    #plt.xlim(0,1)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig("Plots/"+prefix+"paper_comparison_askap_beams.png")
    
    
    ### closing all figures ###
    plt.sca(ax1)
    plt.close()
    
    plt.sca(ax2)
    plt.close()
    
    plt.sca(ax3)
    plt.close()
    
    #### We now print out all relevant sensitivities #####
    # prints relative sensitivitiesa compared to 
    #labels=["CRACO 900", "CRACO 1300", "Primary 900", "Primary 1300", "ICS 900", "ICS 1300"]
    #labels = labels + ["Primary 900", "Primary 1300", "ICS 900", "ICS 1300"]
    mult = (180./np.pi)**2 # effective degrees square, rather than sr
    nlabels = len(slabels)
    for i,label in enumerate(slabels):
        if "900" in label:
            ai=-2
        else:
            ai=-1
        print("Sensitivity of ",label," is ",Senses[i]*mult, " cf ICS: ",Senses[i]/Senses[ai])
        #print("Sensitivity of ",labels[2*i+1]," is ",Senses[2*i+1]*mult, " cf ICS: ",Senses[2*i+1]/Senses[-1])
    
    exit()
    
    ########## LEGACY CODE!!! ########
    """
    The below is deprecated. We are no longer bothering to plot all components.
    This is left here in case somebody wants to resurrect this
    """
    
    ##### plots all components #####
    configs = pd.read_csv("Logs/"+prefix+"configs.csv")# np.loadtxt("configs.dat",dtype="str")
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
    plt.savefig("Plots/"+prefix+"closepack_lowf_component_beams.png")
    plt.close()
    
    
main()
