"""
This script takes in all simulated configs,
and adds them together with appropriate weights to
create an average beamshape.

It also adds the beam sensitivity values to the logfile
"""

import numpy as np
import importlib.resources as resources
import os
import pandas as pd
import glob

# value of assumed scaling with flux
#global alpha=1.5

def main():
    """
    Loads in unique configs, and generates beamfiles for them,
    weighted by the time on sky
    
    Runs to add beamfactor with alpha=1.5, and beamfactora1 with alpha=1.5
    """
    
    config_files = glob.glob("Logs/configs_*.csv") 
    key="a1bfactors"
    alpha=1.0
    finalopdir = os.path.join(resources.files('zdm'), 'data/BeamData/')
    for configfile in config_files:
        
        part1 = configfile.split("_")
        itsamp = part1[1].split(".")[0]
        logfile = "Logs/itsamp_"+itsamp+".csv"
        
        
        print("\n\n\n\n####### Analysing beam data for itsamp of ",itsamp,"#######\n")
        
        # turn on add to add beamfactors column
        # adds a column of beamfactors to the logfile, to weight by
        
        gen_weighted_beams("BeamHistograms/",finalopdir,configfile,logfile,add=key,prefix=itsamp+"_",alpha=alpha)
        
        # turn on this to generate files for the primary beam
        gen_weighted_beams("PrimaryBeams/","PrimaryBeams/",configfile,logfile,add=False,prefix=itsamp+"_",alpha=alpha)
    
def gen_weighted_beams(indir,opdir,configfile,logfile,add=False,prefix="",alpha=1.5):  
    """
    Generates summed beams over all configurations
    
    Args:
        indir [string]: input directory for the beam footprint data
        opdir [string]: output directory
        configfile [string]: name of file containing configuration logs
        logfile [string]: name of file containing logs of all observations
        add [bool]: if true, add weights to the log file
        prefix [string]: prefix to apply to output
    """
    configs = pd.read_csv(configfile)# np.loadtxt("configs.dat",dtype="str")
    nconfigs = len(configs)
    
    pyfile = os.path.join(resources.files('zdm'), 'beam_generator','sim_craco_beam.py')
    
    bins = np.load("BeamHistograms/craco_histogram_bins.npy")
    nbins=bins.size-1
    bcentres = bins[0:-1] * (bins[1]/bins[0])**0.5
    # relative rate per solid angle: Euclidean expectation
    bfactors = bcentres**alpha
    
    fcut = 1100
    name1=prefix+"CRACO_900_hist.npy"
    name2=prefix+"CRACO_1300_hist.npy"
    bname1=prefix+"CRACO_900_bins.npy"
    bname2=prefix+"CRACO_1300_bins.npy"
    t1=0.
    t2=0.
    h1 = np.zeros([nbins])
    h2 = np.zeros([nbins])
    fbar1=0.
    fbar2=0.
    
    bfs=[]
    
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
        
        if fp=="closepack36":
            footprint="closepack"
        elif fp == "square_6x6":
            footprint="square"
        else:
            print("Unrecognised footprint ",fp)
            exit()
        
        gsize = 10.
        gpix = 2560
        basename = f"{indir}hist_craco_{footprint}_p{fpitch:.2f}_f{freq:.1f}MHz_f{gsize:.1f}d_npix{gpix}_.npy"
        
        if os.path.exists(basename):
            hist = np.load(basename)
        else:
            print("Cannot find ",basename)
            exit()
        
        # weights by sensitivity assuming Euclidean scaling
        Bfactor = np.sum(bfactors * hist)
        bfs.append(Bfactor)
        print(i," ",footprint," ",spitch," ",sfreq," beam factor is ",Bfactor," from ",basename)
        
        
        if freq < fcut:
            # low band
            t1 += teff
            h1 += hist*teff
            fbar1 += freq*teff
        else:
            # L-band
            t2 += teff
            h2 += hist*teff
            fbar2 += freq*teff
    
    if t1 > 0.:
        h1 /= t1
        fbar1 /= t1
        print("Total time t1 is ",t1," mean freq of ",fbar1)
        print("Total effective sensitivity of 900 MHz beam is ",np.sum(bfactors*h1))
        np.save(opdir+name1,h1)
        # creates a copy
        os.system("cp "+opdir+"CRACO_900_bins.npy" + " "+opdir+bname1)
    
    if t2 > 0.:
        h2 /= t2
        fbar2 /= t2
        print("Total time t2 is ",t2," mean freq of ",fbar2)
        print("Total effective sensitivity of 1300 MHz beam is ",np.sum(bfactors*h2))
        np.save(opdir+name2,h2)
        os.system("cp "+opdir+"CRACO_900_bins.npy" + " "+opdir+bname2)
    
    bfs = np.array(bfs)
    bfs *= (180./np.pi)**2 # converts to units of effective deg2
    df = pd.read_csv(configfile)
    
    
    # adds beam factors to the  configs file
    if add is not False:
        # if adding, add beamfactors to logfile
        df[add]=bfs
        df.to_csv(configfile,index=False)
        add_beamfactors(configfile,logfile,key=add)
    
def add_beamfactors(configfile,logfile,key="bfactors"):
    """
    Adds relative beam factors as weighting to data
    
    Args:
        configfile [string]: name of file containing configuration logs
        logfile [string]: name of file containing logs of all observations
    """
    
    dfc = pd.read_csv(configfile)
    df = pd.read_csv(logfile)
    nobs = len(df)
    bfs = np.zeros([nobs])
    
    nconfig = len(dfc)
    
    for i in np.arange(nconfig):
        footprint = dfc["footprint"][i]
        pitch = dfc["pitch"][i]
        fbar = dfc["fbar"][i]
        bf = dfc[key][i]
        
        OK1 = np.where(df["footprint"] == footprint)[0]
        OK2 = np.where(df["pitch"] == pitch)[0]
        OK3 = np.where(df["fbar"] == fbar)[0]
        
        OK = np.intersect1d(OK1,OK2)
        OK = np.intersect1d(OK,OK3)
        
        bfs[OK3] = bf
    # Do we add relative or absolute beamfactor? Should be relative to something!
    # maybe choose beamfactor for closepack 6x6 at 1.3 GHz?
    # we choose not to normalise by the mean beamfactor here. Can do all this later
    #mean_bf = np.sum(df["t_eff"]*bfs)/np.sum(df["t_eff"])
    #bfs /= mean_bf
    
    df[key]=bfs
    
    df.to_csv(logfile,index=False)
main()
