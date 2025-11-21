import numpy as np
import importlib.resources as resources
import os
import pandas as pd
def main():
    """
    Loads in unique configs, and generates beamfiles for them,
    weighted by the time on sky
    """
    # turn on add to add beamfactors column
    gen_weighted_beams("BeamHistograms/","FinalBeams/",add=False)
    # turn on this to generate files for the primary beam
    gen_weighted_beams("PrimaryBeams/","PrimaryBeams/")

def gen_weighted_beams(indir,opdir,add=False):  
    
    configs = pd.read_csv("Logs/configs.csv")# np.loadtxt("configs.dat",dtype="str")
    nconfigs = len(configs)
    
    pyfile = os.path.join(resources.files('zdm'), 'beam_generator','sim_craco_beam.py')
    
    bins = np.load("BeamHistograms/craco_histogram_bins.npy")
    nbins=bins.size-1
    bcentres = bins[0:-1] * (bins[1]/bins[0])**0.5
    # relative rate per solid angle: Euclidean expectation
    bfactors = bcentres**1.5
    
    fcut = 1100
    name1="CRACO_900_hist.npy"
    name2="CRACO_1300_hist.npy"
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
        
        Bfactor = np.sum(bfactors * hist)
        bfs.append(Bfactor)
        print(i," ",footprint," ",spitch," ",sfreq," beam factor is ",Bfactor)
        
        
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
    
    h1 /= t1
    h2 /= t2
    fbar1 /= t1
    fbar2 /= t2
    print("Total time t1 is ",t1," mean freq of ",fbar1)
    print("Total time t1 is ",t2," mean freq of ",fbar2)
    
    np.save(opdir+name1,h1)
    np.save(opdir+name2,h2)
    
    print("Total effective sensitivity of beam1 is ",np.sum(bfactors*h1))
    print("Total effective sensitivity of beam2 is ",np.sum(bfactors*h2))
    
    bfs = np.array(bfs)
    df = pd.read_csv("Logs/configs.csv")
    df["bfactors"]=bfs
    df.to_csv("Logs/configs.csv",index=False)
    
    # adds beam factors to the  configs file
    if add:
        add_beamfactors()
    
def add_beamfactors():
    """
    Adds relative beam factors as weighting to data
    """
    
    dfc = pd.read_csv("Logs/configs.csv")
    df = pd.read_csv("Logs/craco_13ms_survey_db_with_weights.csv")
    nobs = len(df)
    bfs = np.zeros([nobs])
    
    nconfig = len(dfc)
    
    for i in np.arange(nconfig):
        footprint = dfc["footprint"][i]
        pitch = dfc["pitch"][i]
        fbar = dfc["fbar"][i]
        bf = dfc["bfactors"][i]
        
        OK1 = np.where(df["footprint"] == footprint)[0]
        OK2 = np.where(df["pitch"] == pitch)[0]
        OK3 = np.where(df["fbar"] == fbar)[0]
        
        OK = np.intersect1d(OK1,OK2)
        OK = np.intersect1d(OK,OK3)
        
        bfs[OK3] = bf
    
    mean_bf = np.sum(df["t_eff"]*bfs)/np.sum(df["t_eff"])
    bfs /= mean_bf
    
    df["bfactors"]=bfs
    
    df.to_csv("Logs/craco_13ms_survey_db_with_weightsB.csv",index=False)
main()
