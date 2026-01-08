"""
This script iterates over available beam configurations
and simulates their beam patterns
"""

import numpy as np
import importlib.resources as resources
import os
import pandas as pd

def main():
    """
    Loads in unique configs, and generates beamfiles for them
    """
    
    configfile="Logs/configs.csv"
    sim_all_configs("BeamHistograms/",configfile,False)
    sim_all_configs("PrimaryBeams/",configfile,True)
    
    configfile="Logs/3ms_configs.csv"
    sim_all_configs("BeamHistograms/",configfile,False)
    sim_all_configs("PrimaryBeams/",configfile,True)

def sim_all_configs(Bdir,configfile,primary=False):
    """
    Args:
        Bdir [string]: directory for beam data
        configfile [string]: file for telescope configuration data
        primary [bool]: if True, simulate primary beam only
    """
    configs = pd.read_csv(configfile)# np.loadtxt("configs.dat",dtype="str")
    nconfigs = len(configs)
    
    pyfile = os.path.join(resources.files('zdm'), 'beam_generator','sim_craco_beam.py')
    
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
        basename = f"{Bdir}/hist_craco_{footprint}_p{fpitch:.2f}_f{freq:.1f}MHz_f{gsize:.1f}d_npix{gpix}_.npy"
        basename2 = f"./hist_craco_{footprint}_p{fpitch:.2f}_f{freq:.1f}MHz_f{gsize:.1f}d_npix{gpix}_.npy"
        basename3 = f"./craco_{footprint}_p{fpitch:.2f}_f{freq:.1f}MHz_f{gsize:.1f}d_npix{gpix}.npy"
        if os.path.exists(basename):
            print("Found ",basename)
        else:
            command = "python "+pyfile +" -fp " + footprint + " -p " + spitch + " -f " + sfreq[0:7] + " --primary="+str(primary)
            
            os.system(command)
            
            command = "mv "+basename2+" "+Bdir
            os.system(command)
            
            command = "mv "+basename3+" "+Bdir
            os.system(command)
        
    


main()
