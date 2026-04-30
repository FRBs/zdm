"""
Script to guide running of PATH on fake FRBs.

Based off 
https://astropath.readthedocs.io/en/latest/nb/Simulate_Run_PATH.html

What we really do here is to write a library of fake observation files.
The actual running of PAHT lives within zdm
"""

from frb.frb import FRB
import pandas as pd
import numpy as np
import os
from astropath.simulations import run_path as rp
from astropath.simulations import utils
from matplotlib import pyplot as plt
from zdm import optical as opt

from astropy.io import ascii
from astropy.table import Table

def main():
    
    
    ### read in frb data, and write these out as individual FRB files ###
    frbs = pd.read_csv("craco_900_mc_sample.csv")
    hosts = pd.read_csv("craco_assigned_galaxies.csv")
    NFRB=len(frbs)
    Nhosts = len(hosts)
    
    # the column "FRBid" from hosts tells you what index within frbs this comes from
    print("FRB info is ",frbs.columns)
    print("Host info is ",hosts.columns)
    
    path = "FRBFiles/"
    if not os.path.exists(path):
        os.mkdir(path)
    
    ###################### WRITING FRB FILES ########################
    # We combine frb and host data to create an "FRB" object
    for i in np.arange(Nhosts):
        host=hosts.loc[i]
        j = int(host["FRB_ID"])
        frb = frbs.loc[j]
        
        fname = "FRB"+str(int(j))+".json"
        if os.path.exists(path+fname):
            continue
        
        # we create a dict containing all necessary FRB data
        frbdict={}
        frbdict["ra"] = host["ra"]
        frbdict["dec"] = host["dec"]
        frbdict["DM"] = {}
        frbdict["DM"]["value"] = frb["DMeg"]
        frbdict["DM"]["unit"] = "pc / cm3"
        frbdict["nu_c"] = 900.
        frbdict["S"] = frb["s"]
        frbdict["FRB"] = "FRB"+str(j)
        frbdict["cosmo"] = "Planck18"
        
        eellipse = {}
        eellipse["a"] = host["a"]
        eellipse["b"] = host["b"]
        eellipse["cl"] = 68.
        eellipse["theta"] = host["PA"] # should not matter
        frbdict["eellipse"] = eellipse
        
        this_frb = FRB.from_dict(frbdict)
        this_frb.write_to_json(outfile=fname,path=path, overwrite=True)
    
    ###################### ASSIGNING CANDIDATES ########################
    
    # we load the catalogue *without* the deeper survey
    os.environ['FRB_APATH'] = "./"
    
    # this one has ang_size
    galaxies = pd.read_parquet("catalog_dudxmmlss_hecate_DECaL.parquet")
    
    # this one has half_light
    #other_g = pd.read_parquet("combined_HSC_DECaLs_HECATE_galaxies_hecatecut.parquet")
    
    # hard-coded parameters giving completeness of this catalogue from PATH paper
    survey_mean = 24
    survey_width = 0.55
    magnitudes = np.linspace(10,30,201)
    pU = opt.pUgm(magnitudes,survey_mean,survey_width)
    #plt.figure()
    #plt.plot(magnitudes,pU)
    #plt.show()
    
    # generates a random number for each true host, to determine if it should be removed from the
    # file, since these are forced to exist. Other galaxies in the catalogue are naturally
    # removed due to the intrinsic completeness
    
    deviates = np.random.rand(Nhosts)
    pUs = np.interp(hosts["mag"],magnitudes,pU)
    
    # plt.figure()
    # if random number is less than p(U), the host is invisible
    #REMOVE = np.where(deviates < pUs)[0]
    #KEEP = np.where(deviates >= pUs)[0]
    #plt.scatter(hosts["mag"][KEEP],pUs[KEEP],color="blue",s=0.1)
    #plt.scatter(hosts["mag"][REMOVE],pUs[REMOVE],color="red",s=0.1)
    #plt.show()
    #exit()
    
    frbs["ra"] = hosts["ra"]
    frbs["dec"] = hosts["dec"]
    # adds host ra and dec
    
    FRB_IDS = hosts["FRB_ID"]
    
    ###################### WRITING CANDIDATES ########################
    opdir="CandidateFiles/"
    if not os.path.exists(opdir):
        os.mkdir(opdir)
        
    ####### Writes out candidate files ######
    for i in np.arange(Nhosts):
        fname = opdir+'FRB'+str(hosts["FRB_ID"][i])+"_PATH.csv"
        #if os.path.exists(fname):
        #    print("Skipping FRB ",i)
        #    continue 
         
        frb = frbs.iloc[i]
        jhost = np.where(FRB_IDS == i)
        host = hosts.iloc[jhost]
        
        #print(host)
        
        host = hosts.iloc[i]
        candidates = extract_candidates(frb,galaxies, search_radius_arcmin=0.167)
        
        if len(candidates) > 0:
            # adds redshift info. Begins by setting all candidates to "unknown" (code = -1)
            candidates["z"] = -1.
            # now searches through for a matching galaxy to the known host.
            # We either remove it, or add its redshift
            
            if host["gal_ID"] in candidates["ID"]:
                j = np.where(host["gal_ID"] == candidates["ID"])[0][0]
                if deviates[i] < pUs[i]:
                    print("removing row ",j)
                    candidates.remove_row(j)
                else:
                    
                    candidates["z"][j] = frb["z"]
                    #print("Found host redshift ",frb["z"],candidates["z"][j])
        else:
            candidates["z"] = -1 # should creatre something empty anyway
        
        print("Writing ",fname)
        ascii.write(candidates, fname, overwrite=False, format='csv')

    
            
            
        
# this function is taken directly from 
# https://astropath.readthedocs.io/en/latest/nb/Simulate_Run_PATH.html
def extract_candidates(frb_row, galaxies, search_radius_arcmin=1.0):
    """
    Extract candidate host galaxies near an FRB position.

    Args:
        frb_row: Row from assignments DataFrame
        galaxies: Galaxy catalog DataFrame
        search_radius_arcmin: Search radius in arcmin

    Returns:
        astropy.table.Table: Candidate catalog for PATH
    """
    # Get FRB position
    frb_ra = frb_row['ra']
    frb_dec = frb_row['dec']

    # Convert search radius to degrees
    search_radius_deg = search_radius_arcmin / 60.

    # Simple box cut (fast approximation)
    cos_dec = np.cos(np.radians(frb_dec))
    ra_mask = np.abs(galaxies['ra'] - frb_ra) * cos_dec < search_radius_deg
    dec_mask = np.abs(galaxies['dec'] - frb_dec) < search_radius_deg

    nearby = galaxies[ra_mask & dec_mask].copy()
    
    if len(nearby) == 0:
        catalog = Table(names=('ra','dec','ang_size','mag','ID'))
    else:
        # Convert to astropy Table for PATH
        catalog = Table()
        catalog['ra'] = nearby['ra'].values
        catalog['dec'] = nearby['dec'].values
        catalog['ang_size'] = nearby['ang_size'].values
        catalog['mag'] = nearby['mag'].values
        catalog['ID'] = nearby['ID'].values

    return catalog

main()
