"""
Generates mean DM response curves for each of the CRACO observations
These are single curves giving the average max DM observed
"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import importlib.resources as resources
import os

def main():
    """
    Reads in logfile
    """
    
    df = pd.read_csv("Logs/craco_13ms_survey_db_with_weightsB.csv")
    
    fcut = 1100 # threshold frequency between low and high
    low = np.where(df["fbar"] < fcut)[0]
    mid = np.where(df["fbar"] >= fcut)[0]
    
    low_max_dm = np.array(df["maxdm"][low])
    # orders from highest to lowest
    order = np.argsort(-low_max_dm)
    low_max_dm_sorted = low_max_dm[order]
    
    lteffs = np.array(df["t_eff"][low])/3600.
    lteffs *= np.array(df["bfactors"][low])
    
    lteffs_sorted = lteffs[order]
    cum_lteffs_sorted = np.cumsum(lteffs_sorted)
    
    
    mid_max_dm = np.array(df["maxdm"][mid])
    # orders from highest to lowest
    order = np.argsort(-mid_max_dm)
    mid_max_dm_sorted = mid_max_dm[order]
    
    midteffs = np.array(df["t_eff"][mid])/3600.
    midteffs *= np.array(df["bfactors"][mid])
    midteffs_sorted = midteffs[order]
    cum_midteffs_sorted = np.cumsum(midteffs_sorted)
    
    
    
    ##### Interpolating and saving ######
    
    savedir = os.path.join(resources.files('zdm'), 'data','Efficiencies')
    
    
    ndm = 1001
    mid_dm_mask = np.zeros([2,ndm])
    dmvals = np.linspace(0,10000,ndm)
    
    mdm_mask = np.interp(dmvals, mid_max_dm_sorted[::-1], cum_midteffs_sorted[::-1])
    mid_dm_mask[0,:] = dmvals
    mid_dm_mask[1,:] = mdm_mask
    
    savefile = os.path.join(savedir,'craco_1300_mask.npy')
    np.save(savefile,mid_dm_mask)
    
    # increments of 10 in DM
    low_dm_mask = np.zeros([2,ndm])
    # switches to be lowest DM to highest DM
    ldm_mask = np.interp(dmvals, low_max_dm_sorted[::-1], cum_lteffs_sorted[::-1])
    low_dm_mask[0,:] = dmvals
    low_dm_mask[1,:] = ldm_mask
    savefile = os.path.join(savedir,'craco_900_mask.npy')
    np.save(savefile,low_dm_mask)
    
    
    
    ######## plots the results #######
    
    # simle figure showing sorted max dm observations
    #plt.figure()
    #plt.plot(low_max_dm_sorted,label="Low")
    #plt.ylabel("Maximum DM searched")
    #plt.xlabel("Number of scans")
    #plt.close()
    
    plt.figure()
    plt.ylim(0,1700)
    plt.xlim(0,20000)
    plt.ylabel("$T_{\\rm eff}$ [hr]")
    plt.xlabel("DM [pc cm$^{-3}$]")
    plt.plot(low_max_dm_sorted,cum_lteffs_sorted,label="900 MHz")
    plt.plot(dmvals,ldm_mask,label=" (interpolated)", linestyle=":")
    plt.plot(mid_max_dm_sorted,cum_midteffs_sorted,label="1300 MHz")
    plt.plot(dmvals,mdm_mask,label=" (interpolated)", linestyle=":")
    plt.legend()
    plt.tight_layout()
    plt.savefig("max_searched_dm.png")
    plt.close()

main()
