"""
Generates mean DM response curves for each of the CRACO observations
These are single curves giving the average max DM observed

Saves these as "masks" in zDM, and creates plots of said masks
"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import importlib.resources as resources
import os

import matplotlib

defaultsize=14
ds=4
font = {'family' : 'Helvetica',
        'weight' : 'normal',
        'size'   : defaultsize}
matplotlib.rc('font', **font)



def main():
    """
    A wrapper around analyseDM, just to run it for both
    3.4ms and 13.8ms surveys
    """
    
    #analyseDM("craco_13ms_survey_db.weight.altaz.csv")
    analyseDM("craco_3ms_survey_db.csv",prefix="3ms_")

def analyseDM(logfile,prefix=""):
    """
    Main program to run masks and plotting for
    - low and mid frequencies
    - raw and effective observation time
    
    Args:
        logfile [string]: name of logfile
        prefix [string]: prefix for output
    """
    
    #logfile="craco_13ms_survey_db.weight.altaz.csv"
    #logfile="craco_3ms_survey_db.csv"
    
    df = pd.read_csv("Logs/"+logfile)
    
    fcut = 1100 # threshold frequency between low and high
    low = np.where(df["fbar"] < fcut)[0]
    mid = np.where(df["fbar"] >= fcut)[0]
    
    
    savedir = os.path.join(resources.files('zdm'), 'data','Efficiencies')
    dmvals = np.linspace(0,10000,1000)
    low_max_dm = np.array(df["maxdm"][low])
    mid_max_dm = np.array(df["maxdm"][mid])
    teffs = np.array(df["t_eff"]*df["bfactors"])/3600.
    lteffs=teffs[low]
    mteffs=teffs[mid]
    
    # effective observation times  
    l_max_dm_sorted,l_cum_teffs_sorted = make_dm_mask(dmvals,lteffs,low_max_dm,savedir,prefix+'craco_900_mask.npy')
    m_max_dm_sorted,m_cum_teffs_sorted = make_dm_mask(dmvals,mteffs,mid_max_dm,savedir,prefix+'craco_1300_mask.npy')
    
    plot_dm_masks([l_max_dm_sorted,m_max_dm_sorted],[l_cum_teffs_sorted,m_cum_teffs_sorted],
                ["900 MHz","1300 MHz"],"Plots/"+prefix+"max_searched_dm.png","$T_{\\rm eff}$ [hr]")
    
    
    # raw observation times
    ltraws=np.array(df["tobs"][low])/3600
    mtraws=np.array(df["tobs"][mid])/3600
    l_max_dm_sorted,l_cum_traws_sorted = make_dm_mask(dmvals,ltraws,low_max_dm,savedir,None)
    m_max_dm_sorted,m_cum_traws_sorted = make_dm_mask(dmvals,mtraws,mid_max_dm,savedir,None)
    
    plot_dm_masks([l_max_dm_sorted,m_max_dm_sorted],[l_cum_traws_sorted,m_cum_traws_sorted],
                ["900 MHz","1300 MHz"],"Plots/"+prefix+"raw_max_searched_dm.png","$T_{\\rm obs}$ [hr]")
    
def make_dm_mask(dmvals,teffs,max_dms,savedir,savename):
    """
    Makes a "DM mask" - cumulative distribution of fraction of time
    a given DM has been searched.
    
    Args:
        dmvals (np.ndarray): dispersion measure values to create mask at
        teffs (np.ndarray): list of effective observations times
        max_dms (np.ndarray): list of maximum searched dms corresponding to teffs
        savedir: directory to save data in
        savename: name of save file
    
    Returns:
        cumulative distribution of max DMs
    
    """    
    
    #### orders from highest to lowest ###
    order = np.argsort(-max_dms)
    max_dm_sorted = max_dms[order]
    
    teffs_sorted = teffs[order]
    cum_teffs_sorted = np.cumsum(teffs_sorted)
    
    ##### Interpolating and saving ######
    
    # increments of 10 in DM
    ndm = dmvals.size
    dm_mask = np.zeros([2,ndm])
    
    # switches to be lowest DM to highest DM
    idm_mask = np.interp(dmvals, max_dm_sorted[::-1], cum_teffs_sorted[::-1])
    dm_mask[0,:] = dmvals
    dm_mask[1,:] = idm_mask/idm_mask[0]
    if savename is not None:
        savefile = os.path.join(savedir,savename)
        np.save(savefile,dm_mask)
    
    return max_dm_sorted,cum_teffs_sorted

def plot_dm_masks(max_dms,cum_teffs_sorteds,labels,savefile,ylabel):
    """
    Makes a plot of obsrvation time for each maximum DM
    
    Args:
        max_dms (list of np.ndarray): list of maximum dispersion measures
        cum_teffs_sorted (list of np.ndarray): list of effective observation times
        labels (list of strings): labels for each item in list when plotted
        savefile (string): name of plot file to save
        ylabel (string): y axis label
    """    
    ######## plots the results #######
    
    styles=["-","--",":","-."]
    
    plt.figure()
    plt.ylim(0,3000)
    plt.xlim(0,5000)
    plt.ylabel(ylabel)
    plt.xlabel("DM [pc cm$^{-3}$]")
    
    for i,max_dm_sorted in enumerate(max_dms):
        max_dm_sorted = np.concatenate((max_dm_sorted,np.array([0])))
        cum_teffs_sorted = cum_teffs_sorteds[i]
        cum_teffs_sorted = np.concatenate((cum_teffs_sorted,np.array([cum_teffs_sorted[-1]])))
        
        plt.plot(max_dm_sorted,cum_teffs_sorted,label=labels[i],linestyle = styles[i])
    plt.legend()
    plt.tight_layout()
    plt.savefig(savefile)
    plt.close()

main()
