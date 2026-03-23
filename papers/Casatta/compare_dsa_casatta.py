"""
Runs a simulation of DSA 1600, compartes that to CASATTA N...
"""
from scipy.interpolate import interp1d
import pandas as pd
import numpy as np
import importlib.resources as resources
import copy
import scipy.constants as constants
from matplotlib import pyplot as plt

from zdm import states
from zdm import misc_functions as mf
from zdm import grid as zdm_grid
from zdm import survey
from zdm import pcosmic


import matplotlib

defaultsize=14
ds=4
font = {'family' : 'Helvetica',
        'weight' : 'normal',
        'size'   : defaultsize}
matplotlib.rc('font', **font)

def main():
    """
    
    """
    # does exactly the same as "sim_casatta.py", just picks out the
    # specific cases we want
    
    # loads
    #df = read_casatta_params()
    #nsims,ncols = df.shape
    
   
    
    ##### SIMULATES DSA 1600 as per CASATTA ######3
    
    # state. Does not use updated scattering, because it takes a long time!
    state = states.load_state("HoffmannHalo25")#scat="updated",rep=None)
    
    zDMgrid, zvals, dmvals = mf.get_zdm_grid(
                    state, new=True, plot=False, method='analytic', 
                    datdir=resources.files('zdm').joinpath('GridData'))
    
    # we can keep this constant - it smears DM due to host DM
    mask = pcosmic.get_dm_mask(dmvals, (state.host.lmean, state.host.lsigma), zvals, plot=False)
    
    renorm = get_constant(state,zDMgrid, zvals, dmvals, mask)
    
    sdir="."
    survey_name = "DSA_1600"
    
    s = survey.load_survey(survey_name, state, dmvals, zvals=zvals, sdir=sdir)
    
    g = zdm_grid.Grid(s, copy.deepcopy(state), zDMgrid, zvals, dmvals, mask, wdist=True)
    
    
    daily = np.sum(g.rates)* 10**state.FRBdemo.lC *renorm
    
    pz = np.sum(g.rates,axis=1)* 10**state.FRBdemo.lC *renorm
    pdm = np.sum(g.rates,axis=0)* 10**state.FRBdemo.lC *renorm
    dz1 = zvals[1]-zvals[0]
    print("Daily DSA rate is ",daily)
    ###### reads CASATTA #####
    
    df = pd.read_csv("CASATTA MFAA SKA2 FRB estimates.csv")
    
    dailys = np.load("dailys.npy")
    pzs = np.load("pzs.npy")*renorm
    pdms = np.load("pdms.npy")*renorm
    zvals2 = np.load("zvals.npy")
    dmvals = np.load("dmvals.npy")
    threshs = np.load("threshs.npy")
    
    
    
    # selects which casatta plot to use
    OK1=np.where(df["FrequencyMHz"]==600)
    OK2=np.where(df["FWHM_deg"]==120)
    OK=np.intersect1d(OK1,OK2)
    
    dz2 = zvals2[1]-zvals2[0]
    
    # selects which casatta we want
    ic = 13
    print(df["Array_name"][ic]," has daily rate ",dailys[ic])
    # plots!
    
    optdir = str(resources.files('zdm').joinpath('data/optical'))+"/"
    
    yf = np.load(optdir+"fz_23.0.npy")
    xf = np.load(optdir+"zvals.npy")
    itp = interp1d(xf,yf)
    hfracs = itp(zvals)
    
    yf2 = np.load(optdir+"fz_24.7.npy")
    xf2 = np.load(optdir+"zvals.npy")
    itp2 = interp1d(xf2,yf2)
    hfracs2 = itp2(zvals)
    
    
    yf3 = np.load(optdir+"fz_27.5.npy")
    xf3 = np.load(optdir+"zvals.npy")
    itp3 = interp1d(xf3,yf3)
    hfracs3 = itp3(zvals)
    
    plt.figure()
    plt.xlim(0,5)
    plt.ylim(0,550)
    # multiplies by z-bin width
    dz = zvals[1]-zvals[0]
    plt.xlabel("z")
    plt.ylabel("p(z) [FRBs / day / z]")
    #plt.ylim(1e-1,1e7)
    plt.plot(zvals,pz/dz1,label="DSA 1600: FRBs",color="red")
    plt.fill_between(zvals,pz/dz1*hfracs,label="   hosts: DECaLS",color="red",alpha=0.5)
    print("DSA hosts: ",np.sum(pz*hfracs))
    plt.plot(zvals,pzs[ic,:]/dz2,label="CASATTA 4000: FRBs",color="blue")
    plt.fill_between(zvals,pzs[ic,:]/dz2*hfracs3,label="   hosts: LSST",color="blue",alpha=0.5)
    print("CASATTA hosts: ",np.sum(pzs[ic,:]*hfracs3))
    plt.legend()
    plt.tight_layout()
    plt.savefig("dsa_pz_30_vs_27.5.png")
    plt.close()
    
    
    
    plt.figure()
    plt.xlim(0,5)
    plt.ylim(0,550)
    # multiplies by z-bin width
    dz = zvals[1]-zvals[0]
    plt.xlabel("z")
    plt.ylabel("p(z) [FRBs / day / z]")
    #plt.ylim(1e-1,1e7)
    plt.plot(zvals,pz/dz1,label="DSA 1600: FRBs",color="red")
    plt.fill_between(zvals,pz/dz1*hfracs2,label="DSA 1600: hosts",color="red",alpha=0.5)
    print("DSA hosts: ",np.sum(pz*hfracs2))
    plt.plot(zvals,pzs[ic,:]/dz2,label="CASATTA 4000: FRBs",color="blue")
    plt.fill_between(zvals,pzs[ic,:]/dz2*hfracs2,label="CASATTA 4000: hosts",color="blue",alpha=0.5)
    print("CASATTA hosts: ",np.sum(pzs[ic,:]*hfracs2))
    plt.legend()
    plt.tight_layout()
    plt.savefig("dsa_pz_both_24.7.png")
    plt.close()


def get_constant(state,zDMgrid, zvals, dmvals, mask):
    """
    gets a normalising constant for this state
    
    Args:
        df: dataframe containing info for this version of casatta
        state: zdm state object
        zDMgrid: underlying zDM grid giving p(DMcosmic|z)
        zvals: redshift values of grid
        dmvals: DM values of grid
        mask: DM smearing mask for grid based on DMhost
    """
    # I am here choosing to renomalise by the CRAFT ICS 892 MHz rates
    #norm_survey = "CRAFT_class_I_and_II"
    norm_survey = "CRAFT_ICS_892"
    s = survey.load_survey(norm_survey, state, dmvals, zvals=zvals)
    g = zdm_grid.Grid(s, copy.deepcopy(state), zDMgrid, zvals, dmvals, mask, wdist=True)
        
    predicted = np.sum(g.rates) * s.TOBS * 10**state.FRBdemo.lC
    observed = s.NORM_FRB
    
    renorm =  observed/predicted
    print("Calculated renomalisation constant as ",renorm)
    return renorm

main()
