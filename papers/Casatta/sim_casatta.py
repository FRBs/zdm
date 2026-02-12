
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
    main file to simulate casatta sensitivity
    """
    df = read_casatta_params()
    nsims,ncols = df.shape
    
    # state. Does not use updated scattering, because it takes a long time!
    state = states.load_state("HoffmannHalo25")#scat="updated",rep=None)
    
    zDMgrid, zvals, dmvals = mf.get_zdm_grid(
                    state, new=True, plot=False, method='analytic', 
                    datdir=resources.files('zdm').joinpath('GridData'))
    
    # we can keep this constant - it smears DM due to host DM
    mask = pcosmic.get_dm_mask(dmvals, (state.host.lmean, state.host.lsigma), zvals, plot=False)
    
    # gets constant of total FRB rate to normalise to
    renorm = get_constant(state,zDMgrid,zvals,dmvals,mask)
    
    threshs = np.zeros([nsims])
    dailys = np.zeros([nsims])
    pzs = np.zeros([nsims,zvals.size])
    pdms = np.zeros([nsims,dmvals.size])
    
    for isim in np.arange(nsims):
        daily,pz,pdm,thresh = sim_casatta(df.iloc[isim],state,zDMgrid,zvals,dmvals,mask)
        dailys[isim]=daily
        pzs[isim,:]=pz
        pdms[isim,:]=pdm
        threshs[isim] = thresh
        print("Done simulation ",isim, " of ", nsims,", daily rate ",daily*renorm)
    
    # modifies rates according to expectations
    dailys *= renorm
    np.save("threshs.npy",threshs)
    np.save("dailys.npy",dailys)
    np.save("pzs.npy",pzs)
    np.save("pdms.npy",pdms)
    np.save("zvals.npy",zvals)
    np.save("dmvals.npy",dmvals)

    
def read_casatta_params(infile="CASATTA MFAA SKA2 FRB estimates.csv"):
    """
    Reads in casatta parameters
    """
    
    import pandas as pd
    df = pd.read_csv(infile)
    
    return df

def sim_casatta(df,state,zDMgrid, zvals, dmvals, mask):
    """
    simulates casatta for specific values given in dataframe
    
    Args:
        df: dataframe containing info for this version of casatta
        state: zdm state object
        zDMgrid: underlying zDM grid giving p(DMcosmic|z)
        zvals: redshift values of grid
        dmvals: DM values of grid
        mask: DM smearing mask for grid based on DMhost
    """
    
    # base name = casatta.ecsv
    # directory where the base survey lives
    sdir = resources.files('zdm').joinpath('../papers/Casatta/')
    sdir = str(sdir) # convert to string, not annoying object
    
    ########### calculates relevant properties for CASATTA #######
    BW = df["BandwidthMHz"]
    BWHz = BW*1e6
    tms = 1e-3 # seconds in one ms
    
    # SEFD in Jy.
    NSAMP = 4.*BWHz *tms # 2 for polarisation, 2 for Nyquist, 1e3 for Jyms
    nsigma=10 # S/N requirement for detection
    THRESH = nsigma*df['SEFDJy']/NSAMP**0.5
    
    FBAR = df["FrequencyMHz"]
    tres = df["Time_resolution_ms"]
    fres = df["Freq_res_MHz"]
    
    fMHz = df['FrequencyMHz']
    #print(df['Array_name'],df['SEFDJy'],NSAMP,THRESH)
    
    
    # calculates what D to use
    FWHM_rad = df["FWHM_deg"]*np.pi/180.
    DIAM=1.22*(constants.c/(fMHz*1e6))/FWHM_rad
    
    # generates a survey dict to modify properties of this survey
    # TOBS is one day
    survey_dict = {"THRESH": THRESH, "TOBS": 1, "FBAR": float(fMHz), "BW": float(BW), "DIAM": DIAM,
                    "FRES": float(fres), "TRES": float(tres)}
    
    survey_name = "casatta_base"
    s = survey.load_survey(survey_name, state, dmvals, zvals=zvals, survey_dict=survey_dict, sdir=sdir)
    
    g = zdm_grid.Grid(s, copy.deepcopy(state), zDMgrid, zvals, dmvals, mask, wdist=True)
    
    
    daily = np.sum(g.rates)* 10**state.FRBdemo.lC
    
    pz = np.sum(g.rates,axis=1)* 10**state.FRBdemo.lC
    pdm = np.sum(g.rates,axis=0)* 10**state.FRBdemo.lC
    
    return daily,pz,pdm,THRESH

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
    
    
