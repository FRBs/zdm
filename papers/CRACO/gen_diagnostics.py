"""
This file prints average weighting factors for CRACO observations

It also plots a whole bunch of diagnostic plots
"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from zdm import misc_functions as mf
import matplotlib

defaultsize=14
ds=4
font = {'family' : 'Helvetica',
        'weight' : 'normal',
        'size'   : defaultsize}
matplotlib.rc('font', **font)


def main():
    """
    main file, to iterate over different CRACO surveys
    """
    
    #13.8ms survey
    print_metrics("Logs/craco_13ms_survey_db.weight.altaz.csv",frblog="Logs/CRACO_13.8ms_zdm.dmgal.altaz.mjd.csv")
    
    #3.4ms survey
    print_metrics("Logs/craco_3ms_survey_db.csv",prefix="3ms_")
    
def print_metrics(logfile,prefix="",frblog=None):
    """
    Prints metrics for given logfile.
    
    Args:
        logfile [string]: name of observation logfile.
        prefix [string]: prefix to append to outouts
        frblog [string or None]: if present, generate a cumulative plot
                                of FRB detection rates vs observation time
    """
    
    df = pd.read_csv(logfile)
    print(df.columns)
    
    LOW = np.where(df["fbar"] < 1000)[0]
    HIGH = np.where(df["fbar"] > 1000)[0]
    
    # prints some characteristic values
    print_mean_values(df,LOW,HIGH)
    
    # produces some basic plots
    do_basic_plots(df,LOW,HIGH,prefix)
    
    # plots example of sampling time effet
    plot_tsamp()
    
    if frblog is not None:
        frbs = pd.read_csv(frblog)
    
        # produces plot of cumulative effective and normal time vs detected FRBs
        plot_cumulative(df,LOW,HIGH,frbs,ks=True)
    
        # load_frbs
        match_values(df,frbs)
    
def match_values(df,frbs):
    """
    For each frb, get slices corresponding to which observation they were found in
    
    Args:
        df: pandas dataframe containing logfile info
        frbs: pandas dataframe containing observed FRB info
    """
    scans = frbs["scan"]
    for i,scan in enumerate(scans):
        j = np.where(df["scan"] == scan)[0]
        fbar = df["fbar"][j].to_string(index=False, header=False)
        nchan = df["nchans"][j].to_string(index=False, header=False)
        print(i,fbar,nchan)
    
    
    
def print_mean_values(df,LOW,HIGH):
    """
    Prints mean values of various quantities
    
    args:
        df: pandas dataframe containing logfile info
        LOW [list]: indices of data frame corresponding to 900 MHz sample
        HIGH [list]: indices of data frame corresponding to 1300 MHz sample
    """
    
    
    Ttot = np.sum(df["tobs"])
    LTtot = np.sum(df["tobs"][LOW])
    HTtot = np.sum(df["tobs"][HIGH])
    LTeff = np.sum(df["t_eff"][LOW])
    HTeff = np.sum(df["t_eff"][HIGH])
    print("Total time is ",Ttot/3600," with ",LTtot/3600," at low, and ",HTtot/3600," at high")
    print("Effective time includes bandwidth, Nant, and gf; NOT beam, DM, or width")
    print("Total effective time is ",LTeff/3600," at low, and ",HTeff/3600," at high")
    print("Mean high frequency is ",np.sum(df["tobs"][HIGH]*df["fbar"][HIGH])/HTtot)
    print("Mean low frequency is ",np.sum(df["tobs"][LOW]*df["fbar"][LOW])/LTtot,"\n\n\n")
    
    
    # on average, we have lost 0.913 due to bandwidth
    print("mean bandwidth ",np.sum(df['nchans']*df["tobs"])/Ttot)
    print("Mean bandwidth loss ",np.sum(df['w_bandwidth']*df["tobs"])/Ttot)
    print("Mean HIGH bandwidth loss ",np.sum(df['w_bandwidth'][HIGH]*df["tobs"][HIGH])/HTtot)
    print("Mean LOW bandwidth loss ",np.sum(df['w_bandwidth'][LOW]*df["tobs"][LOW])/LTtot,"\n\n")
    
    
    # then we lose 70% due to "goodfrac"
    print("Mean goodfrac ",np.sum(df['goodfrac']*df["tobs"])/Ttot)
    print("Mean gf loss ",np.sum(df['w_goodfrac']*df["tobs"])/Ttot)
    print("Optimal gf loss ",np.sum(df['goodfrac']**0.75*df["tobs"])/Ttot)
    print("Mean LOW gf loss ",np.sum(df['w_goodfrac'][LOW]*df["tobs"][LOW])/LTtot)
    print("Mean HIGH gf loss ",np.sum(df['w_goodfrac'][HIGH]*df["tobs"][HIGH])/HTtot,"\n\n")
    
    # then we lose 91% due to less than 25 antennas
    print("Max number of antennas is ",np.max(df['nant']))
    print("Mean antennas ",np.sum(df['nant']*df["tobs"])/Ttot)
    print("Mean nant loss ",np.sum(df['w_nant']*df["tobs"])/Ttot)
    print("Low nant loss ",np.sum(df['w_nant'][LOW]*df["tobs"][LOW])/LTtot)
    print("High nant loss ",np.sum(df['w_nant'][HIGH]*df["tobs"][HIGH])/HTtot,"\n\n")
    
    


def plot_cumulative(df,LOW,HIGH,frbs,ks=True):
    """
    Generates some cumulative plots
    
    args:
        df: pandas dataframe containing logfile info
        LOW [list]: indices of data frame corresponding to 900 MHz sample
        HIGH [list]: indices of data frame corresponding to 1300 MHz sample
        frbs: pandas dataframe giving FRB info
        ks [bool]: if True, perform a ks test on the two distributions
                    for consistency
    """
    
    teff = df["t_eff"]*df["bfactors"]
    
    ctraw = np.cumsum(df["tobs"]/3600)
    Lctraw = np.cumsum(df["tobs"][LOW]/3600)
    Hctraw = np.cumsum(df["tobs"][HIGH]/3600)
    
    cteff = np.cumsum(teff/3600)
    Lcteff = np.cumsum(teff[LOW]/3600)
    Hcteff = np.cumsum(teff[HIGH]/3600)
    
    frbxs,frbys=mf.make_cum_dist(frbs["mjd"])
    frbys *= len(frbs["mjd"])
    
    
    if ks:
        from scipy.stats import kstest
        from scipy.interpolate import interp1d
        rvs = frbs["mjd"]
        cum_func = interp1d(df["tstart"],cteff/cteff.values[-1],kind="linear",assume_sorted=True)
        result = kstest(rvs,cum_func,mode="exact",alternative="two-sided")
        print(result)
    
    plt.figure()
    
    #plt.plot(df["tstart"],ctraw)
    plt.plot(df["tstart"],cteff,label="Total")
    plt.plot(df["tstart"][LOW],Lcteff,label="900 MHz",linestyle="--")
    plt.plot(df["tstart"][HIGH],Hcteff,label="1300 MHz",linestyle=":")
    
    ax = plt.gca()
    # FRBs
    ax2 = ax.twinx()
    ax2.plot(frbxs,frbys,linestyle="-.",color="black")
    ax2.set_ylabel("$N_{\\rm FRB}$")
    plt.ylim(0,18)
    
    plt.sca(ax)
    
    # does a dummy plot
    plt.plot([-1e9,-1e8],[-100,-100],linestyle="-.",color="black",label="CRACO FRBs")
    plt.xlim(60280,60650)
    plt.ylim(0,cteff.values[-1])
    plt.xlabel("mjd")
    plt.ylabel("Cumulative $T_{\\rm eff}$ [hr]")
    plt.legend()
    plt.tight_layout()
    plt.savefig("Plots/eff_cumulative_fig.png")
    plt.close()
    
    plt.figure()
    
    #plt.plot(df["tstart"],ctraw)
    plt.plot(df["tstart"],ctraw,label="Total")
    plt.plot(df["tstart"][LOW],Lctraw,label="900 MHz",linestyle="--")
    plt.plot(df["tstart"][HIGH],Hctraw,label="1300 MHz",linestyle=":")
    
    # FRBs
    ax = plt.gca()
    ax2 = ax.twinx()
    ax2.plot(frbxs,frbys)
    ax2.set_ylabel("$N_{\\rm FRB}$")
    plt.ylim(0,18)
    
    plt.sca(ax)
    plt.ylim(0,ctraw.values[-1])
    plt.xlabel("mjd")
    plt.ylabel("Cumulative $T_{\\rm obs}$ [hr]")
    plt.legend()
    plt.tight_layout()
    plt.savefig("Plots/raw_cumulative_fig.png")
    plt.close()
    
def do_basic_plots(df,LOW,HIGH,prefix):
    """
    Produces basic plots
    
    Args:
        df: pandas dataframe containing info
        LOW: indices corresponding to low frequencies
        HIGH: indices corresponding to high frequencies
        prefix [string]: prefix for outputs
        
    """
    
    #### Number of antennas ####
    plt.figure()
    plt.xlabel("Number of antennas")
    plt.ylabel("Total obs time [hr]")
    bins = np.linspace(0.5,36.5,37)
    plt.hist(df["nant"],bins=bins,weights=df["tobs"]/3600)
    plt.xlim(14,26)
    plt.tight_layout()
    plt.savefig("Plots/"+prefix+"Nant_hist.png")
    plt.close()
    
    #### Bandwidth ####
    plt.figure()
    plt.xlabel("Bandwidth [MHz]")
    plt.ylabel("Total obs time [hr]")
    bins = np.linspace(5,345,35)
    
    
    plt.hist(df["nchans"],bins=bins,weights=df["tobs"]/3600)
    #plt.xlim(14,26)
    plt.tight_layout()
    plt.savefig("Plots/"+prefix+"bw_hist.png")
    plt.close()
    
    ##### central frequency ###
    plt.figure()
    plt.xlabel("Central frequency [MHz]")
    plt.ylabel("Total obs time [hr]")
    bins = np.linspace(0,2000,201)
    plt.hist(df["fbar"],bins=bins,weights=df["tobs"]/3600)
    plt.xlim(750,1500)
    plt.tight_layout()
    plt.savefig("Plots/"+prefix+"Fbar_hist.png")
    plt.close()
    
    
    
    
    ##### good fraction ######
    plt.figure()
    plt.xlabel("Fraction of good data")
    plt.ylabel("Total obs time [hr]")
    bins = np.linspace(0,1,101)
    plt.hist(df["goodfrac"],bins=bins,weights=df["nsamples"]*df["tsamp"]/3600)
    #plt.xlim(14,26)
    plt.tight_layout()
    plt.savefig("Plots/"+prefix+"gf_hist.png")
    plt.close()
    
    plt.figure()
    plt.xlabel("Fraction of good data")
    plt.ylabel("Total obs time [hr]")
    bins = np.linspace(0,1,101)
    plt.hist(df["goodfrac"][LOW],bins=bins,weights=df["nsamples"][LOW]*df["tsamp"][LOW]/3600,label="900 MHz")
    plt.hist(df["goodfrac"][HIGH],bins=bins,weights=df["nsamples"][HIGH]*df["tsamp"][HIGH]/3600,label="1300 MHz")
    plt.legend()
    plt.tight_layout()
    plt.savefig("Plots/"+prefix+"gf_hist_by_freq.png")
    plt.close()
    
    

def plot_tsamp():
    """
    Produces a dummy plot of the sampling time effect
    """
    
    NW=61
    wvals = np.logspace(-2,3,NW)
    ICSthresh = 4.4
    ICS = np.full([NW],4.4*(1.28**0.5))
    CRACOthresh = ICSthresh * 25**0.5 / (24*23)**0.5 * (336/288)**0.5
    print("CRACO thresh is ",CRACOthresh) # why 0.99?
    CRACO = np.full([NW],CRACOthresh*(13.8**0.5))
    
    
    OK = np.where(wvals > 1.28)[0]
    ICS[OK] *= (wvals[OK]/1.28)**0.5
    BAD = np.where(wvals > 1.28*12.5)[0]
    ICS[BAD] *= 1000
    
    OK = np.where(wvals > 13.8)[0]
    CRACO[OK] *= (wvals[OK]/13.8)**0.5
    OK = np.where(wvals > 13.8*8)[0]
    CRACO[OK] *= (wvals[OK]/(13.8*8))**0.5
    
    
    plt.figure()
    plt.xlabel("FRB effective width [ms]")
    plt.ylabel("$F_{\\rm th}$ [Jy ms]")
    plt.ylim(1,50)
    plt.plot(wvals,ICS,label="ICS",linestyle="--")
    c1 = plt.gca().lines[-1].get_color()
    plt.plot(wvals,CRACO,label="CRACO")
    c2 = plt.gca().lines[-1].get_color()
    
    
    plt.plot([13.8,13.8],[0.1,CRACO[0]],color="black",linestyle=":")
    plt.plot([1.28,1.28],[0.1,ICS[0]],color="black",linestyle=":")
    
    plt.plot([13.8*8,13.8*8],[0.1,CRACO[0]*2.83],color="black",linestyle=":")
    plt.plot([1.28*12,1.28*12],[0.1,ICS[0]*3.46],color="black",linestyle=":")
    
    plt.text(2,5.5,"$F_{\\rm th} \\sim w^{0.5}$",rotation=45,color=c1,fontsize=12)
    plt.text(25,3.9,"$F_{\\rm th} \\sim w^{0.5}$",rotation=45,color=c2,fontsize=12)
    plt.text(200,15,"$F_{\\rm th} \\sim w$",rotation=66,color=c2,fontsize=12)
    
    
    plt.text(0.8,1.2,"$t_{\\rm obs} = 1.28$ ms",rotation=90,color=c1,fontsize=12)
    plt.text(9,1.2,"$t_{\\rm obs} = 13.8$ ms",rotation=90,color=c2,fontsize=12)
    
    
    plt.text(18,1.4,"$12 \\times t_{\\rm obs}$",rotation=90,color=c1,fontsize=12)
    plt.text(120,2,"$8 \\times t_{\\rm obs}$",rotation=90,color=c2,fontsize=12)
    
    plt.xscale("log")
    plt.yscale("log")
    plt.legend()
    plt.tight_layout()
    plt.savefig("Plots/width_threshold_sketch.png")
    plt.close()

    
main()
