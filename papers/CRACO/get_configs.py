

import numpy as np
import pandas as pd
import os
from matplotlib import pyplot as plt

def main():
    """
    Reads in CRACO log file, extracts unique configurations,
    produces observation time for each unique configuration
    """
    infile = "Logs/craco_13ms_survey_db.weight.altaz.csv"
    df = read_logfile(infile = infile)
    
    # adds statistical weighting factors
    # turn on if extra info is not yet generated
    df = add_columns(df,infile) 
    
    # gets unique configurations
    logfile="Logs/configs.csv"
    if not os.path.exists(logfile):
        get_unique(df,logfile)
    
    
    ##### 3.4ms survey ####
    
    infile = "Logs/craco_3ms_survey_db.csv"
    df = read_logfile(infile = infile)
    
    # adds statistical weighting factors
    # turn on if extra info is not yet generated
    df = add_columns(df,infile) 
    
    # gets unique configurations
    # gets unique configurations
    logfile="Logs/3ms_configs.csv"
    if not os.path.exists(logfile):
        get_unique(df,logfile)
    
    
    
def add_columns(df,opfile):
    """
    Adds statistical weighting factors and other derived data
    Args:
        df: pandas dataframe containing observation info
        opfile [string]: name of the output logfile
    """
    
    
    # sens is proportional to square root of bandwidth
    # rate goes as ^1.5. Hence, power of 0.75
    maxchan = np.max(df["nchans"])
    print("maxchan is ",maxchan)
    print("Max channels is ",maxchan)
    w_bw = (df["nchans"]/maxchan)**0.75
    df["w_bandwidth"] = w_bw
    
    # just go linearly with this
    w_gf = df["goodfrac"]**1.5 # we lose signal linearly with bandwidth
    df["w_goodfrac"] = w_gf
    
    # sensitivity to the power of 1.5
    maxant = np.max(df["nant"])
    sense = (df["nant"]*(df["nant"]-1))**0.5
    maxsense = (maxant*(maxant-1))**0.5
    w_nant = (sense/maxsense)**1.5
    df["w_nant"] = w_nant
    
    # observation time
    df["tobs"] = df["nsamples"] * df["tsamp"]
    
    # total weight including all factors
    df["w_tot"] = df["w_nant"] * df["w_goodfrac"] * df["w_bandwidth"]
    
    # total effective time (seconds)
    df["t_eff"] = df["tobs"] * df["w_tot"]
    
    
    df["fbar"] = (df["fmin"] + df["fmax"])/2.
    
    df.to_csv(opfile,index=False)
    return df
    
def read_logfile(infile = "Logs/craco_13ms_survey_db.csv"):
    """
    read logfile
    Args:
        infile [string]: name of logfile
    
    Returns:
        df: pandas dataframe containing log info
    """
    
    df = pd.read_csv(infile)
    
    #print(df.columns)
    return df
    
def get_unique(df,opfile):
    """
    gets unique combinations of footprint, frequency, and pitch
    
    Args:
        df: pandas dataframe containing log info
        opfile: name of output file
    """
    
    footprints = np.unique(df["footprint"])
    
    
    config_fp=[]
    config_pitch=[]
    config_fb=[]
    config_total=[]
    config_wtotal=[]
    
    for fp in footprints:
        OK = np.where(df["footprint"] == fp)[0]
        fppitch = df["pitch"][OK]
        pitches = np.unique(fppitch)
        for pitch in pitches:
            OK2 = np.where(fppitch[OK]==pitch)
            # all original indices matching this
            indices2 = OK[OK2]
            fp_p_fbar = df["fbar"][indices2]
            
            fbars = np.unique(fp_p_fbar)
            for fb in fbars:
                OK3 = np.where(fp_p_fbar == fb)[0]
                indices3 = indices2[OK3]
                fp_p_fb_tobs = df["tobs"][indices3]
                fp_p_fb_Nant = df["nant"][indices3]
                weights = df["w_tot"][indices3]
                
                # normalising sensitivity according to Nant
                # relative to Nant = 24, which is max in data
                total = np.sum(fp_p_fb_tobs)/3600
                wtotal = np.sum(fp_p_fb_tobs*weights)/3600
                print(fp,pitch,fb,total,wtotal)
                config_fp.append(fp)
                config_pitch.append(pitch)
                config_fb.append(fb)
                config_total.append(total)
                config_wtotal.append(wtotal)
    fp = np.array(config_fp)
    pitch = np.array(config_pitch)
    fb = np.array(config_fb)
    total = np.array(config_total)
    wtotal = np.array(config_wtotal)
    data = {"footprint": fp,
            "pitch": pitch,
            "fbar": fb,
            "Ttot": total,
            "Teff": wtotal}
    
    df = pd.DataFrame(data)
    df.to_csv(opfile,index=False)


main()
