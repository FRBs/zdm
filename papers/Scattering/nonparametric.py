import numpy as np
from matplotlib import pyplot as plt
import scipy as sp
import matplotlib

import sys
import os


import pandas

defaultsize=16
ds=4
font = {'family' : 'Helvetica',
        'weight' : 'normal',
        'size'   : defaultsize}
matplotlib.rc('font', **font)

def main():
    ##### does scatz plot #####
    #infile="table_scatz.dat"
    #scat,scatmid,err,errl,errh,z,DM,DMG,SNR,Class = read_data(infile)
    #scat,scaterr,z,DM,DMG,SNR,Class = read_data(infile)
    
    tresinfo = np.loadtxt("treslist.dat",dtype="str")
    names=tresinfo[:,0]
    
    for i in np.arange(names.size):
        names[i] = names[i][0:8] # truncates the letter
    
    dataframe = pandas.read_csv("CRAFT_ICS_HTR_Catalogue1.csv")
    for key in dataframe.keys():
        print(key)
    
    ERR=9999.
    tns = dataframe.TNS
    tauobs = dataframe.TauObs
    w95 = dataframe.W95
    wsnr = dataframe.Wsnr
    z = dataframe.Z
    snr = dataframe.SNdet
    freq = dataframe.NUTau
    DM = dataframe.DM
    OK = getOK([tns,tauobs,w95,wsnr,z,snr,freq,DM])
    tns = tns[OK]
    tauobs= tauobs[OK]
    w95 = w95[OK]
    wsnr = wsnr[OK]
    z = z[OK]
    snr = snr[OK]
    freq = freq[OK]
    DM = DM[OK]
    NFRB = len(OK)
    tres = np.zeros([NFRB])
    
    for i,name in enumerate(tns):
        j = np.where(name[0:8] == names)[0]
        if len(j) != 1:
            print("Cannot find tres info for FRB ",name)
        tres[i] = float(tresinfo[j,1])
    
    print(tres)
    
def find_max_tau(wsnr,tauobs,DM,freq,snr,tres,nu_res=1.):
    """
    
    """
    
    tauvals = np.linspace(0.1,100.,1001)
    k_DM = 4.149 # leading constant: ms per pccc GHz^2
    dmsmear = 2*(nu_res/1.e3)*k_DM*DM_frb/(fbar/1e3)**3
    
    
    
    totalw = (uw**2 + dm_smearing**2/3. + t_res**2/3.)**0.5
    
    
    print(OK)
    exit()



def getOK(arrays,ERR=9999.):
    """
    Find indices where all arrays have good values
    """
    OK = np.where(arrays[0] != ERR)
    for array in arrays[1:]:
        OK1 = np.where(array != ERR)
        OK = np.intersect1d(OK,OK1)
    return OK

def make_cum_dist(vals):
    """
    Makes a cumulative distributiuon for plotting purposes
    """
    
    # orders the vals smallest to largest
    ovals = np.sort(vals)
    
    Nvals = ovals.size
    Npts = 2*Nvals+2
    xs=np.zeros([Npts])
    ys=np.zeros([Npts])
    # begins at 0,0
    
    for i in np.arange(Nvals):
        xs[2*i+1] = ovals[i]
        xs[2*i+2] = ovals[i]
        ys[2*i+1] = i / Nvals
        ys[2*i+2] = (i+1) / Nvals
    
    if np.max(xs) >1:
        themax = np.max(xs)
    else:
        themax = 1.
    xs[-1] = themax
    ys[-1] = 1
    return xs,ys
    
def read_chime_scat(infile = "chime_scat.dat"):
    """
    gets chime scat err data
    """
    scats=[]
    errs=[]
    with open(infile) as file:
        for line in file:
            fields = line.split()
            scatstring = fields[1].replace(" ", "")
            if scatstring[0] == "<":
                scat = float(scatstring[1:])/2.
                err = scat
            elif scatstring[0] == "~":
                scat = float(scatstring[1:])
                err = float(fields[2].replace(" ", ""))
            else:
                scat = float(scatstring)
                err = float(fields[2].replace(" ", ""))
            scats.append(scat)
            errs.append(err)
    scats = np.array(scats)
    errs = np.array(errs)
    return scats,errs
            
def cutz(DM,DMG,z,scat,err):
    """
    cuts on z>0
    """
    
    
    OK = np.where(z>0.)[0]
    DM = DM[OK]
    DMG = DMG[OK]
    z = z[OK]
    scat = scat[OK]
    err = err[OK]
    return DM,DMG,z,scat,err


main()
