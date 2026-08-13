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
    
    dataframe = pandas.read_csv("CRAFT_ICS_HTR_Catalogue1.csv")
    tauobs = dataframe.TauObs
    w95 = dataframe.W95
    wsnr = dataframe.wsnr
    
    FRB2,taus,tauerrs,ascat,aerr,alphas,alphaerrs = read_scat_table()
    
    #cscat,cerr = read_chime_scat()

    clx,cly = make_cum_dist(cscat-cerr)
    cmx,cmy = make_cum_dist(cscat)
    cux,cuy = make_cum_dist(cscat+cerr)
    
    alx,aly = make_cum_dist(ascat-aerr)
    amx,amy = make_cum_dist(ascat)
    aux,auy = make_cum_dist(ascat+aerr)
    
    a2lx,a2ly = make_cum_dist(taus-tauerrs)
    a2mx,a2my = make_cum_dist(taus)
    a2ux,a2uy = make_cum_dist(taus+tauerrs)
    
    exponent = 0
    plt.figure()
    plt.plot(cmx*(0.6**exponent),cmy,label="CHIME 600 MHz", linewidth=2)
    plt.plot(clx*(0.6**exponent),cly,linestyle="-.",color=plt.gca().lines[-1].get_color())
    plt.plot(cux*(0.6**exponent),cuy,linestyle="-.",color=plt.gca().lines[-1].get_color())
    
    plt.plot(amx,amy,label="ASKAP 1 GHz",linestyle="--", linewidth=2)
    plt.plot(alx,aly,linestyle=":",color=plt.gca().lines[-1].get_color())
    plt.plot(aux,auy,linestyle=":",color=plt.gca().lines[-1].get_color())
    plt.xscale('log')
    #plt.xlabel("$\\tau_{\\rm 1 GHz}$ [ms]")
    plt.xlabel("$\\tau$ [ms]")
    plt.ylabel("Cumulative distribution")
    plt.ylim(0,1)
    plt.legend()
    plt.tight_layout()
    plt.savefig("scattering_comparison_exp"+str(exponent)+".png")
    plt.close()
    
    exponent = 4
    plt.figure()
    plt.plot(cmx*(0.6**exponent),cmy,label="CHIME 600 MHz", linewidth=2)
    plt.plot(clx*(0.6**exponent),cly,linestyle="-.",color=plt.gca().lines[-1].get_color())
    plt.plot(cux*(0.6**exponent),cuy,linestyle="-.",color=plt.gca().lines[-1].get_color())
    
    plt.plot(amx,amy,label="ASKAP 1 GHz",linestyle="--", linewidth=2)
    plt.plot(alx,aly,linestyle=":",color=plt.gca().lines[-1].get_color())
    plt.plot(aux,auy,linestyle=":",color=plt.gca().lines[-1].get_color())
    plt.xscale('log')
    #plt.xlabel("$\\tau_{\\rm 1 GHz}$ [ms]")
    plt.xlabel("$\\tau$ [ms]")
    plt.ylabel("Cumulative distribution")
    plt.ylim(0,1)
    plt.legend()
    plt.tight_layout()
    plt.savefig("scattering_comparison_exp"+str(exponent)+".png")
    plt.close()
    
    
    plt.figure()
    plt.plot(cmx*(0.6**exponent),cmy,label="CHIME 600 MHz")
    plt.plot(clx*(0.6**exponent),cly,linestyle=":",color=plt.gca().lines[-1].get_color())
    plt.plot(cux*(0.6**exponent),cuy,linestyle=":",color=plt.gca().lines[-1].get_color())
    
    plt.plot(a2mx,a2my,label="ASKAP unscaled",linestyle="--")
    plt.plot(a2lx,a2ly,linestyle="-.",color=plt.gca().lines[-1].get_color())
    plt.plot(a2ux,a2uy,linestyle="-.",color=plt.gca().lines[-1].get_color())
    plt.xscale('log')
    #plt.xlabel("$\\tau_{\\rm 1 GHz}$ [ms]")
    plt.xlabel("$\\tau$ [ms]")
    plt.ylabel("Cumulative distribution")
    plt.legend()
    plt.ylim(0,1)
    plt.tight_layout()
    plt.savefig("scattering_comparison_noscaling.png")
    plt.close()
    

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
