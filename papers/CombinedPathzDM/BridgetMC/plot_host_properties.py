"""
This file generates plots based on the generated FRB host properties
"""

import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
from frb.frb import FRB

from zdm import iteration as it
from zdm import loading
from zdm import optical as opt
from zdm import optical_params as op
from zdm import states

defaultsize=14
ds=4
font = {'family' : 'Helvetica',
        'weight' : 'normal',
        'size'   : defaultsize}
matplotlib.rc('font', **font)

def main():
    """
    
    """
    opdir="Hosts/"
    if not os.path.exists(opdir):
        os.mkdir(opdir)
        
    frbs = pd.read_csv("craco_900_mc_sample.csv")
    hosts = pd.read_csv("craco_assigned_galaxies.csv")
    
    plot_host_properties(frbs,hosts,opdir)
    
    # NOTE: the maghist is only generated using candidates from wide-field obs
    # otherwise, there are not enough stats on field galaxies
    hosts = pd.read_csv("loc30_craco_assigned_galaxies.csv")
    
    # do this for standard generation. Will have less field galaxies.
    get_true_hosts(frbs,hosts,opdir,width=10)
    
    # do this for wide loc30 survey with 2'x2'. Will have correct field galaxies.
    opdir= "loc30Hosts/"
    if not os.path.exists(opdir):
        os.mkdir(opdir)
    get_true_hosts(frbs,hosts,opdir,frbdir="loc30FRBFiles/",indir="loc30CandidateFiles/",width=120)
    
    
    
    
def get_xy_offsets(ra1,dec1,ra2,dec2):
    """
    Converts coordinates into x and y offstes
    """
    torad = np.pi/180.
    ra1 *= torad
    ra2 *= torad
    dec1 *= torad
    dec2 *= torad
    
    # converts coordinates to x,y,z
    x1 = np.cos(ra1) * np.cos(dec1)
    y1 = np.sin(ra1) * np.cos(dec1)
    z1 = np.sin(dec1)
    
    x2 = np.cos(ra2) * np.cos(dec2)
    y2 = np.sin(ra2) * np.cos(dec2)
    z2 = np.sin(dec2)
    
    # constructs EW and North local vectors
    
    # gets a phi angle for this offset relative to North
    # calculate "north" via cross product vector in xy plane
    x3 = -np.sin(ra1)
    y3 = np.cos(ra1)
    z3 = 0.
    
    # should be North. I hope!
    x4 = y2*z3 - y3*z2
    y4 = z2*x3 - z3*x2
    z4 = x2*y3 - x3*y2
    
    # gets vector from one to the other
    x5 = x2-x1
    y5 = y2-y1
    z5 = z2-z1
    
    # projects vector onto N and EW coordinates
    north = x3*x5 + y3*y5 + z3*z5
    ew = x4*x5 + y4*y5 + z4*z5
    
    return ew,north
    
def get_true_hosts(frbs,hosts,opdir,indir="CandidateFiles/",frbdir="FRBFiles/",NMAX=1000,width=10):
    """
    Runs through candidate files, getting which FRBs have a
    host, and which do not
    
    args:
        width: image width is arcseconds
    """
    
    NFRB = len(frbs)
    if NMAX > NFRB:
        NMAX = NFRB
    Nhosts = len(hosts)
    zfrb = np.zeros([NFRB])
    mlist = []
    notmlist = []
    hmlist = []
    Ncands = []
    
    dxlist = []
    dylist = []
    hdxlist = []
    hdylist = []
    sizelist = []
    hsizelist = []
    
    ####### Writes out candidate files ######
    for i in np.arange(NMAX):
        fname = indir+'FRB'+str(i)+"_PATH.csv"
        
        if not os.path.exists(fname):
            zfrb[i] = -1
            continue
        
        cands = pd.read_csv(fname)
        
        # load FRB properties
        frbfile = frbdir+"FRB"+str(i)+".json"
        frbopt = FRB.from_json(frbfile)
        
        # searches for FRB with known z
        zmatch = np.where(cands["z"] > -1)[0]
        if len(zmatch) == 0:
            zfrb[i] = -1
        else:
            zmatch = zmatch[0]
            zfrb[i] = cands["z"][zmatch]
        
        ncands = len(cands["mag"])
        Ncands.append(ncands)
        for j,mag in enumerate(cands["mag"]):
            mlist.append(mag)
            if cands["z"][j] == -1:
                notmlist.append(mag)
                dx,dy = get_xy_offsets(frbopt.coord.ra.deg,frbopt.coord.dec.deg,cands["ra"][j],cands["dec"][j])
                dxlist.append(dx)
                dylist.append(dy)
                sizelist.append(cands["ang_size"][j])
            else:
                hmlist.append(mag)
                dx,dy = get_xy_offsets(frbopt.coord.ra.deg,frbopt.coord.dec.deg,cands["ra"][j],cands["dec"][j])
                hdxlist.append(dx)
                hdylist.append(dy)
                hsizelist.append(cands["ang_size"][j])
            
    # we can now looks at the properties of FRBs with
    # and without the true host
    mlist = np.array(mlist)
    notmlist = np.array(notmlist)
    hmlist = np.array(hmlist)
    dxlist = np.array(dxlist)*3600*180/np.pi
    dylist = np.array(dylist)*3600*180/np.pi
    hdxlist = np.array(hdxlist)*3600*180/np.pi
    hdylist = np.array(hdylist)*3600*180/np.pi
    sizelist = np.array(sizelist)
    hsizelist = np.array(hsizelist)
    
    
    MISSING = np.where(zfrb == -1)[0]
    NMISSING = len(MISSING)
    print("Missing hosts for ",NMISSING," frbs")
    print("Number of field galaxies ",len(notmlist))
    print("Number of true hosts: ",len(hmlist))
    print("Total number of candidates is ",np.sum(Ncands))
    # histogram of redshifts for FRBs with known and unknown hosts
    
    
    
    ########## Makes a figure showing offset positions of field galaxies #########
    
    plt.figure()
    plt.scatter(dxlist[::100],dylist[::100],s=1)
    plt.xlabel("EW offset [arcsec]")
    plt.ylabel("NS offset [arcsec]")
    plt.tight_layout()
    plt.savefig(opdir+"field_xyscat.png")
    plt.close()
    
    plt.figure()
    plt.scatter(hdxlist,hdylist,s=1)
    plt.xlabel("EW offset [arcsec]")
    plt.ylabel("NS offset [arcsec]")
    plt.tight_layout()
    plt.savefig(opdir+"host_xyscat.png")
    plt.close()
    
    
    ########## Makes a figure giving redshift distribution of observed hosts #########
    
    bins = np.linspace(0,2,21)
    plt.figure()
    plt.hist(frbs["z"][:NMAX],label="True redshift",bins=bins,histtype='step')
    plt.hist(zfrb,label="Observed hosts",bins=bins,histtype='step',linestyle="--")
    plt.xlabel("FRB redshift, z")
    plt.ylabel("Counts")
    plt.legend()
    plt.yscale("log")
    plt.tight_layout()
    plt.savefig(opdir+"zhist.png")
    plt.close()
    
    
    # histogram of magnitudes as above
    bins = np.linspace(10,30,21)
    
    
    ########## Makes a figure giving magnitude histograms of hosts #########
    
    from astropath.chance import differential_driver_sigma as rhog
    from astropath.chance import driver_sigma as sigma
    
    # cumulative distributions
    densities = sigma(bins)
    # images are circular with radius width in arcsec
    expectation = np.pi*width**2*(densities[1:] - densities[:-1])*NMAX # 1e4 is number of galaxies, 100 is arcsec^2
    values = 0.5*(bins[1:] + bins[:-1])
    
    plt.figure()
    plt.hist(frbs["m_r"][:NMAX],label="All hosts",histtype='step',ls="-",bins=bins,lw=2)
    plt.hist(hmlist,label="Observed hosts",histtype='step',ls="--",bins=bins,lw=2)
    plt.hist(notmlist,label="Field galaxies",histtype='step',ls=":",bins=bins,lw=2)#,weights=hweight)
    plt.hist(values,label="Expectation",histtype='step',ls=":",bins=bins,lw=2,weights=expectation)
    plt.ylim(0.1,1e5)
    
    plt.legend(loc="upper left")
    plt.yscale("log")
    plt.xlabel("$m_r$")
    plt.ylabel("$N(m_r)$")
    plt.tight_layout()
    plt.savefig(opdir+"maghist.png")
    plt.close()
    
    bcs = (bins[:-1]+bins[1:])/2.
    OK = np.where(bcs < 22.)[0]
    
    bins = bins[:OK[-1]+2]
    bcs = bcs[OK]
    expectation = expectation[OK]
    
    
    OK = np.where(bcs > 14.)[0]
    bins = bins[OK[0]:]
    bcs = bcs[OK[0]:]
    expectation = expectation[OK[0]:]
    
    
    # factor of 36 from using the wide files, i.e. radius of 1 arcmin instead of 10 "
    
    # fit ratio of expectations
    plt.figure()
    obshist,bins = np.histogram(notmlist,bins=bins)
    obshist2,bins = np.histogram(mlist,bins=bins)
    #exphist,bins = np.histogram(values,bins=bins)
    
    fit = np.polyfit(bcs,obshist/expectation,deg=2)
    
    from scipy.optimize import curve_fit as cf
    p0=[1.,1.,16.]
    
    cffit = cf(fit2f,bcs,obshist/expectation,p0=p0)
    cffit2 = cf(fit2f,bcs,obshist2/expectation,p0=p0)
    print("Fit to field galaxies is ",cffit)
    print(cffit2)
    #thefit = np.polyval(fit,bcs)
    fit = cffit[0][0]+cffit[0][1]*np.exp(cffit[0][2]-bcs)
    fit2= cffit2[0][0]+cffit2[0][1]*np.exp(cffit2[0][2]-bcs)
    #plt.plot(bcs,thefit,label="fit")
    plt.plot(bcs,obshist/expectation,label="ratio field galaxies")
    plt.plot(bcs,obshist2/expectation,label="ratio all galaxies")
    plt.plot(bcs,fit,label="fit")
    plt.plot(bcs,fit2,label="fit")
    plt.xlabel("mag")
    plt.ylabel("excess")
    plt.legend()
    plt.tight_layout()
    plt.savefig(opdir+"excess.png")
    plt.close()
    
    print("Minimum and maximum magnitudes are ",np.min(hmlist),np.max(hmlist))
    
def fit2f(x,*args):
    a = args[0]
    b = args[1]
    c = args[2]
    return a + b*np.exp(c-x)
        
def plot_host_properties(frbs,hosts,opdir):
    """
    Makes plots comparing host and assigned host magnitudes
    """
    
    print("FRB keys are ",frbs.keys())
    
    print("Host keys are ",hosts.keys())
    
    m1 = frbs["m_r"][hosts["FRB_ID"]]    
    
    print("Number of assigned hosts is ",len(m1))
    
    plt.figure()
    plt.scatter(m1,hosts["mag"])
    plt.xlabel("Simulated host magnitude")
    plt.ylabel("Assigned catalogue host magnitude")
    plt.tight_layout()
    plt.savefig(opdir+"host_assigned_scatter.png")
    plt.close()
    
    plt.figure()
    plt.scatter(m1,hosts["mag"])
    plt.xlabel("Simulated host magnitude")
    plt.ylabel("Assigned catalogue host magnitude")
    plt.tight_layout()
    plt.savefig(opdir+"host_assigned_scatter.png")
    plt.close()
    
    # calculates a moving average
    bins = np.linspace(10,30,21)
    bbar = np.linspace(10.5,29.5,20)
    h1,b = np.histogram(hosts["mag"],weights=hosts["half_light"],bins=bins)
    h2, b = np.histogram(hosts["mag"],bins=bins)
    hlbar = h1/h2
    
    for i,m in enumerate(bbar):
        print(m,hlbar[i])
    
    plt.figure()
    plt.scatter(hosts["mag"],hosts["half_light"],s=1)
    plt.scatter(bbar,hlbar,s=30,marker="+")
    plt.xlabel("Simulated host magnitude")
    plt.ylabel("Half-light radius [arcsec]")
    plt.tight_layout()
    plt.savefig(opdir+"mag_halflight.png")
    plt.close()
    
    HCTE = np.where(hosts["mag"] < 14.0)[0]
    hosts.loc[HCTE,"half_light"] = hosts["half_light"][HCTE]*20.
    plt.figure()
    plt.scatter(hosts["mag"],hosts["half_light"],s=1)
    hlbar[0:4] *= 20
    plt.scatter(bbar,hlbar,s=30,marker="+")
    plt.xlabel("Simulated host magnitude")
    plt.ylabel("Half-light radius [arcsec]")
    plt.tight_layout()
    plt.savefig(opdir+"mod_mag_halflight.png")
    plt.close()
    

main()
