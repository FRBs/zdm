"""

This script creates a plot of p(z,m) given the DM of a particular FRB, with host unseen
in LSST.
                                                                                                                                                         eerTRAPcoherent']
"""
import os

from astropy.cosmology import Planck18
from zdm import cosmology as cos
from zdm import figures
from zdm import parameters
from zdm import survey
from zdm import pcosmic
from zdm import iteration as it
from zdm import loading
from zdm import io
from zdm import optical as opt
from zdm import states
from zdm import optical
import numpy as np
from zdm import survey
from matplotlib import pyplot as plt
import importlib.resources as resources
from scipy.interpolate import CubicSpline
from scipy import stats
import matplotlib
import importlib.resources as resources


defaultsize=18
ds=4
font = {'family' : 'Helvetica',
        'weight' : 'normal',
        'size'   : defaultsize}
matplotlib.rc('font', **font)

#r-band limits 24.7, 27.5(single visit, 10 year, these are 5 sigma limits)

def main():
    
    plotdir="Plots/"
    opdir="Data/"
    optdir = str(resources.files('zdm').joinpath('data/optical'))+"/"
    
    meerkat_z,meerkat_mr,meerkat_w = read_meerkat()
    
    # we should re-do this shortly.
    Load=False
    repeaters=False
    Test=False # do this for very simplified data
    Scat=False # do not use updated scattering model
    
    Rlim0 = 19.8 # existing magnitude limits
    Rlim1 = 24.7
    Rlim2 = 27.5
    Rlim3 = 23.0 #decals
    
    
    # do this only for CRACO
    names=['CRAFT_CRACO_1300']
    labels=["ASKAP CRACO"]
    prefixes=["CRACO"]
    
    # DM = 1300
    
    linestyles = ["-","--",":"]
    imax=2 # because SKA and mid are so similar
    
    if not os.path.exists(plotdir):
        os.mkdir(plotdir)
    if not os.path.exists(opdir):
        os.mkdir(opdir)
        
    # get the rms and means etc as a function of redshift
    Rs,Rrmss,Rzvals,sbar,srms = process_rbands()
    
    # get the survey properties
    ss,gs = get_surveys_grids(names,opdir,repeaters=repeaters,Test=Test,Scat=Scat)
    g = gs[0]
    zvals = g.zvals
    dmvals = g.dmvals
    
    nz = zvals.size
    
    #fz0 = np.load(optdir+"fz_19.8.npy")
    #fz1 = np.load(optdir+"fz_24.7.npy")
    #fz2 = np.load(optdir+"fz_27.5.npy")
    #Rhist = np.load(opdir+"Rhist.npy")
    #Rvals = np.load(opdir+"Rvals.npy")
    #Rbars = np.load(opdir+"Rbars.npy")
    
    # we now extract the p(z|DM) for each slice in DM
    
    DM = 1300
    nm=101
    mvals = np.linspace(20,30,nm) # mag values every 0.1
    mcut = 24.7
    mlow = np.where(mvals < mcut)[0]
    mhigh = np.where(mvals > mcut)[0]
    dmval = (mvals[1]-mvals[0])/2.
    dzval = (zvals[1]-zvals[0])/2.
    mus = sbar(zvals)
    msigmas = srms(zvals)
    nz = zvals.size
    
    mzgrid = np.zeros([nz,nm])
    iDM = np.where(dmvals >= DM)[0][0]
    
    print("Using DM value ",dmvals[iDM])
    pz = g.rates[:,iDM]
    pz /= np.sum(pz) * dzval
    # for each z, construct a grid on mag values
    for j,z in enumerate(zvals):
        pm = gauss(mvals,mus[j],msigmas[j]) # p(mr |z)
        pm /= np.sum(pm) * dmval # normalise
        pm[mlow] = 0. # set region which would have been visible to zero
        mzgrid[j,:] = pz[j] * pm
    
    
    aspect = len(np.where(mvals >= 24.0)[0])/len(np.where(zvals <= 2.0)[0])
    
    plt.figure()
    plt.imshow(mzgrid.T,origin="lower",extent=[zvals[0]-dzval/2.,zvals[-1]+dzval/2.,mvals[0]-dmval/2.,mvals[-1]+dmval/2.],
                aspect=aspect)
    cb = plt.colorbar(label="$p(z,m_r | {\\rm DM_{\\rm EG}=}"+str(DM)+" {\\rm pc}\,{\\rm cm}^{-3})$")
    plt.xlim(0,2)
    plt.ylim(24,30)
    plt.xlabel("$z$")
    plt.ylabel("$m_r$")
    #cb.set_clim(-2.0, 2.0)
    plt.tight_layout()
    plt.savefig(plotdir+"pmz_1300.png")
    plt.close()
      
def gauss(x,mu,sigma):
    """
    Gaussian distribution
    """
    return np.exp(-0.5 * (x-mu)**2/sigma**2) / (2. * np.pi * sigma**0.5)

def read_meerkat():
    """
    returns z and mr data from Pastor-Morales et al
    https://arxiv.org/pdf/2507.05982
    Detection method provided in private communication (Pastor-Morales)
    """
    
    data=np.loadtxt("Data/meerkat_mr.txt",comments='#')
    z=data[:,2]
    mr = data[:,3]
    loc = data[:,4] # 1 is coherent beam, 0 incoherent only
    z = np.abs(z) # -ve is
    w = data[:,5] #PO|x
    
    # removes incoherent sum data
    good = np.where(loc==1)[0]
    z=z[good]
    loc=loc[good]
    mr=mr[good]
    w = w[good]
    
    # removes missing data
    good = np.where(z != 9999)
    z = z[good]
    loc=loc[good]
    mr=mr[good]
    w=w[good]
    
    return z,mr,w

def plot_R(Rbars,Rrmss,Rzvals,sbar,srms,opdir,Rlim1,Rlim2):
    # plot of mean and rms from Gaussian assumption
    plt.figure()
    plt.xlabel("z")
    plt.ylabel("$m_r$")
    plt.plot(Rzvals,Rbars,label="$\\mu_r$")
    plt.plot(Rzvals,Rbars+Rrmss,linestyle="--",label="$\\mu_r \\pm \\sigma_r$")
    plt.plot(Rzvals,Rbars-Rrmss,linestyle="--",color=plt.gca().lines[-1].get_color())
    plt.plot([0,6],[Rlim1,Rlim1],linestyle=":",color="black")
    plt.plot([0,6],[Rlim2,Rlim2],linestyle=":",color="black")
    
    plt.text(3,Rlim1-1.5,"$m_r=$"+str(Rlim1))
    plt.text(0.01,Rlim2+0.2,"$m_r=$"+str(Rlim2))
    plt.legend()
    plt.xlim(0,6)
    plt.tight_layout()
    plt.savefig(opdir+"Rbar_rms_z.png")
    plt.close()
    

def plot_efficiencies(gs,ss,opdir,prefixes,Test=False,Scat=False):
    """
    Generates a plot of efficiencies at the 0th zbin. Or, for all zbins,
    if we are doing a test
    """
    
    for i,s in enumerate(ss):
        plt.figure()
        g=gs[i]
        
        for j,w in enumerate(s.wlist):
            if Scat:
                plt.plot(g.dmvals,s.efficiencies[j,0,:],label="w="+str(w)[0:5]) # at z=0
            else:
                plt.plot(g.dmvals,s.efficiencies[j,:],label="w="+str(w)[0:5])
        plt.xlabel("DM")
        plt.ylabel("$\\epsilon$")
        plt.yscale("log")
        plt.ylim(0.1,2)
        plt.legend(fontsize=8,loc="upper right")
        plt.tight_layout()
        plt.savefig(opdir+prefixes[i]+"_efficiencies.png")
        plt.close()

def get_surveys_grids(names,opdir,repeaters=True,Test=False,Scat=False):

    # approximate best-fit values from recent analysis
    # load states from Hoffman et al 2025
    # use b or d for rep
    
    if Scat:
        state = states.load_state("HoffmannHalo25",scat="updated",rep='b')
    else:
        state = states.load_state("HoffmannHalo25",rep='b')
    
    # artificially add repeater data - we can't actually know this,
    # because we don't have time per field. Just using one day for now
    survey_dict={}
    survey_dict["TFIELD"] = 24.
    survey_dict["TOBS"] = 24.
    survey_dict["NFIELDS"] = 1
    
    survey_dict["NORM_REPS"] = 0
    survey_dict["NORM_SINGLES"] = 0
    survey_dict["NORM_FRB"] = 0
    
    survey_dict["NBINS"] = 10
    survey_dict["BTHRESH"] = 0.01
    
    
    
    if not os.path.exists(opdir):
        os.mkdir(opdir)
    
    # Initialise surveys and grids
    sdir = resources.files('zdm').joinpath('data/Surveys')
    #names=['SKA_mid']
    
    
    # simple vs complex
    if Test:
        ndm=50
        nz=50
        dmmax=4000
        zmax=4
        
    else:
        ndm=1400
        nz=600
        dmmax=7000
        zmax=6
        # uses redshift-dependent scattering. This takes longer
        # - by a factor of a few!
        #survey_dict["Wmethod"] = 3
    
    if Scat:
        survey_dict["Wmethod"] = 3
    else:
        survey_dict["Wmethod"] = 2
    ss,gs = loading.surveys_and_grids(survey_names=names,repeaters=repeaters,init_state=state,
                                        sdir=sdir,survey_dict=survey_dict,nz=nz,zmax=zmax,ndm=ndm,dmmax=dmmax)
    return ss,gs

def process_rbands():
    """
    Returns parameters of the host magnitude distribution as a function of redshift
    """
    #FRBlist=["FRB20180301A FRB20180916B FRB20190520B FRB20201124A FRB20210410D FRB20121102A FRB20180924B FRB20181112A FRB20190102C FRB20190608B FRB20190611B FRB20190711A FRB20190714A FRB20191001A FRB20200430A FRB20200906A FRB20210117A FRB20210320C FRB20210807D FRB20211127I FRB20211203C FRB20211212A FRB20220105A]
    table = optical.load_marnoch_data()
    colnames = table.colnames
    # gets FRBs
    frblist=[]
    for name in colnames:
        if name[0:3]=="FRB":
            frblist.append(name)
    zlist = table["z"]
    nz = zlist.size
    nfrb = len(frblist)
    Rmags = np.zeros([nfrb,nz])
    
    for i,frb in enumerate(frblist):
        
        Rmags[i,:] = table[frb]
    
    # gets mean and rms
    Rbar = np.average(Rmags,axis=0)
    Rrms = (np.sum((Rmags - Rbar)**2,axis=0)/(nfrb-1))**0.5
    
    sbar = CubicSpline(zlist,Rbar)
    srms = CubicSpline(zlist,Rrms)
    
    
    return Rbar,Rrms,zlist,sbar,srms
main()
