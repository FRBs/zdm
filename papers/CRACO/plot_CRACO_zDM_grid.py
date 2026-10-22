""" 
This script creates zdm grids for CRACO observations.

It exists partly to calculate relative rates from surveys

It also calculates expected p(DM) distributions, and compare
these against the observed DMs

"""
import os

from astropy.cosmology import Planck18
from zdm import cosmology as cos
from zdm import figures
from zdm import iteration as it
from zdm import loading
from zdm import states
from zdm import misc_functions as mf

import pickle
from scipy.stats import ks_1samp
import numpy as np
from zdm import survey
from matplotlib import pyplot as plt
import importlib.resources as resources
import matplotlib

defaultsize=16

font = {'family' : 'Helvetica',
        'weight' : 'normal',
        'size'   : defaultsize}
matplotlib.rc('font', **font)

def main():
    """
    Main program.
    
    Loops through the different CRACO setups, creating
    zDM plots, and cumulative plots
    
    Eventually, it creates an all-CRACO plot
    
    """
    # in case you wish to switch to another output directory
    name="zDMPlots"
    opdir=name+"/"
    if not os.path.exists(opdir):
        os.mkdir(opdir)
    prefix=""
    # main result
    state = states.load_state("HoffmannRepeaters26Pn",scat="updated",rep=None) #latest
    tag="RepPn"
    
    # choose this one to estimate the effect when using old acattering and width
    #state = states.load_state("HoffmannRepeaters26Pn",rep=None) #latest
    #tag="RepPnoldscat"
    
    tag = prefix+tag
    
    # set limits for plots - will be LARGE!   
    DMmax=3000
    zmax=3.0
    
    # Initialise surveys and grids
    sdir = resources.files('zdm').joinpath('../papers/CRACO/Surveys/')
    itsamps = [2,4,8,16,64]
    
    allrate = None
    alldmlist = []
    nozlist = []
    zdmlist = []
    zzlist = []
    Nexp = []
    NFRBs = []
    craconames=[]
    ksps=[]
    
    ##### iterates over all configurations, estimating relative numbers of FRBs, for CRACO #######
    for itsamp in itsamps:
        for ifreq,freq in enumerate(["900","1300"]):
            survey=prefix+"CRACO_"+freq+"_itsamp_"+str(itsamp)
            craconames.append(survey)
            zdmplotfile = "Plots/"+tag+"zdm_"+freq+"_itsamp_"+str(itsamp)+".png"
            dmplotfile = "Plots/"+tag+"cum_dm_"+freq+"_itsamp_"+str(itsamp)+".png"
            pklfile = "Pickle/"+tag+survey+".pkl"
            pklfile=None
            rate,zvals,dmvals,dmlist,noz,z_dms,z_zs,NFRB,ksp = predict_zdm(survey,sdir,state,zdmplotfile,
                                                                    dmplotfile,pklfile,tag,
                                                                    zmax=zmax,dmmax=DMmax,new=False)
            ksps.append(ksp)
            Nexp.append(np.sum(rate))
            NFRBs.append(NFRB)
            alldmlist = alldmlist+dmlist
            nozlist = nozlist+noz
            zdmlist = zdmlist+z_dms
            zzlist = zzlist+z_zs
            if allrate is None:
                allrate = rate
            else:
                allrate = allrate+rate
            
    ##### generates a total cumulative DM plot #####
    craco_ksp = cumulative_dm(allrate,dmvals,alldmlist,"Plots/"+tag+"cum_dm_summed_craco.png")
    
    print("KSP value for all of craco is ",craco_ksp)
    
    ksps = np.array(ksps)
    alldmlist = np.array(alldmlist)
    nozlist = np.array(nozlist)
    zdmlist = np.array(zdmlist)
    zzlist = np.array(zzlist)
    
    figures.plot_grid(allrate,zvals,dmvals,
            name="Plots/"+tag+"zdm_summed_craco.png",norm=3,log=True,
            label='$\\log_{10} p({\\rm DM}_{\\rm cosmic} + {\\rm DM}_{\\rm host},z)$ [a.u.]',
            project=False,ylabel='${\\rm DM}_{\\rm cosmic} + {\\rm DM}_{\\rm host}$',
            zmax=zmax,DMmax=DMmax,Aconts=[0.01,0.1,0.5],
            FRBDMs=zdmlist,FRBZs=zzlist,
            DMlines = nozlist
            )
    # record this for later
    NEXPcraco = Nexp
    NOBScraco = NFRBs
    KSPcraco = ksps
    
    ##### iterates over all configurations, estimating relative numbers of FRBs, for other CRAFT #######
    
    allrate = None
    alldmlist = []
    nozlist = []
    zdmlist = []
    zzlist = []
    Nexp = []
    NFRBs = []
    ksps=[]
    
    sdir = None
    craftnames=['CRAFT_class_I_and_II','CRAFT_ICS_892','CRAFT_ICS_1300','CRAFT_ICS_1632']
    for name in craftnames:
        zdmplotfile = "Plots/"+tag+name+".png"
        dmplotfile = "Plots/"+tag+name+".png"
        pklfile = "Pickle/"+tag+name+".pkl"
        pklfile=None
        rate,zvals,dmvals,dmlist,noz,z_dms,z_zs,NFRB,ksp = predict_zdm(name,sdir,state,zdmplotfile,
                                                                    dmplotfile,pklfile,tag,
                                                                    zmax=zmax,dmmax=DMmax,new=False)
        ksps.append(ksp)
        Nexp.append(np.sum(rate))
        NFRBs.append(NFRB)
        alldmlist = alldmlist+dmlist
        nozlist = nozlist+noz
        zdmlist = zdmlist+z_dms
        zzlist = zzlist+z_zs
        if allrate is None:
            allrate = rate
        else:
            allrate = allrate+rate
    
    
    ##### generates a total cumulative DM plot #####
    craft_ksp = cumulative_dm(allrate,dmvals,alldmlist,"Plots/"+tag+"cum_dm_summed_prev_craft.png")
    
    print("KSP value for all of craft is ",craft_ksp)
    
    ksps = np.array(ksps)
    alldmlist = np.array(alldmlist)
    nozlist = np.array(nozlist)
    zdmlist = np.array(zdmlist)
    zzlist = np.array(zzlist)
    
    figures.plot_grid(allrate,zvals,dmvals,
            name="Plots/"+tag+"zdm_summed_prev_craft.png",norm=3,log=True,
            label='$\\log_{10} p({\\rm DM}_{\\rm cosmic} + {\\rm DM}_{\\rm host},z)$ [a.u.]',
            project=False,ylabel='${\\rm DM}_{\\rm cosmic} + {\\rm DM}_{\\rm host}$',
            zmax=zmax,DMmax=DMmax,Aconts=[0.01,0.1,0.5],
            FRBDMs=zdmlist,FRBZs=zzlist,
            DMlines = nozlist
            )
    # record this for later
    NEXPcraft = Nexp
    NOBScraft = NFRBs
    KSPcraft = ksps
    
    ##### prints ratio compared to CRAFT_class_I_and_II ######
    NORM_RATE = NEXPcraft[1]+NEXPcraft[2]+NEXPcraft[3]
    NORM_FRB = NOBScraft[1]+NOBScraft[2]+NOBScraft[3]
    
    
    print("########## NORMALISING TO all ICS ICS #############")
    for i,name in enumerate(craftnames):
        print("Expected number of FRBs for ",name," is ",NEXPcraft[i]/NORM_RATE*NORM_FRB,
                    " actually detected ",NOBScraft[i]," KSP is ",KSPcraft[i])
    sumE1=0.
    sumE2=0.
    sumO1=0.
    sumO2=0.
    for i,name in enumerate(craconames):
        print("Expected number of FRBs for ",name," is ",NEXPcraco[i]/NORM_RATE*NORM_FRB,
                    " actually detected ",NOBScraco[i]," KSP is ",KSPcraco[i])
        if i%2==0:
            sumE1 += NEXPcraco[i]/NORM_RATE*NORM_FRB
            sumO1 += NOBScraco[i]
        else:
            sumE2 += NEXPcraco[i]/NORM_RATE*NORM_FRB
            sumO2 += NOBScraco[i]
    
    print("Total: 900 Mhz: ",sumE1," cf obs of ",sumO1)
    print("       1300 Mhz: ",sumE2," cf obs of ",sumO2)
    print("       Combined ",sumE1+sumE2," cf obs of ",sumO1+sumO2)
    print("\n\n\n\n")
    
    
    
    #NORM_RATE = NEXPcraft[2]
    #NORM_FRB = NOBScraft[2]
    #
    #print("\n\n\n########## NORMALISING TO 1.3 GHzICS #############")
    #for i,name in enumerate(craftnames):
    #    print("Expected number of FRBs for ",name," is ",NEXPcraft[i]/NORM_RATE*NORM_FRB,
    #                " actually detected ",NOBScraft[i]," KSP is ",KSPcraft[i])
    #for i,name in enumerate(craconames):
    #    print("Expected number of FRBs for ",name," is ",NEXPcraco[i]/NORM_RATE*NORM_FRB,
    #                " actually detected ",NOBScraco[i]," KSP is ",KSPcraco[i])
    
    
def predict_zdm(survey,sdir,state,zdmplotfile,dmplotfile,pklname,tag,new=False,
                zmax=3.,dmmax=3000,nz=300,ndm=300):
    """
    Creates zDM distribution
    """
    
    if pklname is not None and os.path.exists(pklname) and not new:
        with open(pklname, 'rb') as pklfile:
            ss = pickle.load(pklfile)
            gs = pickle.load(pklfile)
    else:
        ss,gs = loading.surveys_and_grids(survey_names=[survey],repeaters=False,
                                        init_state=state,sdir=sdir,
                                        zmax=zmax,nz=nz,dmmax=dmmax,ndm=ndm
                                        ) # should be equal to actual number of FRBs, but for this purpose it doesn't matter
        if pklname is not None:
            with open(pklname, 'ab') as pklfile:
                pickle.dump(ss, pklfile)
                pickle.dump(gs, pklfile)
    
    plt.figure()
    ax1 = plt.gca()
    
    plt.figure()
    ax2 = plt.gca()
    
    
    # chooses the first arbitrarily to extract zvals etc from
    s=ss[0]
    g=gs[0]
        
    noz=s.nozlist
    z=s.zlist
    
    DMEGs = s.DMEGs
        
    if noz is not None:
        DMlines = s.DMEGs[noz]
    else:
        DMlines=None
    
    if z is not None:
        FRBDMs = s.DMEGs[z]
        FRBZs=s.Zs[z]
    else:
        FRBDMs=None
        FRBZs=None
    
    
    figures.plot_grid(gs[0].get_rates(),g.zvals,g.dmvals,
            name=zdmplotfile,norm=3,log=True,
            label='$\\log_{10} p({\\rm DM}_{\\rm cosmic} + {\\rm DM}_{\\rm host},z)$ [a.u.]',
            project=False,ylabel='${\\rm DM}_{\\rm cosmic} + {\\rm DM}_{\\rm host}$',
            zmax=zmax,DMmax=dmmax,Aconts=[0.01,0.1,0.5],
            FRBDMs=FRBDMs,FRBZs=FRBZs,
            DMlines = DMlines
            )
    
    if FRBDMs is None:
        FRBDMs = []
        FRBZs = []
    else:
        FRBDMs = FRBDMs.tolist()
        FRBZs = FRBZs.tolist()
    
    if len(DMEGs) > 0:
        DMEGs = DMEGs.tolist()
    if DMlines is None:
        DMlines = []
    else:
        DMlines = DMlines.tolist()
    
    # this is expected relative rate
    rate = gs[0].get_rates() * s.TOBS
    
    #generates cumulative rate
    ksp = cumulative_dm(rate,gs[0].dmvals,DMEGs,dmplotfile)
    
    NFRB = s.NORM_FRB
    
    #rate = np.sum(rate) # * 10**g.state.FRBdemo.lC
    return rate,gs[0].zvals,gs[0].dmvals,DMEGs,DMlines,FRBDMs,FRBZs,NFRB,ksp
    

def cdf(x,dm,cs):
    """
    Function to return a cdf given dm and cs via linear interpolation
    """
    nx = np.array(x)
    #y=np.zeros(nx.size)
    #y[x <= dm[0]]=0.
    #y[x >= dm[-1])=1.
    
    ddm = dm[1]-dm[0]
    ix1 = (x/ddm).astype('int')
    ix2 = ix1+1
    
    kx2 = x/ddm-ix1
    kx1 = 1.-kx2
    c = cs[ix1]*kx1 + cs[ix2]*kx2
    return c

def cumulative_dm(rate,dms,dmlist,dmplotfile):
    """
    Creates a cumulative expected DM plot
    
    Does a KS test of result
    
    Args:
        rate: 2D zDM grid of rates
        dms: 1D np array of DMEGs
        dmlist: list of DMs of observed FRBs
    """
    
    # sums over z distribution
    pdm = np.sum(rate,axis=0)
    cpdm = np.cumsum(pdm)
    cpdm /= cpdm[-1]
    
    plt.figure()
    
    if len(dmlist) > 0:
        # make cumulative distribution for the data
        x,y = mf.make_cum_dist(dmlist)
        plt.plot(x,y,label="FRBs")
    
    plt.plot(dms,cpdm,label="Prediction")
    plt.xlabel("DM$_{\\rm EG}$ [pc/cm$^{-3}$")
    plt.ylabel("Cumulative probability")
    
    plt.ylim(0,1)
    plt.xlim(0,2000)
    plt.legend()
    plt.tight_layout()
    plt.savefig(dmplotfile)
    plt.close()
    
    if len(dmlist) > 0:
        # what we should be doing
        corder = np.sort(dmlist)
        kstat=ks_1samp(corder,cdf,args=(dms,cpdm),alternative='two-sided',mode='exact')
        ksp = kstat[1]
    else:
        ksp = 1. # always fine if no data!
    return ksp
    
main()
