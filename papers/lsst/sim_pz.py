"""
Simulates p(z) for lSST paper
"""

""" 
This script creates zdm grids for MeerTRAP
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

defaultsize=18
ds=4
font = {'family' : 'Helvetica',
        'weight' : 'normal',
        'size'   : defaultsize}
matplotlib.rc('font', **font)

#r-band limits 24.7, 27.5(single visit, 10 year, these are 5 sigma limits)

def main(opdir="Data/"):
    
    meerkat_z,meerkat_mr = read_meerkat()
    
    Load=True
    repeaters=False
    
    Rlim0 = 19.8 # existing magnitude limits
    Rlim1 = 24.7
    Rlim2 = 27.5
    
    names=['CRAFT_CRACO_1300','MeerTRAPcoherent','SKA_mid']
    labels=["ASKAP CRACO", "MeerKAT/SKA-Mid","SKA-Mid"]
    prefixes=["CRACO","MeerTRAP","SKA_Mid"]
    linestyles = ["-","--",":"]
    imax=2 # because SKA and mid are so similar
    
    if not os.path.exists(opdir):
        os.mkdir(opdir)
    
    Rs,Rrmss,Rzvals,sbar,srms = process_rbands()
    
    plot_R(Rs,Rrmss,Rzvals,sbar,srms,opdir,Rlim1,Rlim2)
      
    if not Load:
        #gs,ss = get_surveys_grids(names,opdir,repeaters=True,Test=False)
        ss,gs = get_surveys_grids(names,opdir,repeaters=False,Test=False)
        
        plot_efficiencies(gs,ss,opdir,prefixes)
        
        plot_beams(ss,labels,opdir)
        # plots telescope efficiencies at z=0
        
        zvals = gs[0].zvals
    else:
        zvals = np.load(opdir+"zvals.npy")
    
    nz = zvals.size
    
    if not Load:
        ##### gets mr distribution for each z #####
        NR=400
        Rvals = np.linspace(0,40,NR+1) # 401 values - defining bin edges
        Rbars = (Rvals[:-1]+Rvals[1:])/2.
        Rhist = np.zeros([nz,NR])
        fz0 = np.zeros([nz])
        fz1 = np.zeros([nz])
        fz2 = np.zeros([nz])
        iz0 = np.where(Rbars < Rlim0)[-1]
        iz1 = np.where(Rbars < Rlim1)[-1]
        iz2 = np.where(Rbars < Rlim2)[-1]
        
        for i,z in enumerate(zvals):
            if z < Rzvals[0]:
                continue
            elif z > Rzvals[-1]:
                continue
            else:
                Rbar = sbar(z)
                Rrms = srms(z)
                norm = stats.Normal(mu=Rbar,sigma=Rrms)
                vals = norm.cdf(Rvals)
                dv = vals[1:] - vals[:-1]
                
                # contains contributions from that redshift range
                Rhist[i,:] = dv
                
                # fractions up to that redshift
                fz0[i] = norm.cdf(Rlim0)
                fz1[i] = norm.cdf(Rlim1)
                fz2[i] = norm.cdf(Rlim2)
        np.save(opdir+"fz_19.8.npy",fz0)
        np.save(opdir+"fz_24.7.npy",fz1)
        np.save(opdir+"fz_27.5.npy",fz2)
        np.save(opdir+"Rhist.npy",Rhist)
        np.save(opdir+"Rvals.npy",Rvals)
        np.save(opdir+"Rbars.npy",Rbars)
    else:
        fz0 = np.load(opdir+"fz_19.8.npy")
        fz1 = np.load(opdir+"fz_24.7.npy")
        fz2 = np.load(opdir+"fz_27.5.npy")
        Rhist = np.load(opdir+"Rhist.npy")
        Rvals = np.load(opdir+"Rvals.npy")
        Rbars = np.load(opdir+"Rbars.npy")
    
    
    plt.figure()
    plt.xlabel("z")
    plt.ylabel("$f_{m_r}$")
    plt.plot(zvals,fz1,label="$f_{24.7}$")
    plt.plot(zvals,fz2,label="$f_{27.5}$")
    plt.legend()
    plt.tight_layout()
    plt.savefig(opdir+"fraction_visible.png")
    plt.close()
    
    DPplot(zvals,[fz1],["$m_r = 24.7$"],opdir + "DP_fraction_visible.png",color="orange")
    
    ####### p(z) plot #####
    plt.figure()
    plt.xlabel("z")
    plt.ylabel("N(z) [a.u.]")
    ax1 = plt.gca()
    
    ####### p(R) plot #####
    plt.figure()
    plt.xlabel("$m_r$")
    plt.ylabel("$p(m_r)$ [a.u.]")
    ax2 = plt.gca()
    
    for i,prefix in enumerate(prefixes):
        if i==imax:
            break
        if Load:
            pz = np.load(opdir+prefixes[i]+"_pz.npy")
            pz0 = np.load(opdir+prefixes[i]+"_fpz0.npy")
            pz1 = np.load(opdir+prefixes[i]+"_fpz1.npy")
            pz2 = np.load(opdir+prefixes[i]+"_fpz2.npy")
        else:
            s=ss[i]
            g=gs[i]
            ########## gets p(z) ##########
            if repeaters:
                # plots individual sources
                rates = (g.exact_singles+g.exact_reps)*g.Rc
            else:
                rates = g.rates*s.TOBS * 10**g.state.FRBdemo.lC
            pz = np.sum(rates,axis=1)
            dz = g.zvals[1]-g.zvals[0]
            
            # Norm is 7 days per week, with per redshift bin factored in
            norm = 7./dz
            pz *= norm
            
            pz0 = pz*fz0
            pz1 = pz*fz1
            pz2 = pz*fz2
            
            np.save(opdir+"zvals.npy",g.zvals)
            np.save(opdir+prefixes[i]+"_pz.npy",pz)
            np.save(opdir+prefixes[i]+"_fpz0.npy",pz0)
            np.save(opdir+prefixes[i]+"_fpz1.npy",pz1)
            np.save(opdir+prefixes[i]+"_fpz2.npy",pz2)
        
        # plots p(z)
        plt.sca(ax1)
        norm = np.max(pz)
        print(i,"Norm is ",norm)
        
        plt.plot(zvals,pz/norm,label=labels[i],linestyle="-")
        plt.plot(zvals,pz0/norm,linestyle="-.",color=plt.gca().lines[-1].get_color())
        plt.plot(zvals,pz1/norm,linestyle="--",color=plt.gca().lines[-1].get_color())
        plt.plot(zvals,pz2/norm,linestyle=":",color=plt.gca().lines[-1].get_color())
        
        print("For survey ",prefixes[i]," number of FRBs will be ",np.sum(pz),np.sum(pz0),np.sum(pz1),\
                np.sum(pz2),np.sum(pz0)/np.sum(pz),np.sum(pz1)/np.sum(pz),np.sum(pz2)/np.sum(pz))
        
        # plots total magnitude distribution
        pzR = (Rhist.T*pz).T
        mag_hist = np.sum(pzR,axis=0)
        
        np.save(opdir+prefixes[i]+"mag_hist.npy",mag_hist)
        
        plt.sca(ax2)
        norm = np.max(mag_hist)
        
        plt.plot(Rbars,mag_hist/norm,label=labels[i])
        
        
    plt.sca(ax1)
    plt.ylim(0,1.02)
    plt.xlim(0,6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(opdir+"lsst_pz.png")
    plt.close()
    
    plt.sca(ax2)
    plt.xlim(10,35)
    plt.ylim(0,1.02)
    
    #meerkat_z,meerkat_mr
    mrbins=np.linspace(10,30,21)
    weights = np.full([11],0.25)
    
    plt.hist(meerkat_mr,bins=mrbins,weights=weights,label="MK 2023 data")
    
    
    plt.plot([Rlim1,Rlim1],[0,1],linestyle=":",color="black")
    plt.plot([Rlim2,Rlim2],[0,1],linestyle=":",color="black")
    plt.text(Rlim1+0.1,0.1,"$m_r=$"+str(Rlim1),rotation=90)
    plt.text(Rlim2+0.1,0.1,"$m_r=$"+str(Rlim2),rotation=90)
    plt.legend()
    plt.tight_layout()
    plt.savefig(opdir+"lsst_pR.png")
    plt.close()
    
    DPplot(zvals,[pz,pz1],["all FRBs","LSST"],opdir + "DP_pz.png",color="orange",legend=False)
    DPplot(zvals,[pz,pz1,pz0],["all FRBs","LSST","Now"],opdir + "DP_pz0.png",color="orange",legend=False)
    
    DPplot(zvals,[pz1,pz0],["LSST","Now"],opdir + "DP_lsst_vs_now.png",color="orange",legend=False)

def DPplot(zvals,yvals,labels,outfile,color="orange",legend=True):
    
    fig = plt.figure()
    
    linestyles=["-","--",":","-."]
    # Plot in orange
    Norm=-1
    for i,yval in enumerate(yvals):
        norm = np.max(yval)
        if norm > Norm:
            Norm=norm
    
    for i,yval in enumerate(yvals):
        plt.plot(zvals, yval/Norm, label=labels[i], color=color,linestyle=linestyles[i])
    
    # Labels in orange
    plt.xlabel("redshift", color=color)
    plt.ylabel("fraction visible", color=color)
    
    # Axis limits
    plt.xlim(0, 3)
    plt.ylim(0, 1.05)
    
    # Make tick labels orange
    plt.tick_params(axis='both', colors=color)
    
    # Make the axes spines orange
    for spine in plt.gca().spines.values():
        spine.set_color(color)
    
    if legend:
        # Legend text + frame in orange
        leg = plt.legend()
        for text in leg.get_texts():
            text.set_color(color)
        leg.get_frame().set_edgecolor(color)
    
    fig.set_facecolor("none")
    plt.gca().set_facecolor("none")
    
    plt.tight_layout()
    plt.savefig(outfile,transparent=True)
    plt.close()

def plot_beams(ss,labels,opdir):
    # plots telescope beams
    plt.figure()
    for i,s in enumerate(ss):
        plt.scatter(s.beam_b,s.beam_o,label=labels[i])
    plt.xlabel("B")
    plt.ylabel("$\\Omega(B)$")
    plt.xscale("log")
    plt.yscale("log")
    plt.legend()
    plt.tight_layout()
    plt.savefig(opdir+"beams.png")
    plt.close()

def read_meerkat():
    """
    returns z and mr data from Pastor-Morales et al
    https://arxiv.org/pdf/2507.05982
    """
    
    data=np.loadtxt("meerkat_mr.txt",comments='#')
    z=data[:,1]
    mr = data[:,2]
    z = np.abs(z)
    return z,mr

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
    

def plot_efficiencies(gs,ss,opdir,prefixes):
    for i,s in enumerate(ss):
        plt.figure()
        g=gs[i]
        
        for j,w in enumerate(s.wlist):
            plt.plot(g.dmvals,s.efficiencies[j,0,:],label="w="+str(w)[0:5]) # at z=0
        plt.xlabel("DM")
        plt.ylabel("$\\epsilon$")
        plt.yscale("log")
        plt.ylim(0.1,2)
        plt.legend(fontsize=8,loc="upper right")
        plt.tight_layout()
        plt.savefig(opdir+prefixes[i]+"_efficiencies.png")
        plt.close()

def get_surveys_grids(names,opdir,repeaters=True,Test=False):

    # approximate best-fit values from recent analysis
    # load states from Hoffman et al 2025
    # use b or d for rep
    state = states.load_state("HoffmannEmin25",scat="updated",rep='b')
    
    
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
        survey_dict["WMETHOD"] = 2
    else:
        ndm=1400
        nz=600
        dmmax=7000
        zmax=6
        # uses redshift-dependent scattering. This takes longer
        # - by a factor of a few!
        survey_dict["WMETHOD"] = 3
    
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
