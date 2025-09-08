"""
This script creates a 3D distribution. For each values
of width w, it determines the relative contribution of 
each scattering and width value to that w.

This then allows FRB w and tau values to be fit directly,
rather than indirectly via a total effective width.

This is not a 5D problem (z,DM,w,scat,tau) because
p(w|z,DM) is independent of the tau and w that
contributed to it.

NOTE: all the calculations for p(w) assume distributions in
    log10 space, i.e. p(log10_w), and all bins are log-spaced
    bins, regardless of whether or not the calculations are
    performed in linear or log space.

"""

import os

from zdm import cosmology as cos
from zdm import figures
from zdm import parameters
from zdm import survey
from zdm import pcosmic
from zdm import misc_functions as mf
from zdm import iteration as it
from zdm import loading
from zdm import io
from pkg_resources import resource_filename
import numpy as np
from matplotlib import pyplot as plt

import matplotlib

defaultsize=14
ds=4
font = {'family' : 'Helvetica',
        'weight' : 'normal',
        'size'   : defaultsize}
matplotlib.rc('font', **font)


def main():
    """
    See main description at top of file.
    """
    
    # in case you wish to switch to another output directory
    opdir = "Plots/"
    if not os.path.exists(opdir):
        os.mkdir(opdir)
    
    # directory where the survey files are located. The below is the default - 
    # you can leave this out, or change it for a different survey file location.
    sdir = os.path.join(resource_filename('zdm', 'data'), 'Surveys')
    
    # make this into a list to initialise multiple surveys art once
    names = ["CRAFT_ICS_1300"] # for example
    
    repeaters=False
    # sets plotting limits
    zmax = 2.
    dmmax = 2000
    
    survey_dict = {"WMETHOD": 3} # 2 is no z-depence, 3 is
    
    state_dict = {}
    state_dict["scat"] = {}
    state_dict["scat"]["Sbackproject"] = True # turns on backprojection of tau and width for our model
    state_dict["width"] = {}
    state_dict["width"]["WNInternalBins"] = 100 # sets it to a small quantity
    state_dict["width"]["WNbins"] = 33 # set to large number for this analysis
    # We set these to smaller values in order to show more-
    # detectable observational effects
    
    wset = 2
    if wset == 1: # set for analytic fits to lognormal
        state_dict["width"]["WidthFunction"] = 1 # lognormal
        state_dict["width"]["ScatFunction"] = 1 # lognormal
        state_dict["width"]["Wlogmean"] = 0.215
        state_dict["width"]["Wlogsigma"] = 0.88
        state_dict["scat"]["Slogmean"] = 0.36
        state_dict["scat"]["Slogsigma"] = 1.05
    elif wset == 2: # set for analytic fits to halflognormal
        state_dict["width"]["WidthFunction"] = 2 # halflognormal
        state_dict["width"]["ScatFunction"] = 2 # halflognormal
        state_dict["width"]["Wlogmean"] = 0.29 # 2
        state_dict["width"]["Wlogsigma"] = 0.64 #2
        state_dict["scat"]["Slogmean"] = -1.3 #1
        state_dict["scat"]["Slogsigma"] = 0.01 #4
    elif wset==3: # set for MCMC fits to lognormal
        state_dict["width"]["WidthFunction"] = 1 # lognormal
        state_dict["width"]["ScatFunction"] = 1 # lognormal
        state_dict["width"]["Wlogmean"] = 2 # 2
        state_dict["width"]["Wlogsigma"] = 1.6 #2
        state_dict["scat"]["Slogmean"] = 1.4 #1
        state_dict["scat"]["Slogsigma"] = 4 #4
    elif wset==4: # set for MCMC fits to halflognormal
        state_dict["width"]["WidthFunction"] = 2 # halflognormal
        state_dict["width"]["ScatFunction"] = 2 # halflognormal
        state_dict["width"]["Wlogmean"] = 1 # 2
        state_dict["width"]["Wlogsigma"] = 1.5 #2
        state_dict["scat"]["Slogmean"] = 1.5 #1
        state_dict["scat"]["Slogsigma"] = 4 #4
    
    surveys, grids = loading.surveys_and_grids(survey_names = names,\
                        repeaters=repeaters, sdir=sdir,nz=70,ndm=140,
                        survey_dict = survey_dict, state_dict = state_dict)

    # gets log-likelihoods including tau,w
    s=surveys[0]
    g=grids[0]
    # this assumes we have both 1D and 2D components
    ll1 = it.calc_likelihoods_1D(g,s,Pn=True,pNreps=True,ptauw=True,dolist=0)
    ll2 = it.calc_likelihoods_2D(g,s,Pn=True,pNreps=True,psnr=True,ptauw=True,dolist=0)
    print("Calculated log-likelihoods including w,scat are ",ll1,ll2)
    
    
    ################ Generates some plots ################
    
    # extracts the p(W) distribution
    Rw,Rtau,Nw,Nwz,Nwdm = mf.get_width_stats(s,g)
    
    # values at z=0
    WidthArgs = (s.wlogmean,s.wlogsigma)
    ScatArgs = (s.slogmean,s.slogsigma)
    
    # these need to be normalised by the internal bin width
    logbinwidth = s.internal_logwvals[-1] - s.internal_logwvals[-2]
    
    pw = s.WidthFunction(s.internal_logwvals, *WidthArgs)*s.dlogw #/logbinwidth
    ptau = s.ScatFunction(s.internal_logwvals, *ScatArgs)*s.dlogw #/logbinwidth
    
    ws = s.wlist
    
    norm=True
    if norm:
        # We require that the following functions obey \int function dlogw = 1
        # hence, we need to divide by total sum, then divide by bin
        # width in units of dlogw
        
        # n1,n2 are probability "per bin". hence, to convert to a p(w) dlogw, we diving by the bin width in logw
        # z is z=0 bin
        n1 = np.sum(s.wplist[:,0]) * s.dlogw # n1 is probability within the bin. n3-6 are probability dlogp
        n2 = np.sum(Nw) * s.dlogw
        
        n3 = np.sum(ptau) * logbinwidth
        n4 = np.sum(Rtau) * logbinwidth
        n5 = np.sum(pw) * logbinwidth
        n6 = np.sum(Rw) * logbinwidth
    else:
        n1=1.
        n2=1.
        n3=1.
        n4=1.
        n5=1.
        n6=1.
    
    
    #### Plot 1: Intrinsic vs detected distributions #####
    plt.figure()
    plt.xscale("log")
    plt.yscale("log")
    plt.ylim(1e-2,1)
    plt.xlim(1e-2,1e2)
    
    plt.plot(ws,s.wplist[:,0]/n1,label="Total width (z=0)",linestyle="-")
    plt.plot(ws,Nw/n2,label="Detected total",linestyle=":",color=plt.gca().lines[-1].get_color())
    
    plt.plot(10**s.internal_logwvals,ptau/n3,label="Scattering (z=0)",linestyle="-")
    plt.plot(10**s.internal_logwvals,Rtau/n4,label="Detected scattering",linestyle=":",color=plt.gca().lines[-1].get_color())
    
    plt.plot(10**s.internal_logwvals,pw/n5,label="Intrinsic widths (z=0)",linestyle="-")
    plt.plot(10**s.internal_logwvals,Rw/n6,label="Detected width",linestyle=":",color=plt.gca().lines[-1].get_color())
    
    plt.xlabel("width [ms]")
    plt.ylabel("$\\rm p(w) d\\log_{10} w$")
    plt.legend()
    plt.tight_layout()
    plt.savefig(opdir+"pw.png")
    plt.close()
    
    
    #### Plot 2: Redsift dependence #####
    
    plt.figure()
    plt.xscale("log")
    plt.yscale("log")
    plt.ylim(1e-3,None)
    plt.plot(ws,s.wplist/np.sum(s.wplist),label="Intrinsic width",color="black")
    plt.plot(ws,Nw/np.sum(Nw),label="Detected FRBs",linestyle="--")
    plt.plot(ws,Nwz[:,4]/np.sum(Nwz[:,4]),label="    (z=0.25)",linestyle="--")
    plt.plot(ws,Nwz[:,18]/np.sum(Nwz[:,18]),label="    (z=1.25)",linestyle=":")
    plt.xlabel("FRB effective width [ms]")
    plt.ylabel("FRBs/day")
    plt.legend()
    plt.tight_layout()
    plt.savefig(opdir+"pw_zdep.png")
    plt.close()
    
    #### Plot 3: DM dependence #####
    
    plt.figure()
    plt.xscale("log")
    plt.yscale("log")
    plt.ylim(1e-3,None)
    plt.plot(ws,s.wplist/np.sum(s.wplist),label="Intrinsic width",color="black")
    plt.plot(ws,Nw/np.sum(Nw),label="Detected FRBs",linestyle="--")
    plt.plot(ws,Nwdm[:,3]/np.sum(Nwdm[:,3]),label="    (DM=125)",linestyle="--")
    plt.plot(ws,Nwdm[:,21]/np.sum(Nwdm[:,21]),label="    (DM=1025)",linestyle=":")
    plt.xlabel("FRB effective width [ms]")
    plt.ylabel("FRBs/day")
    plt.legend()
    plt.tight_layout()
    plt.savefig(opdir+"pw_dmdep.png")
    plt.close()
    
    ###### Plot 4: modelled p(w) #####
    
    widths,pw,zvals,pwz,dmvals,pwdm,finewidths,ptaus,ptausz,piis,piisz = mf.get_w_tau_dist(g)
    dw = np.log10(widths[3]/widths[2])
    bins = 10.**(np.arange(31)*dw-2)
    NFRB = s.frbs['WIDTH'].values.size
    
    plt.figure()
    plt.plot(widths,pw*NFRB)
    plt.hist(s.frbs['WIDTH'].values,bins=bins)
    plt.xlabel("Total width w [ms]")
    plt.ylabel("Probability, p(w)")
    plt.xscale("log")
    plt.yscale("log")
    plt.ylim(0.1,10)
    plt.tight_layout()
    plt.savefig(opdir+"detected_pw.png")
    plt.close()
    
    dfine = np.log10(finewidths[3]/finewidths[2])
    rel_width = dw/dfine
    plt.figure()
    plt.plot(finewidths,ptaus*rel_width*NFRB)
    #plt.hist(s.TAUs,bins=bins)
    plt.hist(s.TAUs,bins=s.wbins)
    plt.plot(finewidths,s.ScatFunction(np.log10(finewidths),s.slogmean,s.slogsigma))
    plt.xlabel("Total width w [ms]")
    plt.ylabel("Probability, p(w)")
    plt.xscale("log")
    plt.yscale("log")
    plt.ylim(0.1,1000)
    plt.tight_layout()
    plt.savefig(opdir+"detected_ptau.png")
    plt.close()
    
    plt.figure()
    plt.plot(finewidths,piis*rel_width*NFRB)
    plt.hist(s.IWIDTHs,bins=bins)
    plt.xlabel("Total width w [ms]")
    plt.ylabel("Probability, p(w)")
    plt.xscale("log")
    plt.yscale("log")
    plt.ylim(0.1,10)
    plt.tight_layout()
    plt.savefig(opdir+"detected_piw.png")
    plt.close()
    
    
    exit()
    # The below code shows how to plot p(tau|w) distributions for particlar widths
    # and redshifts.
    plt.figure()
    iw=10
    for iz in np.arange(6):
        z = g.zvals[iz]
        w = (s.wbins[iw]*s.wbins[iw+1])**0.5
        plt.plot(s.internal_logwvals,s.ptaus[iz,:,iw]/np.max(s.ptaus[iz,:,iw]),
                label="total width "+str(w)[0:5]+" at z="+str(z)[0:3])
    plt.legend()
    plt.ylim(1e-3,10)
    plt.yscale("log")
    plt.show()
    
main()
