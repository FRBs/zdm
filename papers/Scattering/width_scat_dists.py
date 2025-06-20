"""
This script plots observed and fitted width and scattering distributions


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
from zdm import figures as fig
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
    
    """
    
    # in case you wish to switch to another output directory
    opdir = "Plots/"
    if not os.path.exists(opdir):
        os.mkdir(opdir)
    
    # directory where the survey files are located. The below is the default - 
    # you can leave this out, or change it for a different survey file location.
    sdir = os.path.join(resource_filename('zdm', 'data'), 'Surveys')
    
    # make this into a list to initialise multiple surveys art once
    names = ["CRAFT_ICS_892","CRAFT_ICS_1300","CRAFT_ICS_1632"] # for example
    
    repeaters=False
    # sets plotting limits
    zmax = 2.
    dmmax = 2000
    
    survey_dict = {"WMETHOD": 3}
    state_dict = {}
    state_dict["scat"] = {}
    state_dict["scat"]["Sbackproject"] = True # turns on backprojection of tau and width for our model
    state_dict["width"] = {}
    state_dict["width"]["WNInternalBins"] = 1000 # sets it to a small quantity
    state_dict["width"]["WNbins"] = 33 # set to large number for this analysis
    
    surveys, grids = loading.surveys_and_grids(survey_names = names,\
                        repeaters=repeaters, sdir=sdir,nz=70,ndm=140,
                        survey_dict = survey_dict, state_dict = state_dict)
    
    # we now generate plots for the FRBs with known tau and width values
    # iwidths
    ilist = []
    wlist = []
    tlist = []
    
    # sets up a figure to do a scatter plot of frequency dependence of these widths
    plt.figure()
    plt.xlabel("Frequency [MHz]")
    plt.ylabel("$\\log_{10} t$ [ms]")
    for i,s in enumerate(surveys):
        
        nfrb = len(s.OKTAU)
        
        
        taus = s.TAUs[s.OKTAU]
        widths = s.frbs['WIDTH'].values[s.OKTAU]
        iwidths = s.IWIDTHs[s.OKTAU]
        
        logtaus = np.log10(taus)
        logwidths = np.log10(widths)
        logis = np.log10(iwidths)
        
        taubar = np.average(logtaus)
        taurms = (np.sum((logtaus - taubar)**2) / (nfrb-1))**0.5
        
        
        ibar = np.average(logis)
        irms = (np.sum((logis - ibar)**2) / (nfrb-1))**0.5
        
        
        wbar = np.average(logwidths)
        wrms = (np.sum((logwidths - wbar)**2) / (nfrb-1))**0.5
        
        freqs = s.frbs['FBAR'][s.OKTAU]
        fbar = np.average(freqs)
        
        if i==0:
            l1,=plt.plot(freqs, logwidths,marker='s',label="$w_{\\rm app}$",linestyle="")
            l2,=plt.plot(freqs, logtaus,marker="+",label="$\\tau$",linestyle="")
            l3,=plt.plot(freqs, logis,marker="x",label="$w_i$",linestyle="")
        else:
            plt.plot(freqs, logwidths,marker='s',color=l1.get_color(),linestyle="")
            plt.plot(freqs, logtaus,marker="+",color=l2.get_color(),linestyle="")
            plt.plot(freqs, logis,marker="x",color=l3.get_color(),linestyle="")
        
        plt.errorbar([fbar-20],[wbar],yerr=[wrms],color=l1.get_color(),linewidth=2,capsize=5)
        plt.errorbar([fbar],[taubar],yerr=[taurms],color=l2.get_color(),linewidth=2,capsize=6)
        plt.errorbar([fbar+20],[ibar],yerr=[irms],color=l3.get_color(),linewidth=2,capsize=7)
        
        for index in s.OKTAU:
            ilist.append(s.IWIDTHs[index])
            tlist.append(s.TAUs[index])
            wlist.append(s.frbs['WIDTH'].values[index])
    
    
    plt.legend()
    plt.tight_layout()
    # we do not save, because the figure is already done
    #plt.savefig(opdir+"freq_scat.png")
    plt.close()
    
    ####### We now plot against expectations ########
    # For each of w,i,and tau, we plot the true modelled distribution,
    # the de-biased distribution, and the observed distribution
    # We do this as both a cumulative distribution, and a pdf
    # This is also done for all bursts summed together.
    
    # generate cumulative distributions of these quantities
    xi,yi = fig.gen_cdf_hist(iwidths)
    xw,yw = fig.gen_cdf_hist(widths)
    xt,yt = fig.gen_cdf_hist(taus)
    
    
    for i,s in enumerate(surveys):
        g =grids[i]
        ################ Generates some plots ################
        
        ##### gets the expected, de-biased distributions #####
        # extracts the p(W) distribution
        Rw,Rtau,Nw,Nwz,Nwdm = mf.get_width_stats(s,g)
        
        # normalise each survey to actual number of FRBs
        nfrb = len(s.OKTAU)
        
        
        # these need to be normalised by the internal bin width
        logbinwidth = s.internal_logwvals[-1] - s.internal_logwvals[-2]
        
        # divide these histograms by the width of the log bins to get probability distributions
        if i==0:
            # intrinsic width, ptau, total width
            SumRw = Rw / np.sum(Rw) * nfrb / logbinwidth #has Ninternalbins
            SumRtau = Rtau / np.sum(Rtau) * nfrb / logbinwidth
            
            SumNw = Nw / np.sum(Nw) * nfrb / s.dlogw # has NWbins
        else:
            SumRw += Rw / np.sum(Rw) * nfrb
            SumRtau += Rtau / np.sum(Rtau) * nfrb
            SumNw += Nw / np.sum(Nw) * nfrb / s.dlogw
        
        ##### gets the underlying means #######
        # values at z=0
        WidthArgs = (s.wlogmean,s.wlogsigma)
        ScatArgs = (s.slogmean,s.slogsigma)
        
        pw = s.WidthFunction(s.internal_logwvals, *WidthArgs)#*s.dlogw #/logbinwidth
        ptau = s.ScatFunction(s.internal_logwvals, *ScatArgs)#*s.dlogw #/logbinwidth
        
        if i==0:
            # intrinsic width, ptau, total width
            Sumpw = pw / np.sum(pw) * nfrb
            Sumptau = ptau / np.sum(ptau) * nfrb
        else:
            Sumpw += pw / np.sum(pw) * nfrb
            Sumptau += ptau / np.sum(ptau) * nfrb
        
    # we now plot these differential distributions against histograms of the observations
    
    #### Plot 1: Intrinsic vs detected distributions #####
    plt.figure()
    plt.xscale("log")
    plt.yscale("log")
    plt.ylim(1e-1,100)
    plt.xlim(1e-2,1e2)
    
    l1,=plt.plot(s.wlist,SumNw,label="Total widths")
    l2,=plt.plot(10**s.internal_logwvals,SumRtau,label="Scattering",linestyle="-")
    l3,=plt.plot(10**s.internal_logwvals,SumRw,label="Intrinsic width",linestyle=":")
    
    bins=np.logspace(-2.01,2.01,9)
    lbw = np.log10(bins[1]/bins[0])
    #normalisation: p per log bin * Nfrb. 
    weights = np.full([len(wlist)],1./lbw)
    alpha=1.0
    plt.hist(wlist,bins=bins,weights=weights,alpha=alpha,facecolor = l1.get_color(),edgecolor = l1.get_color(),linewidth=2,histtype='step')
    plt.hist(tlist,bins=bins,weights=weights,alpha=alpha,facecolor = l2.get_color(),edgecolor = l2.get_color(),linewidth=2,histtype='step')
    plt.hist(ilist,bins=bins,weights=weights,alpha=alpha,facecolor = l3.get_color(),edgecolor = l3.get_color(),linewidth=2,histtype='step')
    
    
    plt.legend()
    plt.tight_layout()
    plt.savefig(opdir+"differential.png")
    
    plt.close()
    exit()
    
    plt.plot(ws,s.wplist/n1,label="Total width (z=0)",linestyle="-")
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

main()
