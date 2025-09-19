"""
This routine fits scattering and width distributions to CRAFT FRBs.

It loads data from the HTR and ICS survey papers, and performs the following
analysis:

- Calculation of completeness
- Best-fit using the KS-test
- Best-fit using maximum likelihood test
- Calculation of Bayes factors by integrating
  over distributions of priors
  
For each, we fit a range of functions defined at global level (yes, it's lazy!)
These are indexed according to the routine "function_wrapper".

Plots that get generated, for each of tau and scattering, are:
- histograms in observer frame
- histograms in host frame showing best fit functions
- histograms in host frames showing only best fits relevant to paper
- cumulative distributions with best fit KS functions
- cumulative distributions with best-fit likelihood functions
- spline fits to completeness

A scatter vs width plot
A plot showing function examples
"""


import numpy as np
from matplotlib import pyplot as plt
from zdm import misc_functions
import scipy as sp
import matplotlib
from scipy import stats
import sys
import os
import scipy as sp
from functions import *

import pandas

defaultsize=14
ds=4
font = {'family' : 'Helvetica',
        'weight' : 'normal',
        'size'   : defaultsize}
matplotlib.rc('font', **font)

###################################################################################################
######### These define the kinds of fitting functions, and their initial fit first guesses ########
###################################################################################################

# number of functions implemented, function names, and initial guesses
NFUNC=7
FNAMES=["lognormal","half-lognormal","boxcar","log constant","smooth boxcar", "upper sb", "lower sb","hln"]
ARGS0=[[0.,1.],[0.,1.],[-2,2],[-2],[-1.,1,1.],[-2.,2,1.],[-2.,2,1.],[0.,1.]]
WARGS0=[[0.,1.],[0.,1.],[-3.5,2],[-3.5],[-1.,1,1.],[-3.5,2,1.],[-3.5,2,1.],[0.,1.]]

# set up min/max ranges of integrations for Bayes factors
min_sigma = 0.3
max_sigma = 4
min_scat = 1e-6
max_scat = 1000
log_min_sigma = np.log10(min_sigma)
log_max_sigma = np.log10(max_sigma)
log_min_scat = np.log10(min_scat)
log_max_scat = np.log10(max_scat)

# priors for Bayes factors
MIN_PRIOR = [
            [log_min_scat,log_min_sigma],
            [log_min_scat,log_min_sigma],
            [log_min_scat,log_min_scat],
            [log_min_scat],
            [log_min_scat,log_min_scat,log_min_sigma],
            [log_min_scat,log_min_scat,log_min_sigma],
            [log_min_scat,log_min_scat,log_min_sigma],
            [log_min_scat,log_min_sigma]
            ]

MAX_PRIOR = [
            [log_max_scat,log_max_sigma],
            [log_max_scat,log_max_sigma],
            [log_max_scat,log_max_scat],
            [log_max_scat],
            [log_max_scat,log_max_scat,log_max_sigma],
            [log_max_scat,log_max_scat,log_max_sigma],
            [log_max_scat,log_max_scat,log_max_sigma],
            [log_max_scat,log_max_sigma]
            ]

# CHIME values
# raw were 
# scaled to 1GHz and log10 are
# $(\mu_w,\sigma_w) = (\log_{10} 1.0 {\rm ms},0.42)$
# $(\mu_\tau,\sigma_\tau) = (\log_{10} 0.262 {\rm ms},0.75)
CHIME_muw = 0.
CHIME_sw = 0.42
CHIME_mut = 0.3
CHIME_st = 0.75

def main(outdir="Fitting_Outputs/"):
    # does k-s test to scattering distribution
    
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    
    plot_functions(outdir=outdir)
    
    tns,tauobs,w95,wsnr,z,snr,freq,DM,tres = get_data()
    
    # gets observed intrinsic widths
    wobs = wsnr**2 - (tauobs/0.816)**2
    BAD = np.where(wobs < 0.)
    # this is maybe a bit dodgy. Here, we set it to scattering/10.
    #wobs[BAD] = 0.00001
    
    wobs[BAD] = 0.01*tauobs[BAD]**2
    #wobs[BAD] = 1.e-2
    wobs = wobs**0.5
    
    # generates scatter plot of tau and width
    plt.figure()
    plt.scatter(wsnr,tauobs)
    plt.xlabel("$w_{\\rm SNR}$")
    plt.ylabel("$\\tau_{\\rm obs}$")
    plt.xscale("log")
    plt.yscale("log")
    slope = 0.816
    x=np.array([1e-2,20])
    y=slope*x
    plt.plot(x,y,linestyle="--",color="black")
    plt.tight_layout()
    plt.savefig(outdir+"scatter_w_tau.png")
    plt.close()
    
    NFRB = snr.size
    
    maxtaus = np.zeros([NFRB])
    for i in np.arange(NFRB):
        taumax = find_max_tau(tauobs[i],wsnr[i],snr[i],freq[i],DM[i],tres[i],plot=False)
        maxtaus[i] = taumax
    
    
    maxws = np.zeros([NFRB])
    for i in np.arange(NFRB):
        
        wmax = find_max_w(tauobs[i],wsnr[i],snr[i],freq[i],DM[i],tres[i],plot=False)
        maxws[i] = wmax
    
    # scale taus to 1 GHz in host rest frame
    alpha = -4
    host_tau = tauobs*(1+z)**3 * (freq/1e3)**-alpha
    host_maxtaus = maxtaus * (1+z)**3 * (freq/1e3)**-alpha
    
    host_w = wobs/(1+z)
    host_maxw = maxws/(1+z)
    #for i,x in enumerate(host_maxw):
    #    print(i,z[i],maxws[i],host_maxw[i])
    
    
    
    Nbins=11
    bins = np.logspace(-3.,2.,Nbins)
    
    
    ##################################################################
    ############################   TAU   #############################
    ##################################################################
    
    ############## Observed Histogram #################
    
    plt.figure()
    #obshist,bins = np.histogram(tauobs,bins=bins)
    #hosthist,bins = np.histogram(host_tau,bins=bins)
    ax1=plt.gca()
    plt.xscale("log")
    plt.ylim(0,8)
    l1 = plt.hist(tauobs,bins=bins,label="Observed",alpha=0.5)
    
    # makes a function of completeness
    xvals,yvals = make_completeness_plot(maxtaus)
    
    tau_comp = get_completeness(tauobs,xvals,yvals)
    
    l2 = plt.hist(tauobs,bins=bins,weights = 1./tau_comp,label="Observed",alpha=0.5)
    
    
    ax2 = ax1.twinx()
    l3 = ax2.plot(xvals,yvals,label="Completeness")
    plt.ylim(0,1)
    plt.xlim(1e-2,1e3)
    plt.ylabel("Completeness")
    plt.sca(ax1)
    plt.legend(handles=[l1[2],l3[0],l2[2]],labels=["Observed","Completeness","Corrected"],fontsize=12)
    plt.xlabel("$\\tau_{\\rm obs}$ [ms]")
    plt.ylabel("Number of FRBs")
    plt.text(2e-3,8,"(a)",fontsize=18)
    plt.tight_layout()
    plt.savefig(outdir+"tau_observed_histogram.png")
    plt.close()
    

    
    ############## 1 GHz Rest-frame Histogram #################
    
    plt.figure()
    #obshist,bins = np.histogram(tauobs,bins=bins)
    #hosthist,bins = np.histogram(host_tau,bins=bins)
    ax1=plt.gca()
    plt.xscale("log")
    plt.ylim(0,8)
    l1 = plt.hist(host_tau,bins=bins,label="Observed",alpha=0.5)
    
    # makes a function of completeness
    xvals,yvals = make_completeness_plot(host_maxtaus)
    
    # get completeness at points of measurement
    tau_comp = get_completeness(host_tau,xvals,yvals)
    
    l2 = plt.hist(host_tau,bins=bins,weights = 1./tau_comp,label="Corrected",alpha=0.5)
    
    
    ax2 = ax1.twinx()
    l3 = ax2.plot(xvals,yvals,label="Completeness")
    use_this_color = l3[0].get_color()
    
    plt.ylim(0,1)
    plt.xlim(1e-2)
    plt.ylabel("Completeness")
    plt.sca(ax1)
    plt.legend(handles=[l1[2],l3[0],l2[2]],labels=["Observed","Completeness","Corrected"],fontsize=12)
    plt.xlabel("$\\tau_{\\rm host, 1\,GHz}$ [ms]")
    plt.ylabel("Number of FRBs")
    
    plt.text(2e-3,8,"(b)",fontsize=18)
    plt.tight_layout()
    plt.savefig(outdir+"tau_host_histogram.png")
    
    #### creates a copy of the above, for paper purposes
    
    plt.figure()
    #obshist,bins = np.histogram(tauobs,bins=bins)
    #hosthist,bins = np.histogram(host_tau,bins=bins)
    ax1v2=plt.gca()
    plt.xscale("log")
    plt.ylim(0,8)
    l1v2 = plt.hist(host_tau,bins=bins,label="Observed",alpha=0.5)
    
    # makes a function of completeness
    xvals,yvals = make_completeness_plot(host_maxtaus)
    
    # get completeness at points of measurement
    tau_comp = get_completeness(host_tau,xvals,yvals)
    
    l2v2 = plt.hist(host_tau,bins=bins,weights = 1./tau_comp,label="Corrected",alpha=0.5)
    
    
    ax2v2 = ax1v2.twinx()
    l3v2 = ax2v2.plot(xvals,yvals,label="Completeness")
    
    plt.ylim(0,1)
    plt.xlim(1e-2)
    plt.ylabel("Completeness")
    plt.sca(ax1)
    plt.xlabel("$\\tau_{\\rm host, 1\,GHz}$ [ms]")
    plt.ylabel("Number of FRBs")
    # keeps open for later plotting - don't close this here
    
    
    ####################################### TAU - CDF and fitting ################################
    print("\n\n   KS test evaluation for tau \n")
    # amplitude, mean, and std dev of true distribution
    
    ksbest = []
    # begins minimisation for KS statistic
    for ifunc in np.arange(NFUNC):
        args = (host_tau,xvals,yvals,ifunc)
        x0=ARGS0[ifunc]
        
        result = sp.optimize.minimize(get_ks_stat,x0=x0,args=args,method = 'Nelder-Mead')
        psub1 = get_ks_stat(result.x,*args,plot=False)
        ksbest.append(result.x)
        #Best-fitting parameters are  [0.85909445 1.45509687]  with p-value  0.9202915513749959
        print("FUNCTION ",ifunc,",",FNAMES[ifunc]," Best-fitting parameters are ",result.x," with p-value ",1.-result.fun)
    
    
    xtemp = xvals[::2]
    ytemp = yvals[::2]
    s = xtemp.size
    xt = np.zeros([s+2])
    yt = np.zeros([s+2])
    xt[0] = xtemp[0]
    yt[0] = ytemp[0]
    xt[1] = 1.
    yt[1] = 1.
    xt[2:-1] = xtemp[1:]
    yt[2:-1] = ytemp[1:]
    xt[-1] = 1e5
    yt[-1] = 0.
    cspline = sp.interpolate.make_interp_spline(np.log10(xt), yt,k=1)
        # get a spline interpolation of completeness. Should be removed from function!
    
    # do a test plot of the spline
    if True:
        plt.figure()
        plt.plot(np.logspace(-5,5,101),cspline(np.linspace(-5,5,101)))
        plt.plot(xvals,yvals)
        plt.xscale("log")
        plt.savefig(outdir+"tau_spline_example.png")
        plt.close()
    
    # make a cdf plot of the best fits
    make_cdf_plot(ksbest,host_tau,xvals,yvals,outdir+"bestfit_ks_scat_cumulative.png",cspline)
    
    ############################################### TAU - likelihood analysis #################################################
    print("\n\n   Max Likelihood Calculation for tau\n")
    xbest=[]
    for ifunc in np.arange(NFUNC):
        args = (host_tau,cspline,ifunc)
        x0=ARGS0[ifunc]
        
        result = sp.optimize.minimize(get_ll_stat,x0=x0,args=args,method = 'Nelder-Mead')
        #psub1 = get_ks_stat(result.x,*args,plot=True)
        xbest.append(result.x)
        # llbest returns negative ll
        llbest = get_ll_stat(result.x,host_tau,cspline,ifunc) * -1
        
        
        print("FUNCTION ",ifunc,",",FNAMES[ifunc]," Best-fitting log-likelihood parameters are ",result.x," with p-value ",1.-result.fun)
        print("          , BIC is ",2*np.log(host_tau.size) - len(x0)*llbest)
        if ifunc == 0:
            llCHIME = get_ll_stat([CHIME_mut,CHIME_st],host_tau,cspline,ifunc) * -1
            print("Compare with CHIME ",llCHIME)
    
    
    make_cdf_plot(xbest,host_tau,xvals,yvals,outdir+"bestfit_ll_scat_cumulative.png",cspline)
    
    
    ######## does plot with  all fits added ########
    
    plt.sca(ax1)
    NFRB=host_tau.size
    handles=[l1[2],l3[0],l2[2]]
    labels=["Observed","Completeness","Corrected$"]
    for i in np.arange(NFUNC):
        print("plotting function ",i," with xbest ",xbest[i])
        xs,ys = function_wrapper(i,xbest[i])#cspline=None):
        plotnorm = NFRB * (np.log10(bins[1])-np.log10(bins[0]))
        l=plt.plot(xs,ys*plotnorm,label=FNAMES[i])
        handles.append(l[0])
        labels.append(FNAMES[i])
        xs,ys = function_wrapper(i,xbest[i],cspline=cspline)
        plt.plot(xs,ys*plotnorm,linestyle=":",color=plt.gca().lines[-1].get_color())
    plt.xscale("log")
    plt.xlim(1e-2,1e3)
    plt.legend()
    
    plt.legend(handles=handles,labels=labels,fontsize=6) #fontsize=12)
    
    
    plt.savefig(outdir+"tau_host_histogram_fits.png")
    plt.close()
    
    ######## plot for paper ########
    
    plt.sca(ax1v2)
    NFRB=host_tau.size
    handles=[l1v2[2],l3v2[0],l2v2[2]]
    labels=["Observed","Completeness","Corrected"]
    for i in [0,2,3]:
        print("plotting function ",i," with xbest ",xbest[i])
        xs,ys = function_wrapper(i,xbest[i])#cspline=None):
        plotnorm = NFRB * (np.log10(bins[1])-np.log10(bins[0]))
        l=plt.plot(xs,ys*plotnorm,label=FNAMES[i])
        handles.append(l[0])
        labels.append(FNAMES[i])
        xs,ys = function_wrapper(i,xbest[i],cspline=cspline)
        plt.plot(xs,ys*plotnorm,linestyle=":",color=plt.gca().lines[-1].get_color())
    plt.xscale("log")
    plt.xlim(1e-2,1e3)
    
    plt.text(1e-3,8,"(b)",fontsize=18)
    plt.legend()
    plt.xlabel("$\\tau_{\\rm host, 1\,GHz}$ [ms]")
    plt.ylabel("Number of FRBs")
    plt.legend(handles=handles,labels=labels,fontsize=10) #fontsize=12)
    
    plt.tight_layout()
    plt.savefig(outdir+"paper_tau_host_histogram_fits.png")
    plt.close()
    
    
    ############################################# TAU - bayes factor #######################################
    # priors
    # uses Galactic pulsars as example
    # min sigma: from pulsar scattering relation factor of
    # max sigma: size of distribution
    # min mean/min/max: size of distribution
    # NOTE: Bayes factor is P(data|model1)/P(data|model2)
    
    print("\n\n   Bayes Factor Calculation\n")
    for ifunc in np.arange(NFUNC):
        if True:
            print("skipping Bayes factor calculation for Tau, remove this line to re-run")
            #FUNCTION  0  has likelihood sum  2.6815961887322472e-21  now compute Bayes factor!
            #FUNCTION  1  has likelihood sum  1.4462729739641349e-15  now compute Bayes factor!
            #FUNCTION  2  has likelihood sum  5.364450880196842e-16  now compute Bayes factor!
            #FUNCTION  3  has likelihood sum  1.2416558548887762e-15  now compute Bayes factor!
            #FUNCTION  4  has likelihood sum  7.269113417017299e-16  now compute Bayes factor!
            #FUNCTION  5  has likelihood sum  6.248172055823414e-16  now compute Bayes factor!
            #FUNCTION  6  has likelihood sum  6.339647378300337e-16  now compute Bayes factor!
            continue
        
        llsum=0.
        N1=100
        N2=100
        N3=100
        # integrates likelihood over prior space
        for p1 in np.linspace(MIN_PRIOR[ifunc][0],MAX_PRIOR[ifunc][0],N1):
            if len(ARGS0[ifunc])>1:
                for p2 in np.linspace(MIN_PRIOR[ifunc][1],MAX_PRIOR[ifunc][1],N2):
                    
                    if (ifunc == 2 or ifunc == 4 or ifunc==5 or ifunc==6) and p2 <= p1:
                        # skip if max is less than min
                        continue
                    
                    if len(ARGS0[ifunc])>2:
                        for p3 in np.linspace(MIN_PRIOR[ifunc][2],MAX_PRIOR[ifunc][2],N3):
                            ll = get_ll_stat([p1,p2,p3],host_tau,cspline,ifunc,log_min=-5, log_max = 5, NTau=600)
                            if np.isfinite(ll):
                                llsum += 10.**-ll
                    else:
                        ll = get_ll_stat([p1,p2],host_tau,cspline,ifunc,log_min=-5, log_max = 5, NTau=600)
                        if np.isfinite(ll):
                            llsum += 10.**-ll
            else:
                ll = get_ll_stat([p1],host_tau,cspline,ifunc,log_min=-5, log_max = 5, NTau=600)
                if np.isfinite(ll):
                    llsum += 10.**-ll
        # normalisation
        llsum /= N1
        if len(ARGS0[ifunc])>1:
            llsum /= N2
        if len(ARGS0[ifunc])>2:
            llsum /= N3
        if (ifunc == 2 or ifunc == 4 or ifunc==5 or ifunc==6):
            llsum *= 2 #because the parameter space is actually half that calculated above
            
        print("FUNCTION ",ifunc,",",FNAMES[ifunc]," has likelihood sum ",llsum, " now compute Bayes factor!")
    
    
    
    
    ######################################################################################################
    ##############################################  WIDTH  ###############################################
    ######################################################################################################
    
    print("\n\n\n\n#########   WIDTH    #########\n")
    ######################################### Observed Histogram ############################################
    
    plt.figure()
    ax1=plt.gca()
    plt.xscale("log")
    plt.ylim(0,8)
    l1 = plt.hist(wobs,bins=bins,label="Observed",alpha=0.5)
    
    # makes a function of completeness
    wxvals,wyvals = make_completeness_plot(maxws)
    
    wi_comp = get_completeness(wobs,wxvals,wyvals)
    
    l2 = plt.hist(wobs,bins=bins,weights = 1./wi_comp,label="Observed",alpha=0.5)
    
    
    ax2 = ax1.twinx()
    l3 = ax2.plot(wxvals,wyvals,label="Completeness")
    plt.ylim(0,1)
    plt.xlim(5e-3,1e2)
    plt.ylabel("Completeness")
    plt.sca(ax1)
    plt.legend(handles=[l1[2],l3[0],l2[2]],labels=["Observed","Completeness","Corrected $w_i$"],fontsize=12)
    plt.xlabel("$w_i$ [ms]")
    plt.ylabel("Number of FRBs")
    
    plt.text(1e-3,8,"(a)",fontsize=18)
    plt.tight_layout()
    plt.savefig(outdir+"w_observed_histogram.png")
    plt.close()
    
    ################################ 1 GHz Rest-frame Histogram ##########################
    
    plt.figure()
    ax1=plt.gca()
    plt.xscale("log")
    plt.ylim(0,8)
    l1 = plt.hist(host_w,bins=bins,label="Host",alpha=0.5)
    
    # makes a function of completeness
    wxvals,wyvals = make_completeness_plot(host_maxw)
    
    # get completeness at points of measurement
    w_comp = get_completeness(host_w,wxvals,wyvals)
    
    l2 = plt.hist(host_w,bins=bins,weights = 1./w_comp,label="Observed",alpha=0.5)
    
    
    ax2 = ax1.twinx()
    l3 = ax2.plot(wxvals,wyvals,label="Completeness")
    
    plt.ylim(0,1)
    plt.xlim(5e-3,1e2)
    plt.ylabel("Completeness")
    plt.sca(ax1)
    plt.legend(handles=[l1[2],l3[0],l2[2]],labels=["Observed","Completeness","Corrected"],fontsize=12)
    plt.xlabel("$w_{\\rm host}$ [ms]")
    plt.ylabel("Number of FRBs")
    plt.tight_layout()
    plt.savefig(outdir+"w_host_histogram.png")
    
    # keeps open for plotting later
    
    #### new plot, for paper - just a copy of the above ####
    
    plt.figure()
    plt.xscale("log")
    plt.ylim(0,8)
    l1v2 = plt.hist(host_w,bins=bins,label="Host",alpha=0.5)
    
    ax1v2=plt.gca()
    # makes a function of completeness
    wxvals,wyvals = make_completeness_plot(host_maxw)
    
    # get completeness at points of measurement
    w_comp = get_completeness(host_w,wxvals,wyvals)
    
    l2v2 = plt.hist(host_w,bins=bins,weights = 1./w_comp,label="Observed",alpha=0.5)
    
    ax2v2 = ax1v2.twinx()
    l3v2 = ax2v2.plot(wxvals,wyvals,label="Completeness")#,color=use_this_color)
    
    plt.ylim(0,1)
    plt.xlim(1e-3,1e3)
    plt.ylabel("Completeness")
    plt.sca(ax1v2)
    plt.legend(handles=[l1v2[2],l3v2[0],l2v2[2]],labels=["Observed","Completeness","Corrected"],fontsize=12)
    plt.xlabel("$w_{\\rm host}$ [ms]")
    plt.ylabel("Number of FRBs")
    
    
    ####################### W - likelihood maximisation #################
    
    ####################################### Width - CDF and fitting ################################
    print("\n\n   KS test evaluation for width \n")
    # amplitude, mean, and std dev of true distribution
    
    ksbest = []
    # begins minimisation for KS statistic
    for ifunc in np.arange(NFUNC):
        args = (host_w,wxvals,wyvals,ifunc)
        x0=WARGS0[ifunc]
        
        result = sp.optimize.minimize(get_ks_stat,x0=x0,args=args,method = 'Nelder-Mead')
        psub1 = get_ks_stat(result.x,*args,plot=False)
        ksbest.append(result.x)
        print("FUNCTION ",ifunc,",",FNAMES[ifunc]," Best-fitting parameters are ",result.x," with p-value ",1.-result.fun)
    
    
    print("\n\n   Maximum likelihood for width \n")
    ### makes temporary values for completeness
    xtemp = wxvals[::2]
    ytemp = wyvals[::2]
    s = xtemp.size
    xt = np.zeros([s+2])
    yt = np.zeros([s+2])
    xt[0] = xtemp[0]
    yt[0] = ytemp[0]
    xt[1] = 0.1
    yt[1] = 1.
    xt[2:-1] = xtemp[1:]
    yt[2:-1] = ytemp[1:]
    xt[-1] = 1e5
    yt[-1] = 0.
    cspline = sp.interpolate.make_interp_spline(np.log10(xt), yt,k=1)
    # do a test plot of the spline?
    if True:
        plt.figure()
        plt.plot(np.logspace(-5,5,101),cspline(np.linspace(-5,5,101)))
        plt.plot(xvals,yvals)
        plt.xscale("log")
        plt.savefig(outdir+"width_spline_example.png")
        plt.close()
    # make a cdf plot of the best fits
    make_cdf_plot(ksbest,host_tau,xvals,yvals,outdir+"bestfit_ks_width_cumulative.png",cspline)
    
    ####################################### Width - max likelihood ################################
    print("\n\n   Likelhiood maximasation for width \n")
    # amplitude, mean, and std dev of true distribution
      
    xbest=[]
    
    # iterate over functions to calculate max likelihood
    for ifunc in np.arange(NFUNC):
        
        args = (host_w,cspline,ifunc)
        x0=WARGS0[ifunc]
        result = sp.optimize.minimize(get_ll_stat,x0=x0,args=args,method = 'Nelder-Mead')
        #psub1 = get_ks_stat(result.x,*args,plot=True)
        xbest.append(result.x)
        print("width FUNCTION ",ifunc,",",FNAMES[ifunc]," Best-fitting log-likelihood parameters are ",result.x," with p-value ",1.-result.fun)
        if ifunc == 0:
            llCHIME = get_ll_stat([CHIME_muw,CHIME_sw],host_tau,cspline,ifunc) * -1
            print("Compare with CHIME ",llCHIME)
    
    make_cdf_plot(xbest,host_w,wxvals,wyvals,outdir+"bestfit_ll_width_cumulative.png",cspline,width=True)
    
    
    ### does plot with fits added ###
    
    plt.sca(ax1)
    NFRB=host_tau.size
    handles=[l1[2],l3[0],l2[2]]
    labels=["$w_{\\rm host}$","Completeness","Corrected $w_{\\rm host}$"]
    for i in np.arange(NFUNC):
        print("plotting function ",i," with xbest ",xbest[i])
        xs,ys = function_wrapper(i,xbest[i])#cspline=None):
        plotnorm = NFRB * (np.log10(bins[1])-np.log10(bins[0]))
        l=plt.plot(xs,ys*plotnorm,label=FNAMES[i])
        handles.append(l[0])
        labels.append(FNAMES[i])
        xs,ys = function_wrapper(i,xbest[i],cspline=cspline)
        plt.plot(xs,ys*plotnorm,linestyle=":",color=plt.gca().lines[-1].get_color())
    plt.xscale("log")
    plt.xlim(1e-4,1e3)
    plt.legend()
    
    plt.legend(handles=handles,labels=labels,fontsize=6) #fontsize=12)
    
    
    plt.savefig(outdir+"w_host_histogram_fits.png")
    plt.close()
    
    #### for paper ####
    
    
    plt.sca(ax1v2)
    plt.ylim(0,12)
    NFRB=host_w.size
    handles=[l1v2[2],l3v2[0],l2v2[2]]
    labels=["Observed","Completeness","Corrected"]
    for i in [0,1,3]:
        print("plotting function ",i," with xbest ",xbest[i])
        xs,ys = function_wrapper(i,xbest[i])#cspline=None):
        plotnorm = NFRB * (np.log10(bins[1])-np.log10(bins[0]))
        l=plt.plot(xs,ys*plotnorm,label=FNAMES[i])
        handles.append(l[0])
        labels.append(FNAMES[i])
        xs,ys = function_wrapper(i,xbest[i],cspline=cspline)
        plt.plot(xs,ys*plotnorm,linestyle=":",color=plt.gca().lines[-1].get_color())
    plt.xscale("log")
    
    plt.text(1e-3,12,"(b)",fontsize=18)
    plt.xlim(5e-3,1e2)
    plt.legend()
    plt.xlabel("$w_{\\rm host}$ [ms]")
    plt.ylabel("Number of FRBs")
    plt.legend(handles=handles,labels=labels,fontsize=12) #fontsize=12)
    
    plt.tight_layout()
    plt.savefig(outdir+"paper_w_host_histogram_fits.png")
    plt.close()
    
    ############################################# WIDTH - bayes factor #######################################
    # priors
    # uses Galactic pulsars as example
    # min sigma: from pulsar scattering relation factor of
    # max sigma: size of distribution
    # min mean/min/max: size of distribution
    # NOTE: Bayes factor is P(data|model1)/P(data|model2)
    
    print("\n\n   Bayes Factor Calculation\n")
    for ifunc in np.arange(NFUNC):
        if True:
            print("skipping Bayes factor calculation for width, remove this line to re-run")
            #FUNCTION  0 , lognormal  has likelihood sum  4.287511548315901e-15  now compute Bayes factor!
            #FUNCTION  1 , half-lognormal  has likelihood sum  1.3919340351669428e-12  now compute Bayes factor!
            #FUNCTION  2 , boxcar  has likelihood sum  3.1091474793575766e-13  now compute Bayes factor!
            #FUNCTION  3 , log constant  has likelihood sum  6.41635848895752e-13  now compute Bayes factor!
            #FUNCTION  4 , smooth boxcar  has likelihood sum  8.176332379444694e-13  now compute Bayes factor!
            #FUNCTION  5 , upper sb  has likelihood sum  3.595417631395158e-13  now compute Bayes factor!
            #FUNCTION  6 , lower sb  has likelihood sum  6.806166849685634e-13  now compute Bayes factor!
            continue
        
        llsum=0.
        N1=100
        N2=100
        N3=100
        # integrates likelihood over prior space
        for p1 in np.linspace(MIN_PRIOR[ifunc][0],MAX_PRIOR[ifunc][0],N1):
            if len(ARGS0[ifunc])>1:
                for p2 in np.linspace(MIN_PRIOR[ifunc][1],MAX_PRIOR[ifunc][1],N2):
                    
                    if (ifunc == 2 or ifunc == 4 or ifunc==5 or ifunc==6) and p2 <= p1:
                        # skip if max is less than min
                        continue
                    
                    if len(ARGS0[ifunc])>2:
                        for p3 in np.linspace(MIN_PRIOR[ifunc][2],MAX_PRIOR[ifunc][2],N3):
                            ll = get_ll_stat([p1,p2,p3],host_w,cspline,ifunc,log_min=-3, log_max = 3, NTau=300)
                            if np.isfinite(ll):
                                llsum += 10.**-ll
                    else:
                        ll = get_ll_stat([p1,p2],host_w,cspline,ifunc,log_min=-3, log_max = 3, NTau=300)
                        if np.isfinite(ll):
                            llsum += 10.**-ll
            else:
                ll = get_ll_stat([p1],host_w,cspline,ifunc,log_min=-3, log_max = 3, NTau=300)
                if np.isfinite(ll):
                    llsum += 10.**-ll
        # normalisation
        llsum /= N1
        if len(ARGS0[ifunc])>1:
            llsum /= N2
        if len(ARGS0[ifunc])>2:
            llsum /= N3
        if (ifunc == 2 or ifunc == 4 or ifunc==5 or ifunc==6):
            llsum *= 2 #because the parameter space is actually half that calculated above
            
        print("FUNCTION ",ifunc,",",FNAMES[ifunc]," has likelihood sum ",llsum, " now compute Bayes factor!")
    
    

def plot_functions(outdir=""):
    """
    Plots example functions for the paper
    """
    plt.figure()
    
    tmin=-1.5
    tmax=1.5
    tbar=0
    sigma=1
    
    
    plt.plot([10**tmin,10**tmin],[0,5.8],color="black",linestyle=":")
    plt.text(10**tmin*0.5,-0.3,"$t_{\\rm min}$")
    
    plt.plot([10**tmax,10**tmax],[0,5.8],color="black",linestyle=":")
    plt.text(10**tmax*0.5,-0.3,"$t_{\\rm max}$")
    
    
    plt.text(10**(tmax+sigma)*0.25,1.5,"$\\sigma_{t}$")
    plt.arrow(10**(tmax)*5,1.5,10**(tmax+sigma)-10**tmax*5,0,color="black",linestyle=":",
                shape="full",head_width=0.3,head_length=100,length_includes_head=True)
    plt.arrow(10**(tmax)*2.5,1.5,-10**(tmax)*1.5,0,color="black",linestyle=":",
                shape="full",head_width=0.3,head_length=15,length_includes_head=True)
                
                
                
    #plt.plot([10**(tmax),10**(tmax+sigma)],[1.5,1.5],color="black",linestyle=":",)
    plt.plot([10**(tmax+sigma),10**(tmax+sigma)],[1.5,3.2],color="black",linestyle=":")
    
    
    plt.text(10**tbar*0.7,6,"$\\mu_{t}$")
    plt.plot([10**tbar,10**tbar],[6.3,8],color="black",linestyle=":")
    
    plot_args=[[tbar,sigma],[tbar,sigma],[tmin,tmax],[tmin],[tmin,tmax,sigma],
                                [tmin,tmax,sigma],[tmin,tmax,sigma],[0.,1.]]
    styles=["-","--",":","-."]
    xlabels=[1e-3,1e1,1e-1,1e-1,1e-1,1e-1,1e-1]
    
    for ifunc in np.arange(NFUNC):
        args = plot_args[ifunc]
        xs,ys = function_wrapper(ifunc,args,xvals=None,logxmin=-6,logxmax=6,N=1000,cspline=None)
        plt.plot(xs,7+ys/np.max(ys)-1.1*ifunc,label=FNAMES[ifunc],linestyle=styles[ifunc%4])
        plt.text(xlabels[ifunc],7.5-1.1*ifunc,FNAMES[ifunc],color=plt.gca().lines[-1].get_color())
    
    
    
    
    plt.xlim(1e-4,1e4)
    plt.xlabel("t [ms]")
    plt.xscale("log")
    #plt.legend()
    plt.ylabel("f(t)")
    plt.tight_layout()
    plt.savefig(outdir+"example_functions.png")
    plt.close()
    
def get_ll_stat(args,tau_obs,cspline,ifunc,plot=False, log_min=-5, log_max = 5, NTau=600):
    #(tau_obs, completeness_x, completeness_y, prediction, log_min=-3, log_max = 3, NTau=600):
    """
    Returns a likelihood, given:
    
    Args:
        args[1]: logmean, or logmin, of distribution
        aargs[2]: sigma, or logmax, of distribution
        tau_obs: observed values of scattering (in host frame)
        cspline: spline giving completeness as function of log_tau
        ifunc: which function to use for modelling (0,1, or 2)
    """
    
    # modifies the likelihood function by the completeness, and normalises
    taus = np.logspace(log_min, log_max, NTau)
    logtaus = np.linspace(log_min, log_max, NTau)
    dlx = logtaus[1]-logtaus[0]
    ctaus = cspline(logtaus)
    
    log_tau_obs = np.log10(tau_obs)
    
    #a1 = args[0]
    #a2 = args[1]
    # get expected distribution and hence cdf
    A=1.
    
    ftaus = function_wrapper(ifunc,args,taus,cspline=cspline)
    ptaus = function_wrapper(ifunc,args,tau_obs,cspline=cspline)
    
    # checks normalisation in log-space
    #modified = ctaus*ftaus
    #norm = np.sum(modified) * dlx #/NTau # sets the posterior to the correct normalisation in log space
    #print("Norm is ",norm)
    
    # already includes nornalisation
    ls = ptaus
    #ls = ptaus * cspline(log_tau_obs ) / norm
    #print("ls are ",ls)
    lls = np.log10(ls)
    ll = np.sum(lls)
    
    return -ll

      
def get_ks_stat(args,taus,cxvals,cyvals,ifunc,plot=False):
    """
    Args:
        a1: logmean, or logmin, of distribution
        a2: sigma, or logmax, of distribution
        taus: observed values of scattering
        cxvals: x-values at which completeness is calculated
        cyvals: y-values at which completeness is calculated
        ifunc: which function to use for modelling (0,1, or 2)
        
        
    """
    #a1 = args[0]
    #a2 = args[1]
    
    # Step 1: create an expected cumulative distribution
    xvals = np.logspace(-4,4,101)
    
    yvals=function_wrapper(ifunc,args,xvals)
    
    # modify the expected distribution by the completeness
    modyvals = yvals*get_completeness(xvals,cxvals,cyvals)
    
    # make a cdf
    cum_dist = np.cumsum(modyvals)
    cum_dist /= cum_dist[-1]
    
    # make these global variables to pass to the cdf
    global xc,yc
    xc = np.log10(xvals)
    yc = cum_dist
    
    # does this in log10 space
    result = stats.ks_1samp(np.log10(taus),cdf,alternative='two-sided', method='auto')
    
    pvalue = result.pvalue
    
    if plot:
        plt.figure()
        plt.xlabel("$\\tau$")
        plt.ylabel("CDF")
        plt.xscale("log")
        plt.plot(xvals,yc,label="Model")
        
        # makes a function of completeness
        xvals,yvals = make_completeness_plot(taus,reverse=False)
        plt.plot(xvals,yvals,label="Observed")
        
        plt.legend()
        plt.tight_layout()
        plt.show()
    
    # we are trying to minimise 1.-p, hence maximise p-value
    return  1.-pvalue

  
def make_cdf_plot(args,taus,cxvals,cyvals,outfile,cspline,plot=False,width=False):
    """
    Args:
        args1: arguments to lognormal
        args2: arguments to halflognormal
        args3: arguments to loguniform
        taus: observed values of scattering
        cxvals: x-values at which completeness is calculated
        cyvals: y-values at which completeness is calculated
        
        
    """
    
    # Base plotting commands
    plt.figure()
    plt.xlim(1e-3,1e3)
    plt.ylim(0,1)
    if width:
        plt.xlabel("$w_{\\rm rest}$")
    else:
        plt.xlabel("$\\tau_{\\rm 1 GHz,rest}$")
    plt.ylabel("CDF")
    plt.xscale("log")
    
    # Step 1: create an expected cumulative distribution
    xvals = np.logspace(-3,3,101)
    
    ### Loop over functions ###
    
    #def function_wrapper(ifunc,args,logxmin=-6,logxmax=6,N=1000,cspline=None):
    # get expected distribution and hence cdf
    A=1.
    nfunc=len(args)
    yvals=[]
    cum_dists=[]
    for i in np.arange(nfunc):
        xs,ys=function_wrapper(i,args[i],logxmin=-3,logxmax=3,N=101,cspline=cspline)
        #ys *= cspline(xs)
        yvals.append(ys)
        
        cum_dist = np.cumsum(ys)
        cum_dist /= cum_dist[-1]
        
        plt.plot(xs,cum_dist,label=FNAMES[i],linestyle="--")
        
    #yvals1 = lognormal(xvals,A,*args1)
    #yvals2 = halflognormal(xvals,A,*args2)
    #yvals3=logconstant(xvals,A,*args3)
    
    # modify the expected distribution by the completeness
    #completeness = get_completeness(xvals,cxvals,cyvals)
    #modyvals1 = yvals1*completeness
    #modyvals2 = yvals2*completeness
    #modyvals3 = yvals3*completeness
    
    # make a cdf
    
    #cum_dist1 = np.cumsum(modyvals1)
    #cum_dist1 /= cum_dist1[-1]
    
    #cum_dist2 = np.cumsum(modyvals2)
    #cum_dist2 /= cum_dist2[-1]
    
    #cum_dist3 = np.cumsum(modyvals3)
    #cum_dist3 /= cum_dist3[-1]
    
    
    #plt.plot(xvals,cum_dist1,label="Lognormal fit",linestyle="--")
    #plt.plot(xvals,cum_dist2,label="Half-lognormal fit",linestyle=":")
    #plt.plot(xvals,cum_dist3,label="Log-uniform fit",linestyle="-.")
        
    # makes a function of completeness
    xvals,yvals = make_completeness_plot(taus,reverse=False)
    plt.plot(xvals,yvals,label="Observed",color="black")
        
    plt.legend()
    plt.tight_layout()
    plt.savefig(outfile)
    plt.close()
    
    return

def cdf(xs):
    """
    cumulative distribution function used for KS test
    
    This is governed by global variables because ks_1samp
    doesn't allow other arguments ot be passed to this function
    """
    global xc # x values for cumulative distribution
    global yc # y values for cumnulative distribution
    
    cs = np.zeros([xs.size])
    # gets the values of the cdf at a given point
    for i,x in enumerate(xs):
        ix1 = np.where(x > xc)[0][-1]
        ix2 = ix1+2
        dx = xc[ix2] - xc[ix1]
        kx2 =  (x - xc[ix1])/dx
        kx1 = 1.-kx2
        c = kx1*yc[ix1] + kx2*yc[ix2]
        cs[i] = c
    return cs
    

    
def get_completeness(x,cx,cy):
    """
    Looks up the value of completeness for data values x,
    where completeness at values cx is defined by cy
    """
    N = x.size
    comp = np.zeros([N])
    for i in np.arange(N):
        # gets first value where completeness x is larger than observed
        ix = np.where(cx > x[i])[0][0]
        # gets completeness at that value
        comp[i] = cy[ix]
    return comp

def make_completeness_plot(data,reverse=True):
    """
    gets x,y values for a completeness plot
    this is just like a cumulative distribution,
    but reversed
    """
    xvals,yvals = misc_functions.make_cum_dist(data)
    
    xvals = np.concatenate((np.array([1e-5]),xvals,np.array([1e5])))
    yvals = np.concatenate((np.array([0.]),yvals,np.array([1.])))
    
    if reverse:
        yvals = 1.-yvals
    
    
    return xvals,yvals
    
def find_max_tau(tauobs,w,snrobs,fbar,DMobs,tres,nu_res = 1.,\
                SNRthresh=9.,max_iwidth=12, plot=False):
    """
    Finds the maximum value of tau which is probed by this FRB,
    assuming some basic calculations.
    
    Args:
        tauobs (float):     scattering time [ms]
        w (float):          total width, including scattering, in ms
        snrobs (float):        signal to noise ratio at detection
        fbar (float):       mean frequecy in MHz
        DM (float):         dispersion measure, in pc/cm3
        tres (float):       time resolution in ms
        nu_res (float):     frequency resolution in MHz
        SNR_thresh (float): threshold signal-to-noise for detection
        max_iwidth (float): Maximum multiple of tres for width search
        plot (bool):        If True, make a plot
    Returns:
        Maximum value of tau which is probed by this FRB
    """
    
    tauvals = np.linspace(tauobs,100.,1001)
    
    k_DM = 4.149 # leading constant: ms per pccc GHz^2
    dmsmear = 2*(nu_res/1.e3)*k_DM*DMobs/(fbar/1e3)**3
    
    
    # square of intrinsic widht. This *can* be negative! Obviously
    # not physically, by we assume width scales as the below, and
    # forcing it to be positive can artificially inflate the total width
    intrinsic_width_squared = (w**2 - (tauobs/0.816)**2)
    simulated_widths_squared = intrinsic_width_squared + (0.816*tauvals)**2
    simulated_widths = simulated_widths_squared**0.5
    
    # maximum width that is less than our cutoff of 12 times the integration time
    
    
    imaxtau1 = np.where(simulated_widths < tres*(max_iwidth+0.5))[0][-1]
    
    totalw_obs_square = (w**2 + dmsmear**2/3. + tres**2/3.)
    totalw_square = (simulated_widths_squared + dmsmear**2/3. + tres**2/3.)
    
    SNR = snrobs * (totalw_obs_square/totalw_square)**0.25
    imaxtau2 = np.where(SNR > SNRthresh)[0][-1]
    
    imaxtau = min(imaxtau1,imaxtau2)
    maxtau = tauvals[imaxtau]
    
    
    if plot:
        plt.figure()
        plt.plot(tauvals,SNR, label="Simulated S/N")
        plt.plot(tauvals,simulated_widths,label="scat + widths [ms]")
        plt.xlabel("Scattering time [ms]")
        plt.legend()
        plt.tight_layout()
        plt.show()
        plt.close()
    
    return maxtau

def find_max_w(tauobs,wtot,snrobs,fbar,DMobs,tres,nu_res = 1.,\
                SNRthresh=9.,max_iwidth=12, plot=False):
    """
    Finds the maximum value of tau which is probed by this FRB,
    assuming some basic calculations.
    
    Args:
        tauobs (float):     scattering time [ms]
        w (float):          total width, including scattering, in ms
        snrobs (float):        signal to noise ratio at detection
        fbar (float):       mean frequecy in MHz
        DM (float):         dispersion measure, in pc/cm3
        tres (float):       time resolution in ms
        nu_res (float):     frequency resolution in MHz
        SNR_thresh (float): threshold signal-to-noise for detection
        max_iwidth (float): Maximum multiple of tres for width search
        plot (bool):        If True, make a plot
    Returns:
        Maximum value of wi which is probed by this FRB
    """
    
    
    
    k_DM = 4.149 # leading constant: ms per pccc GHz^2
    dmsmear = 2*(nu_res/1.e3)*k_DM*DMobs/(fbar/1e3)**3
    
    
    # square of intrinsic widht. This *can* be negative! Obviously
    # not physically, by we assume width scales as the below, and
    # forcing it to be positive can artificially inflate the total width
    intrinsic_width_squared = (wtot**2 - (tauobs/0.816)**2)
    
    intrinsic_width_squared = max(0,intrinsic_width_squared)
    intrinsic_width = intrinsic_width_squared**0.5
    
    # simulated widths = begin by adding times, min is zero
    simulated_intrinsic_widths = np.linspace(intrinsic_width, 100.,1001) # adds up to 100 ms to this
    
    # simulated total width square
    simulated_widths_squared = simulated_intrinsic_widths**2 + (tauobs/0.816)**2
    simulated_widths = simulated_widths_squared**0.5
    
    # maximum width that is less than our cutoff of 12 times the integration time
    
    imaxw1 = np.where(simulated_widths < tres*(max_iwidth+0.5))[0][-1]
    
    totalw_obs_square = (wtot**2 + dmsmear**2/3. + tres**2/3.)
    totalw_square = (simulated_widths_squared + dmsmear**2/3. + tres**2/3.)
    
    SNR = snrobs * (totalw_obs_square/totalw_square)**0.25
    imaxw2 = np.where(SNR > SNRthresh)[0][-1]
    
    imaxw = min(imaxw1,imaxw2)
    maxw = simulated_intrinsic_widths[imaxw]
    
    
    if plot:
        plt.figure()
        plt.plot(simulated_intrinsic_widths,SNR, label="Simulated S/N")
        plt.plot(simulated_intrinsic_widths,simulated_widths,label="scat + widths [ms]")
        plt.xlabel("Intrinsic width [ms]")
        plt.legend()
        plt.tight_layout()
        plt.show()
        plt.close()
    
    return maxw
    
def get_data():
    # extract relevant data.
    # this is the error code
    
    
    tresinfo = np.loadtxt("treslist.dat",dtype="str")
    names=tresinfo[:,0]
    
    for i in np.arange(names.size):
        names[i] = names[i][0:8] # truncates the letter
    
    dataframe = pandas.read_csv("CRAFT_ICS_HTR_Catalogue1.csv")
    
    
    ERR=9999.
    tns = dataframe.TNS
    tauobs = dataframe.TauObs
    w95 = dataframe.W95
    wsnr = dataframe.Wsnr
    z = dataframe.Z
    snr = dataframe.SNdet
    freq = dataframe.NUTau
    DM = dataframe.DM
    
    # check which FRBs do not have nay errors in all columns
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
            exit()
        
        tres[i] = float(tresinfo[j[0],1])
    
    # cats everything to numpy arrays, because pands throws
    # some bonkers errors I can't debug
    tns = np.array(tns)
    tauobs = np.array(tauobs)
    w95 = np.array(w95)
    wsnr = np.array(wsnr)
    z = np.array(z)
    snr = np.array(snr)
    freq = np.array(freq)
    DM = np.array(DM)
    tres = np.array(tres)
    
    return tns,tauobs,w95,wsnr,z,snr,freq,DM,tres



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
