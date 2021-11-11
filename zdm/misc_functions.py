import os
import sys
import numpy as np
import argparse
import pickle
import json
import copy
from numpy.core.fromnumeric import mean

import scipy as sp

import matplotlib.pyplot as plt
import matplotlib

from frb import dlas
from frb.dm import igm
from frb.dm import cosmic

import time
from zdm import iteration as it
from zdm import beams
from zdm import cosmology as cos
from zdm import survey
from zdm import grid as zdm_grid
from zdm import pcosmic
from zdm import parameters


def marginalise(pset,grids,surveys,which,vals,disable=None,psnr=True,PenTypes=None,PenParams=None,Verbose=False,steps=None):
    """
    Calculates limits for a single variable
    """
    t0=time.process_time()
    t1=t0
    lls=np.zeros([vals.size])
    psets=[]
    if disable is not None:
        disable.append(which)
    else:
        disable=[which]
    steps=np.full([8],0.5)
    for i,v in enumerate(vals):
        pset[which]=v
        print("Setting parameter ",which," = ",v)
        
        C_ll,C_p=it.my_minimise(pset,grids,surveys,disable=disable,psnr=psnr,PenTypes=PenTypes,PenParams=PenParams,Verbose=False,steps=steps)
        steps=np.full([8],0.1)
        print(i,v,C_ll,pset)
        t1=time.process_time()
        psets.append(C_p)
        lls[i]=C_ll
        t2=time.process_time()
        print("Iteration ",i," took ",t2-t1," seconds")
        t1=t2
    print("Done - total time ",t1-t0," seconds")
    psets=np.array(psets)
    np.save('Marginalise1D/'+str(which)+'_lls.npy',lls)
    np.save('Marginalise1D/'+str(which)+'_psets.npy',psets)


def get_source_counts(grid,plot=None,Slabel=None):
    """
    Calculates the source-counts function for a given grid
    It does this in terms of p(SNR)dSNR
    """
    # this is closely related to the likelihood for observing a given psnr!
    
    # calculate vector of grid thresholds
    Emax=grid.Emax
    Emin=grid.Emin
    gamma=grid.gamma
    
    nsnr=71
    snrmin=0.001
    snrmax=1000.
    ndm=grid.dmvals.size
    snrs=np.logspace(0,2,nsnr) # histogram array of values for s=SNR/SNR_th
    
    # holds cumulative and differential source counts
    cpsnrs=np.zeros([nsnr])
    psnrs=np.zeros([nsnr-1])
    
    # holds DM-dependent source counts
    dmcpsnrs=np.zeros([nsnr,ndm])
    dmpsnrs=np.zeros([nsnr-1,ndm])
    
    backup1=np.copy(grid.thresholds)
    Emin=grid.Emin
    Emax=grid.Emax
    gamma=grid.gamma
    
    # modifies grid to simplify beamshape
    grid.beam_b=np.array([grid.beam_b[-1]])
    grid.beam_o=np.array([grid.beam_o[-1]])
    grid.b_fractions=None
    
    for i,s in enumerate(snrs):
        
        grid.thresholds=backup1*s
        grid.calc_pdv(Emin,Emax,gamma)
        grid.calc_rates()
        rates=grid.rates
        dmcpsnrs[i,:]=np.sum(rates,axis=0)
        cpsnrs[i]=np.sum(dmcpsnrs[i,:])
    
    # the last one contains cumulative values
    for i,s in enumerate(snrs):
        if i==0:
            continue
        psnrs[i-1]=cpsnrs[i-1]-cpsnrs[i]
        dmpsnrs[i-1,:]=dmcpsnrs[i-1,:]-dmcpsnrs[i,:]
    
    mod=1.5
    snrs=snrs[:-1]
    imid=int((nsnr+1)/2)
    xmid=snrs[imid]
    ymid=psnrs[imid]
    slopes=np.linspace(1.3,1.7,5)
    ys=[]
    for i,s in enumerate(slopes):
        ys.append(ymid*xmid**s*snrs**-s)
    
    if plot is not None:
        fixpoint=ys[0][0]*snrs[0]**mod
        plt.figure()
        plt.xscale('log')
        plt.yscale('log')
        plt.ylim(1,3)
        plt.xlabel('$s=\\frac{\\rm SNR}{\\rm SNR_{\\rm th}}$')
        plt.ylabel('$p(s) s^{1.5} d\\,\\log(s)$ [a.u.]')
        plt.plot(snrs,psnrs*snrs**mod/fixpoint,label='Prediction ('+Slabel+')',color='black',linewidth=2) # this is in relative units
        for i,s in enumerate(slopes):
            plt.plot(snrs,ys[i]*snrs**mod/fixpoint,label='slope='+str(s)[0:3])
        ax=plt.gca()
        #labels = [item.get_text() for item in ax.get_yticklabels()]
        #print("Labels are ",labels)
        #labels[0] = '1'
        #labels[1] = '2'
        #labels[2] = '3'
        #ax.set_yticklabels(labels)
        ax.set_yticks([1,2,3])
        ax.set_yticklabels(['1','2','3'])
        plt.legend(fontsize=12)#,loc=[6,8])
        plt.tight_layout()
        plt.savefig(plot)
        plt.close()
    return snrs,psnrs,dmpsnrs
    
def get_test_pks_surveys():
    
    # load Parkes data
    # generates a set of surveys with fake central beam histograms
    pksa=survey.survey()
    pksa.process_survey_file('Surveys/parkes_mb.dat')
    pksa.meta["BEAM"]="a_b0"
    pksa.init_beam(method=3,plot=True) # need more bins for Parkes!
    
    pkse=survey.survey()
    pkse.process_survey_file('Surveys/parkes_mb.dat')
    pkse.meta["BEAM"]="e_b0"
    pkse.init_beam(method=3,plot=True) # need more bins for Parkes!
    
    pksk=survey.survey()
    pksk.process_survey_file('Surveys/parkes_mb.dat')
    pksk.meta["BEAM"]="k_b0"
    pksk.init_beam(method=3,plot=True) # need more bins for Parkes!
    
    pksh=survey.survey()
    pksh.process_survey_file('Surveys/parkes_mb.dat')
    pksh.meta["BEAM"]="h_b0"
    pksh.init_beam(method=3,plot=True) # need more bins for Parkes!
    
    pksl=survey.survey()
    pksl.process_survey_file('Surveys/parkes_mb.dat')
    pksl.meta["BEAM"]="l_b0"
    pksl.init_beam(method=3,plot=True) # need more bins for Parkes!
    
    surveys=[pksa,pkse,pksk,pksh,pksl]
    return surveys

    
def do_single_errors(grids,surveys,pset,outdir):
    """ iterates over sensible ranges of all single-parameter errors """
    
    # for each parameter, investigate the best-fit as a function of range
    # in each case, we fix the parameter at the value
    # then we let the optimisation go while holding it fixed
    
    # we now set the base ranges
    fig1=plt.figure()
    plt.xlabel('Relative variation')
    plt.ylabel('log-likelihood')
    
    ### Emax ###
    which=1
    rels=np.linspace(-1,1,3)
    Emaxes=pset[1]*10**rels
    delta=0.1
    lls1,psets1=one_parameter_error_range(grids,surveys,pset,which,delta,crit=0.5) #0.5 is about 1 sigma
    opdir=outdir+it.get_names(1)+'/'
    if not os.path.exists(opdir):
        os.mkdir(opdir)
    savename=opdir+'correlation_'+it.get_lnames(1)+'.pdf'
    do_correlation_plots(Emaxes,lls1,psets1,[0,1],savename) # tells it that parameters 0 and 1 are not to be plotted
    
    plt.plot(rels,lls1,label=it.get_lnames(1))
    
    
    plt.tight_layout()
    plt.savefig(outdir+'varying_likelihoods.pdf')
    plt.close()
    

def do_correlation_plots(vals,lls,psets,const,savename):
    """ Plots correlations of different variables """
    
    plt.figure()
    nv,np=psets.shape()
    for i in np.arange(np):
        if i in const:
            continue
        plt.plot(vals,psets[:,i],label=it.get_lnames(i))
    plt.savefig(savename)
    
def one_parameter_error_range(grids,surveys,pset,which,delta,crit=0.5):
    """ Investigates a range of errors for each parameter in 1D only
    which is which parameter to investigate
    rels are the list of relative
    """
    
    # keep original pset
    tpset=np.copy(pset)
    #lls=np.zeros([values.size])
    #sets=np.zeros([values.size,pset.size])
    
    lls=[]
    vals=[]
    psets=[pset]
    print("About to minimise...")
    ll,ps=it.my_minimise(tpset,grids,surveys,disable=[0,which])
    lls=[ll]
    psets=[ps]
    ll0=ll
    llcrit=ll0-crit
    vals=[tpset[which]]
    
    print("Found initial minimum at ",ll,ps)
    
    tpset[3]=0.1
    
    # goes down
    while(ll > llcrit):
        tpset[which] -= delta
        t0=time.process_time()
        ll,ps=it.my_minimise(tpset,grids,surveys,disable=[0,which])
        t1=time.process_time()
        lls.insert(0,ll)
        psets.insert(0,ps)
        vals.insert(0,tpset[which])
        print("In time ",t1-t0," values now ",ll,ps)
        if ll==0.:
            break # means parameter are meaningless
    
    # resets
    tpset = pset
    ll=ll0
    
    # goes up
    while(ll > llcrit):
        tpset[which] += delta
        t0=time.process_time()
        ll,ps=it.my_minimise(tpset,grids,surveys,disable=[0,which])
        t1=time.process_time()
        lls.append(ll)
        psets.append(ps)
        vals.append(tpset[which])
        print("In time ",t1-t0," values now ",ll,ps)
        if ll==0.:
            break # means we got an nan and parameters are meaningless
    
    return lls,psets
        
def get_zgdm_priors(grid,survey,savename):
    """ Plots priors as a function of redshift for each FRB in the survey
    Likely outdated, should use the likelihoods function.
    """
    priors=grid.get_p_zgdm(survey.DMEGs)
    plt.figure()
    plt.xlabel('$z$')
    plt.ylabel('$p(z|{\\rm DM})$')
    for i,dm in enumerate(survey.DMs):
        if i<10:
            style="-"
        else:
            style=":"
        plt.plot(grid.zvals,priors[i,:],label=str(dm),linestyle=style)
    plt.xlim(0,0.5)
    plt.legend(fontsize=8,ncol=2)
    plt.tight_layout()
    plt.savefig(savename)
    plt.close()

def make_dm_redshift(grid,savename="",DMmax=1000,
                     zmax=1,loc='upper left',Macquart=None,
                     H0=None,showplot=False):
    ''' generates full dm-redhsift (Macquart) relation '''
    if H0 is None:
        H0 = cos.cosmo.H0
    ndm=1000
    cvs=[0.025,0.16,0.5,0.84,0.975]
    nc=len(cvs)
    names=['$2\\sigma$','$1\\sigma$','Median','','']
    styles=[':','--','-','--',':']
    colours=["white","white","black","white","white"]
    DMs=np.linspace(DMmax/ndm,DMmax,ndm,endpoint=True)
    priors=grid.get_p_zgdm(DMs)
    zvals=grid.zvals
    means=np.mean(priors,axis=1)
    csums=np.cumsum(priors,axis=1)
    
    crits=np.zeros([nc,ndm])
    
    for i in np.arange(ndm):
        for j,c in enumerate(cvs):
            ic=np.where(csums[i]>c)[0][0]
            if ic>0:
                kc=(csums[i,ic]-c)/(csums[i,ic]-csums[i,ic-1])
                crits[j,i]=zvals[ic]*(1-kc)+zvals[ic-1]*kc
            else:
                crits[j,i]=zvals[ic]
    
    # now we convert this between real values and integer units
    dz=zvals[1]-zvals[0]
    crits /= dz
    
    ### concatenate for plotting ###
    delete=np.where(zvals > zmax)[0][0]
    plotpriors=priors[:,0:delete]
    plotz=zvals[0:delete]
    
    plt.figure()
    
    ############# sets the x and y tics ################3
    ytvals=np.arange(plotz.size)
    every=int(plotz.size/5)
    ytickpos=np.insert(ytvals[every-1::every],[0],[0])
    yticks=np.insert(plotz[every-1::every],[0],[0])
    
    #plt.yticks(ytvals[every-1::every],plotz[every-1::every])
    plt.yticks(ytickpos,yticks)
    xtvals=np.arange(ndm)
    everx=int(ndm/5)
    xtickpos=np.insert(xtvals[everx-1::everx],[0],[0])
    xticks=np.insert(DMs[everx-1::everx],[0],[0])
    plt.xticks(xtickpos,xticks)
    #plt.xticks(xtvals[everx-1::everx],DMs[everx-1::everx])
    
    ax=plt.gca()
    labels = [item.get_text() for item in ax.get_xticklabels()]
    for i in np.arange(len(labels)):
        thisl=len(labels[i])
        labels[i]=labels[i][0:thisl-1]
    ax.set_xticklabels(labels)
    
    #### rescales priors to max value for visibility's sake ####
    dm_max=np.max(plotpriors,axis=1)
    for i in np.arange(ndm):
        plotpriors[i,:] /= np.max(plotpriors[i,:])
    
    
    
    cmx = plt.get_cmap('cubehelix')
    plt.xlabel('${\\rm DM}_{\\rm EG}$')
    plt.ylabel('z')
    
    aspect=float(ndm)/plotz.size
    plt.imshow(plotpriors.T,origin='lower',cmap=cmx,aspect=aspect)
    cbar=plt.colorbar()
    cbar.set_label('$p(z|{\\rm DM})/p_{\\rm max}(z|{\\rm DM})$')
    ###### now we plot the specific thingamies #######
    for i,c in enumerate(cvs):
        plt.plot(np.arange(ndm),crits[i,:],linestyle=styles[i],label=names[i],color=colours[i])
    
    #Macquart=None
    if Macquart is not None:
        plt.ylim(0,ytvals.size)
        nz=zvals.size
        
        plt.xlim(0,xtvals.size)
        zmax=zvals[-1]
        DMbar, zeval = igm.average_DM(zmax, cumul=True, neval=nz+1)
        DMbar = DMbar*H0/(cos.DEF_H0)  # NOT SURE THIS IS RIGHT
        DMbar=np.array(DMbar)
        DMbar += Macquart #should be interpreted as muDM
        
        
        #idea is that 1 point is 1, hence...
        zeval /= (zvals[1]-zvals[0])
        DMbar /= (DMs[1]-DMs[0])
        
        plt.plot(DMbar,zeval,linewidth=2,label='Macquart',color='blue')
        #l=plt.legend(loc='lower right',fontsize=12)
        #l=plt.legend(bbox_to_anchor=(0.2, 0.8),fontsize=8)
        #for text in l.get_texts():
            #	text.set_color("white")
    
    #plt.plot([30,40],[0.5,10],linewidth=10)
    
    plt.legend(loc=loc)
    plt.savefig(savename)
    if H0 is not None:
        plt.title("H0 " + str(H0))
    if showplot:
        plt.show()
    plt.close()



def fit_width_test(pset,surveys,grids,names):
    x0=[2.139,0.997]
    x0=[1.7712265624999926, 0.9284453124999991]
    args=[pset,surveys,grids,names]
    result=min_wt(x0,args)
    print("result of fit is ",result)
    

def min_wt(x,args):
    
    logmean=x[0]
    logsigma=x[1]
    
    pset=args[0]
    surveys=args[1]
    grids=args[2]
    names=args[3]
    
    oldchi2=1e10
    dlogmean=0.1
    dlogsigma=0.1
    
    for i in np.arange(10):
        
        while(True):
            logmean -= dlogmean
            W,C=basic_width_test(pset,[surveys[0]],[grids[0]],logmean,logsigma)
            chi2=np.sum((W-C)**2)
            if chi2 > oldchi2:
                logmean += dlogmean
                break
            else:
                oldchi2=chi2
        while(True):
            logmean += dlogmean
            W,C=basic_width_test(pset,[surveys[0]],[grids[0]],logmean,logsigma)
            chi2=np.sum((W-C)**2)
            if chi2 > oldchi2:
                logmean -= dlogmean
                break
            else:
                oldchi2=chi2
        while(True):
            logsigma += dlogsigma
            W,C=basic_width_test(pset,[surveys[0]],[grids[0]],logmean,logsigma)
            chi2=np.sum((W-C)**2)
            if chi2 > oldchi2:
                logsigma -= dlogsigma
                break
            else:
                oldchi2=chi2
        while(True):
            logsigma -= dlogsigma
            W,C=basic_width_test(pset,[surveys[0]],[grids[0]],logmean,logsigma)
            chi2=np.sum((W-C)**2)
            if chi2 > oldchi2:
                logsigma += dlogsigma
                break
            else:
                oldchi2=chi2
        dlogsigma /= 2.
        dlogmean /= 2.
        print(i,logmean,logsigma,chi2)
    return logmean,logsigma,chi2


def basic_width_test(pset,surveys,grids,logmean=2,logsigma=1):
    """ Tests the effects of intrinsic widths on FRB properties """
    
    IGNORE=0. # a parameter that gets ignored
    
    ############ set default parameters for width distribution ###########
    # 'real' version
    wmin=0.1
    wmax=20
    NW=200
    #short version
    #wmin=0.1
    #wmax=50
    #NW=100
    dw=(wmax-wmin)/2.
    widths=np.linspace(wmin,wmax,NW)
    
    probs=pcosmic.linlognormal_dlin(widths,[logmean,logsigma]) #not integrating, just amplitudes
    # normalise probabilities
    probs /= np.sum(probs)
    
    MAX=1
    norm=MAX/np.max(probs)
    probs *= norm
    
    Emin=10**pset[0]
    Emax=10**pset[1]
    alpha=pset[2]
    gamma=pset[3]
    NS=len(surveys)
    rates=np.zeros([NS,NW])
    rp=np.zeros([NS,NW])
    
    #calculating total rate compared to what is expected for ~width 0
    sumw=0.
    DMvals=grids[0].dmvals
    zvals=grids[0].zvals
    
    dmplots=np.zeros([NS,NW,DMvals.size])
    zplots=np.zeros([NS,NW,zvals.size])
    
    #### does this for width distributions #######
    
    for i,s in enumerate(surveys):
        g=grids[i]
        fbar=s.meta['FBAR']
        tres=s.meta['TRES']
        fres=s.meta['FRES']
        thresh=s.meta['THRESH']
        ###### "Full" version ######
        for j,w in enumerate(widths):
            # artificially set response function
            sens=survey.calc_relative_sensitivity(IGNORE,DMvals,w,fbar,tres,fres,model='Quadrature',dsmear=False)
            
            g.calc_thresholds(thresh,sens,alpha=alpha)
            g.calc_pdv(Emin,Emax,gamma)
            
            g.calc_rates()
            rates[i,j]=np.sum(g.rates)
            dmplots[i,j,:]=np.sum(g.rates,axis=0)
            zplots[i,j,:]=np.sum(g.rates,axis=1)
        
        rates[i,:] /= rates[i,0]
        
        #norm_orates[i,:] = orates[i,0]/rates[i,0]
        rp[i,:]=rates[i,:]*probs
        
        rp[i,:]*=MAX/np.max(rp[i,:])
    
    # shows the distribution of widths using Wayne's method
    WAlogmean=np.log(2.67)
    WAlogsigma=np.log(2.07)
    
    waprobs=pcosmic.linlognormal_dlin(widths,[WAlogmean,WAlogsigma])
    wasum=np.sum(waprobs)
    
    #rename
    WAprobs = waprobs*MAX/np.max(waprobs)
    
    return WAprobs,rp[0,:] #for lat50

def width_test(pset,surveys,grids,names,logmean=2,logsigma=1,plot=True,outdir='Plots/',NP=5,scale=3.5):
    """ Tests the effects of intrinsic widths on FRB properties 
    Considers three cases:
        - width distribution of Wayne Arcus et al (2020)
        - width distribution specified by user (logmean, logsigma)
        - "practical" width distribution with a few width parameters
        - no width distribution (i.e. for width = 0)
    
    
    """
    
    if plot:
        print("Performing test of intrinsic width effects")
        t0=time.process_time()
    
    IGNORE=0. # a parameter that gets ignored
    
    ############ set default parameters for width distribution ###########
    # 'real' version
    wmin=0.1
    wmax=30
    NW=300
    #short version
    #wmin=0.1
    #wmax=50
    #NW=100
    dw=(wmax-wmin)/2.
    widths=np.linspace(wmin,wmax,NW)
    
    probs=pcosmic.linlognormal_dlin(widths,logmean,logsigma) #not integrating, just amplitudes
    # normalise probabilities
    probs /= np.sum(probs)
    pextra,err=sp.integrate.quad(pcosmic.loglognormal_dlog,np.log(wmax+dw/2.),np.log(wmax*2),args=(logmean,logsigma))
    probs *= (1.-pextra) # now sums to 1.-pextra
    probs[-1] += pextra # now sums back to 1
    
    styles=['--','-.',':']
    
    MAX=1
    norm=MAX/np.max(probs[:-1])
    probs *= norm
    wsum=np.sum(probs)
    if plot:
        plt.figure()
        plt.xlabel('w [ms]')
        plt.ylabel('p(w)')
        plt.xlim(0,wmax)
        plt.plot(widths[:-1],probs[:-1],label='This work: $\\mu_w=5.49, \\sigma_w=2.46$',linewidth=2)
    
    Emin=10**pset[0]
    Emax=10**pset[1]
    alpha=pset[2]
    gamma=pset[3]
    NS=len(surveys)
    rates=np.zeros([NS,NW])
    rp=np.zeros([NS,NW])
    warp=np.zeros([NS,NW])
    #loop over surveys
    #colours=['blue','orange','
    
    names=['ASKAP/FE','ASKAP/ICS','Parkes/MB']
    colours=['blue','red','green','orange','black']
    
    #calculating total rate compared to what is expected for ~width 0
    sumw=0.
    DMvals=grids[0].dmvals
    zvals=grids[0].zvals
    
    dmplots=np.zeros([NS,NW,DMvals.size])
    zplots=np.zeros([NS,NW,zvals.size])
    
    ##### values for 'practical' arrays #####
    
    #NP=5 #NP 10 at scale 2 good
    #scale=3.5
    
    pdmplots=np.zeros([NS,NP,DMvals.size])
    pzplots=np.zeros([NS,NP,zvals.size])
    prates=np.zeros([NS,NP])
    
    # collapsed over width dimension with appropriate weights
    spdmplots=np.zeros([NS,DMvals.size])
    spzplots=np.zeros([NS,zvals.size])
    sprates=np.zeros([NS])
    
    ######## gets original rates for DM and z distributions #########
    #norm_orates=([NS,zvals.size,DMvals.size) # normed to width=0!
    # wait - does this include beamshape and the others not?
    orates=np.zeros([NS])
    norates=np.zeros([NS]) #for normed version
    odms=np.zeros([NS,DMvals.size])
    ozs=np.zeros([NS,zvals.size])
    for i,g in enumerate(grids):
        odms[i,:]=np.sum(g.rates,axis=0)
        ozs[i,:]=np.sum(g.rates,axis=1)
        orates[i]=np.sum(g.rates) #total rate for grid - 'original' rates
    
    ############ Wayne Arcus's fits ##########3
    # calculates probabilities and uses this later; WAprobs
    WAlogmean=np.log(2.67)
    WAlogsigma=np.log(2.07)
    waprobs=pcosmic.linlognormal_dlin(widths,WAlogmean,WAlogsigma)
    waprobs /= np.sum(waprobs)
    pextra,err=sp.integrate.quad(pcosmic.loglognormal_dlog,np.log(wmax+dw/2.),np.log(wmax*2),args=(WAlogmean,WAlogsigma))
    waprobs *= (1.-pextra) # now sums to 1.-pextra
    waprobs[-1] += pextra # now sums back to 1
    wasum=np.sum(waprobs)
    
    #rename
    WAprobs = waprobs*MAX/np.max(waprobs)
    WAsum=np.sum(WAprobs)
    #print(np.max(rates[0,:]),np.max(WAprobs))
    ls=['-','--',':','-.','-.']
    
    
    
    #### does this for width distributions #######
    
    for i,s in enumerate(surveys):
        g=grids[i]
        #DMvals=grids[i].dmvals
        
        # gets the 'practical' widths for this survey
        pwidths,pprobs=survey.make_widths(s,logmean,logsigma,NP,scale=scale)
        
        pnorm_probs = pprobs / np.max(pprobs)
        
        #if plot:
        #    plt.plot(pwidths,pnorm_probs,color=colours[i],marker='o',linestyle='',label='Approx.')
        # gets the survey parameters
        fbar=s.meta['FBAR']
        tres=s.meta['TRES']
        fres=s.meta['FRES']
        thresh=s.meta['THRESH']
        
        
        ######## "practical" version ### (note: not using default behaviour) ########
        for j,w in enumerate(pwidths):
            # artificially set response function
            sens=survey.calc_relative_sensitivity(IGNORE,DMvals,w,fbar,tres,fres,model='Quadrature',dsmear=False)
            g.calc_thresholds(thresh,sens,alpha=alpha)
            g.calc_pdv(Emin,Emax,gamma)
            
            g.calc_rates()
            prates[i,j]=np.sum(g.rates)*pprobs[j]
            pdmplots[i,j,:]=np.sum(g.rates,axis=0)*pprobs[j]
            pzplots[i,j,:]=np.sum(g.rates,axis=1)*pprobs[j]
        #sum over weights - could just do all this later, but whatever
        sprates[i]=np.sum(prates[i],axis=0)
        spdmplots[i]=np.sum(pdmplots[i],axis=0)
        spzplots[i]=np.sum(pzplots[i],axis=0)
        
        ######### "Full" (correct) version #########
        for j,w in enumerate(widths):
            # artificially set response function
            sens=survey.calc_relative_sensitivity(IGNORE,DMvals,w,fbar,tres,fres,model='Quadrature',dsmear=False)
            
            g.calc_thresholds(thresh,sens,alpha=alpha)
            g.calc_pdv(Emin,Emax,gamma)
            
            g.calc_rates()
            rates[i,j]=np.sum(g.rates)
            
            dmplots[i,j,:]=np.sum(g.rates,axis=0)
            zplots[i,j,:]=np.sum(g.rates,axis=1)
        
        # this step divides by the full rates for zero width
        norates[i]=orates[i]/rates[i,0] #normalises original weight by rate if no width
        sprates[i] /= rates[i,0]
        rates[i,:] /= rates[i,0]
        
        #norm_orates[i,:] = orates[i,0]/rates[i,0]
        rp[i,:]=rates[i,:]*probs
        warp[i,:]=rates[i,:]*WAprobs
        
        if plot:
            plt.plot(widths[:-1],rp[i,:-1],linestyle=styles[i],linewidth=1)
            
        norm=MAX/np.max(rp[i,:-1])
        if plot:
            plt.plot(widths[:-1],rp[i,:-1]*norm,label=names[i],linestyle=styles[i],color=plt.gca().lines[-1].get_color(),linewidth=2)
    
    print("The total fraction of events detected as a function of experiment are")
    print("Survey  name   [input_grid]   WA   lognormal     practical")
    for i,s in enumerate(surveys):
        print(i,names[i],norates[i],np.sum(warp[i,:])/WAsum,np.sum(rp[i,:])/wsum,sprates[i])
        #print(i,rates[i,:])
        #print(i,names[i],np.sum(rates[i,:]),np.sum(rp[i,:]),wsum,np.sum(rp[i,:])/wsum)
    
    
    
    if plot:
        plt.plot(widths[:-1],WAprobs[:-1],label='Arcus et al: $\\mu_w=2.67, \\sigma_w=2.07$',color='black',linestyle='-',linewidth=2)
        
        plt.legend(loc='upper right')
        plt.tight_layout()
        plt.xlim(0,30)
        plt.savefig(outdir+'/width_effect.pdf')
        plt.close()
        t1=time.process_time()
        print("Done. Took ",t1-t0," seconds.")
    
    
        #### we now do DM plots ###
        plt.figure()
        plt.xlabel('DM [pc cm$^{-3}$]')
        plt.ylabel('p(DM) [a.u.]')
        plt.xlim(0,3000)
        #dmplots[i,j,:]=np.sum(g.rates,axis=0)
        
        twdm=np.zeros([NS,DMvals.size])
        wadm=np.zeros([NS,DMvals.size])
        w0dm=np.zeros([NS,DMvals.size]) 
        for i,s in enumerate(surveys):
            
            w0dm[i]=dmplots[i,0,:]
            wadm[i]=np.sum((waprobs.T*dmplots[i,:,:].T).T,axis=0)/wasum
            twdm[i]=np.sum((probs.T*dmplots[i,:,:].T).T,axis=0)/wsum
            
            
            print("Mean DM for survey ",i," is (0) ",np.sum(DMvals*w0dm[i,:])/np.sum(w0dm[i,:]))
            print("                  (full verson) ",np.sum(DMvals*twdm[i,:])/np.sum(twdm[i,:]))
            print("                (wayne arcus a) ",np.sum(DMvals*wadm[i,:])/np.sum(wadm[i,:]))
            print("                    (practical) ",np.sum(DMvals*spdmplots[i,:])/np.sum(spdmplots[i,:]))
            
            #plt.plot(DMvals,w0dm[i]/np.max(w0dm[i]),label=names[i],linewidth=0.1)
            #plt.plot(DMvals,twdm[i]/np.max(twdm[i]),color=plt.gca().lines[-1].get_color(),linestyle='--')
            #plt.plot(DMvals,wadm[i]/np.max(wadm[i]),color=plt.gca().lines[-1].get_color(),linestyle='-.')
            #plt.plot(DMvals,odms[i]/np.max(odms[i]),color=plt.gca().lines[-1].get_color(),linestyle=':')
            #plt.plot(DMvals,spdmplots[i]/np.max(spdmplots[i]),color=plt.gca().lines[-1].get_color(),linestyle=':')
            if i==0:
                plt.plot(DMvals,w0dm[i]/np.max(w0dm[i]),linestyle=ls[0],label='$w_{\\rm inc}=0$',color=colours[0])
                #plt.plot(DMvals,wadm[i]/np.max(wadm[i]),linestyle=ls[2],label='Arcus et al: $\\mu_w=2.67, \\sigma_w=2.07$',color=colours[2])
                #plt.plot(DMvals,twdm[i]/np.max(twdm[i]),linestyle=ls[1],label='This work: $\\mu_w=5.49, \\sigma_w=2.46$',color=colours[1])
                plt.plot(DMvals,wadm[i]/np.max(wadm[i]),linestyle=ls[2],label='Arcus et al.',color=colours[2])
                plt.plot(DMvals,twdm[i]/np.max(twdm[i]),linestyle=ls[1],label='This work',color=colours[1])
                
                #plt.plot(DMvals,odms[i]/np.max(odms[i]),linestyle=ls[3],label='old',color=colours[3])
                plt.plot(DMvals,spdmplots[i]/np.max(spdmplots[i]),linestyle=ls[4],label='This work',color=colours[4])
            else:
                plt.plot(DMvals,w0dm[i]/np.max(w0dm[i]),linestyle=ls[0],color=colours[0])
                plt.plot(DMvals,twdm[i]/np.max(twdm[i]),linestyle=ls[1],color=colours[1])
                plt.plot(DMvals,wadm[i]/np.max(wadm[i]),linestyle=ls[2],color=colours[2])
                #plt.plot(DMvals,odms[i]/np.max(odms[i]),linestyle=ls[3],color=colours[3])
                plt.plot(DMvals,spdmplots[i]/np.max(spdmplots[i]),linestyle=ls[4],color=colours[4])
        plt.legend(loc='upper right')
        plt.tight_layout()
        plt.savefig(outdir+'/width_dm_effect.pdf')
        plt.close()
        
        ##### z plots ####
        plt.figure()
        plt.xlabel('z')
        plt.ylabel('p(z) [a.u.]')
        plt.xlim(0,3)
        #zplots[i,j,:]=np.sum(g.rates,axis=1)
        
        twz=np.zeros([NS,zvals.size])
        waz=np.zeros([NS,zvals.size])
        w0z=np.zeros([NS,zvals.size]) 
        for i,s in enumerate(surveys):
            
            w0z[i]=zplots[i,0,:]
            waz[i]=np.sum((waprobs.T*zplots[i,:,:].T).T,axis=0)/wasum
            twz[i]=np.sum((probs.T*zplots[i,:,:].T).T,axis=0)/wsum
            
            print("Mean z for survey ",i," is (0) ",np.sum(zvals*w0z[i])/np.sum(w0z[i]))
            print("                           (tw) ",np.sum(zvals*twz[i])/np.sum(twz[i]))
            print("                           (wa) ",np.sum(zvals*waz[i])/np.sum(waz[i]))
            print("                            (p) ",np.sum(zvals*spzplots[i,:])/np.sum(spzplots[i,:]))
            
            #plt.plot(zvals,w0z[i]/np.max(w0z[i]),label=names[i])
            #plt.plot(zvals,twz[i]/np.max(twz[i]),color=plt.gca().lines[-1].get_color(),linestyle='--')
            #plt.plot(zvals,waz[i]/np.max(waz[i]),color=plt.gca().lines[-1].get_color(),linestyle='-.')
            #plt.plot(zvals,ozs[i]/np.max(ozs[i]),color=plt.gca().lines[-1].get_color(),linestyle=':')
            #plt.plot(zvals,spzplots[i]/np.max(spzplots[i]),color=plt.gca().lines[-1].get_color(),linestyle=':')
            if i==0:
                plt.plot(zvals,w0z[i]/np.max(w0z[i]),label='$w_{\\rm inc}=0$',linestyle=ls[0],color=colours[0])
                #plt.plot(zvals,waz[i]/np.max(waz[i]),linestyle=ls[2],label='Arcus et al: $\\mu_w=2.67, \\sigma_w=2.07$',color=colours[2])
                #plt.plot(zvals,twz[i]/np.max(twz[i]),label='This work: $\\mu_w=5.49, \\sigma_w=2.46$',linestyle=ls[1],color=colours[1])
                plt.plot(zvals,waz[i]/np.max(waz[i]),linestyle=ls[2],label='Arcus et al.',color=colours[2])
                plt.plot(zvals,twz[i]/np.max(twz[i]),label='This work',linestyle=ls[1],color=colours[1])
                #plt.plot(zvals,ozs[i]/np.max(ozs[i]),linestyle=ls[3],label='orig',color=colours[3])
                plt.plot(zvals,spzplots[i]/np.max(spzplots[i]),linestyle=ls[4],label='This work',color=colours[4])
            else:
                plt.plot(zvals,w0z[i]/np.max(w0z[i]),linestyle=ls[0],color=colours[0])
                plt.plot(zvals,twz[i]/np.max(twz[i]),linestyle=ls[1],color=colours[1])
                plt.plot(zvals,waz[i]/np.max(waz[i]),linestyle=ls[2],color=colours[2])
                #plt.plot(zvals,ozs[i]/np.max(ozs[i]),linestyle=ls[3],color=colours[3])
                plt.plot(zvals,spzplots[i]/np.max(spzplots[i]),linestyle=ls[4],color=colours[4])
        plt.legend(loc='upper right')
        plt.tight_layout()
        plt.savefig(outdir+'/width_z_effect.pdf')
        plt.close()
    
    return WAprobs,rp[0,:] #for lat50

def test_pks_beam(surveys,zDMgrid, zvals,dmvals,pset,outdir='Plots/BeamTest/',zmax=1,DMmax=1000):
    
    if not os.path.isdir(outdir):
        os.mkdir(outdir)
    
    # get parameter values
    lEmin,lEmax,alpha,gamma,sfr_n,logmean,logsigma=pset
    Emin=10**lEmin
    Emax=10**lEmax
    
    # generates a DM mask
    # creates a mask of values in DM space to convolve with the DM grid
    mask=pcosmic.get_dm_mask(dmvals,(logmean,logsigma),zvals,plot=True)
    
    # get an initial grid with no beam values
    grids=[]
    bbs=[]
    bos=[]
    
    
    print("Just got into test parkes beam")
    
    
    #norms=np.zeros([len(surveys)])
    #numbins=np.zeros([len(surveys)])
    rates=[]
    New=False
    for i,s in enumerate(surveys):
        print("Starting",i)
        #s.beam_b
        #s.beam_o
        print("Sum of i is ",np.sum(s.beam_o))
        print(s.beam_o)
        print(s.beam_b)
        if New==True:
        
            grid=zdm_grid.Grid()
            grid.pass_grid(zDMgrid,zvals,dmvals)
            grid.smear_dm(mask,logmean,logsigma)
            efficiencies=s.get_efficiency(dmvals)
            grid.calc_thresholds(s.meta['THRESH'],s.mean_efficiencies,alpha=alpha)
            grid.calc_dV()
            grid.set_evolution(sfr_n) # sets star-formation rate scaling with z - here, no evoltion...
            grid.calc_pdv(Emin,Emax,gamma,s.beam_b,s.beam_o) # calculates volumetric-weighted probabilities
            grid.calc_rates() # calculates rates by multiplying above with pdm plot
            name=outdir+'rates_'+s.meta["BEAM"]+'.pdf'
            plot_grid_2(grid.rates,grid.zvals,grid.dmvals,zmax=zmax,DMmax=DMmax,name=name,norm=2,log=True,label='$f(DM,z)p(DM,z)dV$ [Mpc$^3$]',project=True)
            grids.append(grid)
            np.save(outdir+s.meta["BEAM"]+'_rates.npy',grid.rates)
            rate=grid.rates
        else:
            rate=np.load(outdir+s.meta["BEAM"]+'_rates.npy')
        print("Sum of rates: ",np.sum(rate),s.meta["BEAM"])
        rates.append(rate)
    fig1=plt.figure()
    plt.xlabel('z')
    plt.xlim(0,zmax)
    fig2=plt.figure()
    plt.xlabel('DM')
    plt.xlim(0,DMmax)
    
    fig3=plt.figure()
    plt.xlabel('z')
    plt.xlim(0,zmax)
    fig4=plt.figure()
    plt.xlabel('DM')
    plt.xlim(0,DMmax)
    
    #plt.yscale('log')
    # now does z-only and dm-only projection plots for Parkes
    for i,s in enumerate(surveys):
        r=rates[i]
        z=np.sum(r,axis=1)
        dm=np.sum(r,axis=0)
        plt.figure(fig1.number)
        plt.plot(zvals,z,label=s.meta["BEAM"])
        plt.figure(fig2.number)
        plt.plot(dmvals,dm,label=s.meta["BEAM"])
        
        z /= np.sum(z)
        dm /= np.sum(dm)
        plt.figure(fig3.number)
        plt.plot(zvals,z,label=s.meta["BEAM"])
        plt.figure(fig4.number)
        plt.plot(dmvals,dm,label=s.meta["BEAM"])
        
    plt.figure(fig1.number)
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(outdir+'z_projections.pdf')
    plt.close()
    
    plt.figure(fig2.number)
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(outdir+'dm_projections.pdf')
    plt.close()
    
    plt.figure(fig3.number)
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(outdir+'normed_z_projections.pdf')
    plt.close()
    
    plt.figure(fig4.number)
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(outdir+'normed_dm_projections.pdf')
    plt.close()
    
    ###### makes a 1d set of plots in dm and redshift ########
    font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 6}

    matplotlib.rc('font', **font)
    fig,ax=plt.subplots(3,2,sharey='row',sharex='col')#,sharey='row',sharex='col')
    
    ax[1,0].set_xlabel('z')
    ax[1,1].set_xlabel('DM')
    ax[2,0].set_xlabel('z')
    ax[2,1].set_xlabel('DM')
    ax[0,0].set_ylabel('Abs')
    ax[0,1].set_ylabel('Abs')
    ax[1,0].set_ylabel('Dabs')
    ax[1,1].set_ylabel('Dabs')
    ax[2,0].set_ylabel('Rel diff')
    ax[2,1].set_ylabel('Rel diff')
    
    # force relative range only
    ax[2,0].set_ylim(-1,1)
    ax[2,1].set_ylim(-1,1)
    
    ax[0,0].set_xlim(0,zmax)
    ax[0,1].set_xlim(0,DMmax)
    ax[1,0].set_xlim(0,zmax)
    ax[2,0].set_xlim(0,zmax)
    ax[1,1].set_xlim(0,DMmax)
    ax[2,1].set_xlim(0,DMmax)
    
    # gets Keith's normalised rates
    kr=rates[2]
    kz=np.sum(kr,axis=1)
    kdm=np.sum(kr,axis=0)
    kz /= np.sum(kz)
    kdm /= np.sum(kdm)
    
    ax[0,0].plot(zvals,kz,label=surveys[2].meta["BEAM"],color='black')
    ax[0,1].plot(dmvals,kdm,label=surveys[2].meta["BEAM"],color='black')
    
    for i,s in enumerate(surveys):
        if i==2:
            continue
        
        #calculates relative and absolute errors in dm and z space
        z=np.sum(rates[i],axis=1)
        dm=np.sum(rates[i],axis=0)
        z /= np.sum(z)
        dm /= np.sum(dm)
        
        dz=z-kz
        ddm = dm-kdm
        rdz=dz/kz
        rdm=ddm/kdm
        
        ax[0,0].plot(zvals,z,label=s.meta["BEAM"])
        ax[0,1].plot(dmvals,dm,label=s.meta["BEAM"])
        ax[1,0].plot(zvals,dz,label=s.meta["BEAM"])
        ax[1,1].plot(dmvals,ddm,label=s.meta["BEAM"])
        ax[2,0].plot(zvals,rdz,label=s.meta["BEAM"])
        ax[2,1].plot(dmvals,rdm,label=s.meta["BEAM"])
    ax[0,0].legend(fontsize=6)
    plt.figure(fig.number)
    plt.tight_layout()
    plt.savefig(outdir+'montage.pdf')
    plt.close()

def final_plot_beam_rates(surveys,zDMgrid, zvals,dmvals,pset,binset,names,logsigma,logmean,outdir,LOAD=True):
    """ For each survey, compare 'full' calculation to 'relative' in dm and z space
    binset is one for each survey, to be compared to 'all'
    """
    
    #need new ones for new grid shape
    
    # hard-coded best values
    method=2
    thresh=0
    
    
    ###### makes a 1d set of plots in dm and redshift ########
    font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 10}

    matplotlib.rc('font', **font)
    
    
    # get parameter values
    lEmin,lEmax,alpha,gamma,sfr_n,logmean,logsigma,C=pset
    Emin=10**lEmin
    Emax=10**lEmax
    
    # generates a DM mask
    # creates a mask of values in DM space to convolve with the DM grid
    mask=pcosmic.get_dm_mask(dmvals,(logmean,logsigma),zvals)
    
    f1, (ax11, ax12) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3, 1]},sharex=True)
    
    plt.subplots_adjust(wspace=0, hspace=0)
    ax11.set_xlim(0,2)
    ax12.set_xlim(0,2)
    ax11.set_ylabel('$p(z)$ [a.u.]')
    ax12.set_ylabel('$p_{\\rm Full}(z)-p(z)$')
    ax12.set_xlabel('z')
    
    f2, (ax21, ax22) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3, 1]},sharex=True)
    plt.subplots_adjust(wspace=0, hspace=0)
    ax21.set_xlim(0,2500)
    ax22.set_xlim(0,2500)
    ax21.set_ylabel('$p(\\rm DM_{\\rm EG})$ [a.u.]')
    ax22.set_ylabel('$p(\\rm DM_{\\rm EG})-p_{\\rm Full}(\\rm DM_{\\rm EG})$')
    ax22.set_xlabel('${\\rm DM}_{\\rm EG}$')
    
    # does this for each survey
    # for lat50, FE, and Parkes
    FWHM0=np.array([32,32,0.54])*(np.pi/180)**2 # nominal deg square
    print("Generating plots illustrating the effect of beamshape")
    print("Order is FWHM, Numerical, Gaussian")
    for i,s in enumerate(surveys):
        #efficiencies=s.get_efficiency(dmvals)
        efficiencies=s.efficiencies # two dimensions
        weights=s.wplist
        
        ######## Naive FWHM case - single beam value, single angle ########
        
        if LOAD:
            rates=np.load('TEMP/'+str(i)+'1.npy')
        
        else:
            t0=t0=time.process_time()
            # set up grid, which should be common for this survey
            grid=zdm_grid.Grid()
            grid.pass_grid(zDMgrid,zvals,dmvals)
            grid.smear_dm(mask,logmean,logsigma)
            grid.calc_thresholds(s.meta['THRESH'],efficiencies,alpha=alpha,weights=weights)
            grid.calc_dV()
            grid.set_evolution(sfr_n) # sets star-formation rate scaling with z - here, no evoltion...
            grid.b_fractions=None # trick!
            grid.calc_pdv(Emin,Emax,gamma,np.array([1]),np.array([FWHM0[i]])) # calculates volumetric-weighted probabilities
            grid.calc_rates() # calculates rates by multiplying above with pdm plot
            t1=time.process_time()
            np.save('TEMP/'+str(i)+'1.npy',grid.rates)
            rates=grid.rates
        
        total1=np.sum(rates)
        rates1=rates / total1
        
        fz1=np.sum(rates1,axis=1)
        fdm1=np.sum(rates1,axis=0)
        
        ######## full case - very detailed! ########
        if LOAD:
            rates=np.load('TEMP/'+str(i)+'2.npy')
        else:
            s.init_beam(nbins=1,method=3,thresh=thresh) #nbins ignored for method=3
            #s.init_beam(nbins=1,method=2,thresh=thresh) #make it fast!
            
            grid.b_fractions=None # trick!
            grid.calc_pdv(Emin,Emax,gamma,s.beam_b,s.beam_o) # calculates volumetric-weighted probabilities
            grid.calc_rates() # calculates rates by multiplying above with pdm plot
            np.save('TEMP/'+str(i)+'2.npy',grid.rates)
            rates=grid.rates
        total2=np.sum(rates)
        rates2=rates / total2
        fz2=np.sum(rates2,axis=1)
        fdm2=np.sum(rates2,axis=0)
        
        ######## Case of nbins bins - 'standard' #########
        if LOAD:
            rates=np.load('TEMP/'+str(i)+'3.npy')
        else:
            
            s.init_beam(nbins=binset[i],method=method,thresh=thresh)
            grid.b_fractions=None # trick!
            grid.calc_pdv(Emin,Emax,gamma,s.beam_b,s.beam_o) # calculates volumetric-weighted probabilities
            grid.calc_rates() # calculates rates by multiplying above with pdm plot
            np.save('TEMP/'+str(i)+'3.npy',grid.rates)
            rates=grid.rates
        total3=np.sum(rates)
        rates3=rates / total3
        fz3=np.sum(rates3,axis=1)
        fdm3=np.sum(rates3,axis=0)
        
        ######## Gaussian case #########
        
        if LOAD:
            rates=np.load('TEMP/'+str(i)+'4.npy')
        else:
            
            thresh=1e-3 #argh!
            s.init_beam(nbins=100,method=method,thresh=thresh,Gauss=True)
            
            grid.b_fractions=None # trick!
            grid.calc_pdv(Emin,Emax,gamma,s.beam_b,s.beam_o) # calculates volumetric-weighted probabilities
            grid.calc_rates() # calculates rates by multiplying above with pdm plot
            np.save('TEMP/'+str(i)+'4.npy',grid.rates)
            rates=grid.rates
        total4=np.sum(rates)
        rates4=rates / total4
        fz4=np.sum(rates4,axis=1)
        fdm4=np.sum(rates4,axis=0)
        
        
        ######## calculate some statistics #########
        
        # stats for redshift z
        
        true_mean=np.sum(fz2*zvals)
        dm_mean=np.sum(fdm2*dmvals)
        
        nerr1=total1/total2
        nerr3=total3/total2
        nerr4=total4/total2
        zerr1=np.sum(fz1*zvals)/true_mean
        zerr3=np.sum(fz3*zvals)/true_mean
        zerr4=np.sum(fz4*zvals)/true_mean
        dmerr1=np.sum(fdm1*dmvals)/dm_mean
        dmerr3=np.sum(fdm3*dmvals)/dm_mean
        dmerr4=np.sum(fdm4*dmvals)/dm_mean
        
        print("\n\n\nNormalisation errors: ",nerr1,nerr3,nerr4)
        print("zerr : ",zerr1,zerr3,zerr4)
        print("dmerr : ",dmerr1,dmerr3,dmerr4)
        
        ############## plotting ##########
        # normalise by amplitude of 'true':
        normz=np.max(fz2)
        normdm=np.max(fdm2)
        
        fz1 /= normz
        fz2 /= normz
        fz3 /= normz
        fz4 /= normz
        
        fdm1 /= normdm
        fdm2 /= normdm
        fdm3 /= normdm
        fdm4 /= normdm
        
        plt.sca(ax11)
        plt.plot(zvals,fz1,linestyle='--',label=names[i]+' FWHM')
        c1=plt.gca().lines[-1].get_color()
        
        plt.sca(ax21)
        plt.plot(dmvals,fdm1,linestyle='--',color=c1,label=names[i]+' FWHM')
        
        plt.sca(ax11)
        plt.plot(zvals,fz2,color=c1,linestyle='-',label='      Full beam')
        
        plt.sca(ax21)
        plt.plot(dmvals,fdm2,color=c1,linestyle='-',label='      Full beam')
        
        plt.sca(ax11)
        plt.plot(zvals,fz3,color=plt.gca().lines[-1].get_color(),linestyle=':',label='      This work')
        
        plt.sca(ax21)
        plt.plot(dmvals,fdm3,color=plt.gca().lines[-1].get_color(),linestyle=':',label='      This work')
        
        plt.sca(ax11)
        plt.plot(zvals,fz4,color=plt.gca().lines[-1].get_color(),linestyle='-.',label='      Gauss')
        
        plt.sca(ax21)
        plt.plot(dmvals,fdm4,color=plt.gca().lines[-1].get_color(),linestyle='-.',label='      Gauss')
        
        
        ###### now does relative values #######
        dz=fz3-fz2
        ddm=fdm3-fdm2
        
        dz0=fz1-fz2
        ddm0=fdm1-fdm2
        
        
        dzG=fz4-fz2
        ddmG=fdm4-fdm2
        
        print("For survey ",i," maximum dz deviation is ",np.max(np.abs(dz0)),np.max(np.abs(dz)),np.max(np.abs(dzG)))
        print("                         dm deviation is ",np.max(np.abs(ddm0)),np.max(np.abs(ddm)),np.max(np.abs(ddmG)))
        
        # plots differences
        plt.sca(ax12)
        ax12.set_ylim(-0.2,0.2)
        plt.plot(zvals,dz0,color=c1,linestyle='--')
        plt.plot(zvals,dz,color=c1,linestyle=':')
        plt.plot(zvals,dzG,color=c1,linestyle='-.')
        #ax12.tick_params(axis='y')
        
        #ax122=ax12.twinx()
        #ax122.set_ylim(-0.2,0.2)
        
        #ax122.tick_params(axis='y')
        
        plt.sca(ax22)
        ax22.set_ylim(-0.2,0.2)
        plt.plot(dmvals,ddm0,color=c1,linestyle='--')
        plt.plot(dmvals,ddm,color=c1,linestyle=':')
        plt.plot(dmvals,ddmG,color=c1,linestyle='-.')
        #ax222=ax22.twinx()
        #ax222.set_ylim(-0.2,0.2)
        
        
        print("Total rates for are ",i,total1,total2,total3,total4)
    
    plt.figure(f1.number)
    leg1=ax11.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(outdir+'/beam_z_comp.pdf')
    plt.close()
    
    plt.figure(f2.number)
    leg2=ax21.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(outdir+'/beam_dm_comp.pdf')

    

def final_plot_beam_values(surveys,zDMgrid, zvals,dmvals,pset,binset,names,logsigma,logmean,outdir):
    """ For each survey, get the beamshape, and plot it vs the dots on the one plot
    """
    # hard-coded best values
    method=2
    thresh=0
    
    # does this for each survey
    # for lat50, FE, and Parkes
    #FWHM0=np.array([32,32,0.54])*(np.pi/180)**2 # nominal deg square
    
    plt.figure()
    plt.yscale('log')
    plt.xscale('log')
    plt.xlabel('$B$')
    plt.ylabel('$\\Omega(B)\\, d\\log_{10}B$ [sr]') # data is dlogB for constant logB
    
    
    markers=['o','o','o']
    lss=["-",":","--"]
    names=["ASKAP","ASKAP","Parkes/Mb"]
    n=[5,1,1]
    for i,s in enumerate(surveys):
        if i==1:
            continue
        #efficiencies=s.get_efficiency(dmvals)
        efficiencies=s.efficiencies # two dimensions
        weights=s.wplist
        
        # gets Gaussian beam
        s.init_beam(nbins=binset[i]*10,method=method,thresh=1e-3,Gauss=True)
        gb=np.copy(s.beam_b)
        go=np.copy(s.beam_o)
        gdb=np.log10(gb[0]/gb[1])
        
        # simple point of Nbeams * 1 * FWHM
        simple_x=1
        HPBW=1.22*(sp.constants.c/(s.meta["FBAR"]*1e6))/s.meta["DIAM"]
        simple_y=np.pi*HPBW**2*s.meta["NBEAMS"]
        
        # standard method
        s.init_beam(nbins=binset[i],method=method,thresh=thresh)
        
        
        #calculates normalisation factor: integral B db
        orig_db=np.log10(s.orig_beam_b[1]/s.orig_beam_b[0])
        # log10 grid spacing of original plot
        # now db is per natural log
        # first divide by db factor
        # since d Omega dlogB = d Omega/dB * dB/dlogB = dOmega/dB B
        # d Omega/dB = d Omega dlogB/B
        # but we will not do this!
        
        db=np.log10(s.beam_b[1]/s.beam_b[0]) # also divides this one by the log spacing!
        part=np.where(s.orig_beam_b > 1e-3)
        to_sqr_deg=(180/np.pi)**2
        #print("The log(10) corrected sums are [deg2]",np.sum(s.orig_beam_o[part])*to_sqr_deg,np.sum(go)/np.log(10)*to_sqr_deg,np.sum(s.beam_o)*to_sqr_deg)
        print("The uncorrected sums are [deg2]",np.sum(s.orig_beam_o[part])*to_sqr_deg,np.sum(go)*to_sqr_deg,np.sum(s.beam_o)*to_sqr_deg)
        
        plt.plot(s.orig_beam_b[::n[i]],s.orig_beam_o[::n[i]]/orig_db,linestyle=lss[i],label=names[i])
        plt.plot(gb,go/gdb,color=plt.gca().lines[-1].get_color(),linestyle=lss[i])
        plt.plot(s.beam_b,s.beam_o/db,marker=markers[i],color=plt.gca().lines[-1].get_color(),linestyle="",markersize=10)
        plt.plot(simple_x,simple_y,marker='+',color=plt.gca().lines[-1].get_color(),markersize=10)
    
    plt.legend()
    plt.tight_layout()
    
    plt.savefig(outdir+'/beam_approx.pdf')

    
def test_beam_rates(survey,zDMgrid, zvals,dmvals,pset,binset,method=2,outdir='Plots/BeamTest/',thresh=1e-3,zmax=1,DMmax=1000):
    """ For a list of surveys, construct a zDMgrid object
    binset is the set of bins which we use to simplify the
    beamset
    We conclude that method=2, nbeams=5, acc=0 is the best here
    """
    
    #zmax=4
    #DMmax=4000
    
    if not os.path.isdir(outdir):
        os.mkdir(outdir)
    
    # get parameter values
    lEmin,lEmax,alpha,gamma,sfr_n,logmean,logsigma,C=pset
    Emin=10**lEmin
    Emax=10**lEmax
    
    # generates a DM mask
    # creates a mask of values in DM space to convolve with the DM grid
    mask=pcosmic.get_dm_mask(dmvals,(logmean,logsigma),zvals,plot=True)
    efficiencies=survey.get_efficiency(dmvals)
    
    # get an initial grid with no beam values
    grids=[]
    bbs=[]
    bos=[]
    
    
    norms=np.zeros([len(binset)])
    numbins=np.zeros([len(binset)])
    
    for k,nbins in enumerate(binset):
        grid=zdm_grid.Grid()
        grid.pass_grid(zDMgrid,zvals,dmvals)
        grid.smear_dm(mask,logmean,logsigma)
        grid.calc_thresholds(survey.meta['THRESH'],survey.mean_efficiencies,alpha=alpha)
        grid.calc_dV()
        grid.set_evolution(sfr_n) # sets star-formation rate scaling with z - here, no evoltion...
        
        if nbins != 0 and nbins != 'all':
            survey.init_beam(nbins=nbins,method=method,thresh=thresh)
            bbs.append(np.copy(survey.beam_b))
            bos.append(np.copy(survey.beam_o))
            grid.calc_pdv(Emin,Emax,gamma,survey.beam_b,survey.beam_o) # calculates volumetric-weighted probabilities
            numbins[k]=nbins
        elif nbins ==0:
            grid.calc_pdv(Emin,Emax,gamma) # calculates volumetric-weighted probabilities
            bbs.append(np.array([1]))
            bos.append(np.array([1]))
            numbins[k]=nbins
        else:
            survey.init_beam(nbins=nbins,method=3,thresh=thresh)
            bbs.append(np.copy(survey.beam_b))
            bos.append(np.copy(survey.beam_o))
            numbins[k]=survey.beam_o.size
            grid.calc_pdv(Emin,Emax,gamma,survey.beam_b,survey.beam_o) # calculates volumetric-weighted probabilities
        
        
        grid.calc_rates() # calculates rates by multiplying above with pdm plot
        name=outdir+'beam_test_'+survey.meta["BEAM"]+'_nbins_'+str(nbins)+'.pdf'
        plot_grid_2(grid.rates,grid.zvals,grid.dmvals,zmax=zmax,DMmax=DMmax,name=name,norm=2,log=True,label='$f(DM_{\\rm EG},z)p(DM_{\\rm EG},z)dV$ [Mpc$^3$]',project=True)
        grids.append(grid)
    
    # OK, we now have a list of grids with various interpolating factors
    # we produce plots of the rate for each, and also difference plots with the best
    #Does a linear plot relative to the best case
    
    
    bestgrid=grids[-1] # we always have the worst grid at 0
    #bestgrid.rates=bestgrid.rates / np.sum(bestgrid.rates)
    
    # normalises
    
    for i,grid in enumerate(grids):
        norms[i]=np.sum(grid.rates)
        grid.rates=grid.rates / norms[i]
    
    np.save(outdir+survey.meta["BEAM"]+'_total_rates.npy',norms)
    np.save(outdir+survey.meta["BEAM"]+'_nbins.npy',numbins)
    
    bestz=np.sum(grid.rates,axis=1)
    bestdm=np.sum(grid.rates,axis=0)
    
    ###### makes a 1d set of plots in dm and redshift ########
    font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 6}

    matplotlib.rc('font', **font)
    fig,ax=plt.subplots(3,2,sharey='row',sharex='col')#,sharey='row',sharex='col')
    
    #ax[0,0]=fig.add_subplot(221)
    #ax[0,1]=fig.add_subplot(222)
    #ax[1,0]=fig.add_subplot(223)
    #ax[1,1]=fig.add_subplot(224)
    
    #ax[0,0].plot(grid.zvals,bestz,color='black',label='All')
    ax[1,0].set_xlabel('z')
    ax[1,1].set_xlabel('DM_{\\rm EG}')
    ax[2,0].set_xlabel('z')
    ax[2,1].set_xlabel('DM_{\\rm EG}')
    ax[0,0].set_ylabel('Abs')
    ax[0,1].set_ylabel('Abs')
    ax[1,0].set_ylabel('Dabs')
    ax[1,1].set_ylabel('Dabs')
    ax[2,0].set_ylabel('Rel diff')
    ax[2,1].set_ylabel('Rel diff')
    
    # force relative range only
    ax[2,0].set_ylim(-1,1)
    ax[2,1].set_ylim(-1,1)
    
    ax[0,0].set_xlim(0,zmax)
    ax[0,1].set_xlim(0,DMmax)
    ax[1,0].set_xlim(0,zmax)
    ax[2,0].set_xlim(0,zmax)
    ax[1,1].set_xlim(0,DMmax)
    ax[2,1].set_xlim(0,DMmax)
    
    ax[0,0].plot(grid.zvals,np.sum(bestgrid.rates,axis=1),label='All',color='black')
    ax[0,1].plot(grid.dmvals,np.sum(bestgrid.rates,axis=0),label='All',color='black')
    
    for i,grid in enumerate(grids[:-1]):
        
        diff=grid.rates-bestgrid.rates
        
        #calculates relative and absolute errors in dm and z space
        dz=np.sum(diff,axis=1)
        ddm=np.sum(diff,axis=0)
        rdz=dz/bestz
        rdm=ddm/bestdm
        
        thisz=np.sum(grid.rates,axis=1)
        thisdm=np.sum(grid.rates,axis=0)
        
        ax[0,0].plot(grid.zvals,thisz,label=str(binset[i]))
        ax[0,1].plot(grid.dmvals,thisdm,label=str(binset[i]))
        ax[1,0].plot(grid.zvals,dz,label=str(binset[i]))
        ax[1,1].plot(grid.dmvals,ddm,label=str(binset[i]))
        ax[2,0].plot(grid.zvals,rdz,label=str(binset[i]))
        ax[2,1].plot(grid.dmvals,rdm,label=str(binset[i]))
    ax[0,0].legend(fontsize=6)
    plt.figure(fig.number)
    plt.tight_layout()
    plt.savefig(outdir+'d_dm_z_'+survey.meta["BEAM"]+'_nbins_'+str(binset[i])+'.pdf')
    plt.close()
    
    acc=open(outdir+'accuracy.dat','w')
    mean=np.mean(bestgrid.rates)
    size=bestgrid.rates.size
    string='#Nbins   Acc     StdDev    StdDev/mean; mean={:.2E}\n'.format(mean)
    acc.write('#Nbins   Acc     StdDev    StdDev/mean; mean='+str(mean)+'\n')
    
    for i,grid in enumerate(grids[:-1]):
        
        diff=grid.rates-bestgrid.rates
        
        inaccuracy=np.sum(diff**2)
        std_dev=(inaccuracy/size)**0.5
        rel_std_dev=std_dev/mean
        #print("Beam with bins ",binset[i]," has total inaccuracy ",inaccuracy)
        string="{:.0f} {:.2E} {:.2E} {:.2E}".format(binset[i],inaccuracy,std_dev,rel_std_dev)
        acc.write(string+'\n')
        name=outdir+'diff_beam_test_'+survey.meta["BEAM"]+'_nbins_'+str(binset[i])+'.pdf'
        
        plot_grid_2(diff,grid.zvals,grid.dmvals,zmax=zmax,DMmax=DMmax,name=name,norm=0,log=False,label='$f(DM_{\\rm EG},z)p(DM_{\\rm EG},z)dV$ [Mpc$^3$]',project=True)
        diff=diff/bestgrid.rates
        nans=np.isnan(diff)
        diff[nans]=0.
        name=outdir+'rel_diff_beam_test_'+survey.meta["BEAM"]+'_nbins_'+str(binset[i])+'.pdf'
        plot_grid_2(diff,grid.zvals,grid.dmvals,zmax=zmax,DMmax=DMmax,name=name,norm=0,log=False,label='$f(DM_{\\rm EG},z)p(DM_{\\rm EG},z)dV$ [Mpc$^3$]',project=True)
    
    
    acc.close()

def initialise_grids(surveys: list, zDMgrid: np.ndarray, 
                     zvals: np.ndarray, 
                     dmvals: np.ndarray, state:parameters.State, 
                     wdist=True): 
    """ For a list of surveys, construct a zDMgrid object
    wdist indicates a distribution of widths in the survey,
    i.e. do not use the mean efficiency
    Assumes that survey efficiencies ARE initialised

    Args:
        surveys (list): [description]
        zDMgrid (np.ndarray): [description]
        zvals (np.ndarray): [description]
        dmvals (np.ndarray): [description]
        state (parameters.State): parameters guiding the analysis
            Each grid gets its *own* copy
        wdist (bool, optional): [description]. Defaults to False.

    Returns:
        list: list of Grid objects
    """
    if not isinstance(surveys,list):
        surveys=[surveys]
    
    # get parameter values
    #lEmin,lEmax,alpha,gamma,sfr_n,logmean,logsigma,lC,H0=pset
    #lEmin,lEmax,alpha,gamma,sfr_n,logmean,logsigma,lC,H0=parameters.unpack_pset(params)
    #Emin=10**lEmin
    #Emax=10**lEmax
    
    # generates a DM mask
    # creates a mask of values in DM space to convolve with the DM grid
    mask=pcosmic.get_dm_mask(
        dmvals,(state.host.lmean,state.host.lsigma),
        zvals,plot=True)
    grids=[]
    for survey in surveys:
        '''
        if wdist:
            efficiencies=survey.efficiencies # two dimensions
            weights=survey.wplist
        else:
            efficiencies=survey.mean_efficiencies
            weights=None
            #efficiencies=survey.get_efficiency(dmvals)
        '''
        
        grid=zdm_grid.Grid(survey, copy.deepcopy(state),
                           zDMgrid, zvals, dmvals, mask, wdist)
        '''
        grid.pass_grid(zDMgrid,zvals,dmvals)
        grid.smear_dm(mask)#,logmean,logsigma)
        
        # TODO -- avoid code duplication with grid.update_grid()
        # note - survey frequencies in MHz
        grid.calc_thresholds(survey.meta['THRESH'],
                             efficiencies,
                             weights=weights,
                             nuObs=survey.meta['FBAR']*1e6)
        grid.calc_dV()
        grid.calc_pdv()#survey.beam_b,
                      #survey.beam_o) # calculates volumetric-weighted probabilities
        grid.set_evolution() # sets star-formation rate scaling with z - here, no evoltion...
        grid.calc_rates() # calculates rates by multiplying above with pdm plot
        '''
        grids.append(grid)
    
    return grids
    
def generate_example_plots():
    """ Loads the lat50survey and generates some example plots """
    
    #cos.set_cosmology(Omega_m=1.2) setup for cosmology
    cos.init_dist_measures()
    
    #parser.add_argument(", help
    # get the grid of p(DM|z)
    zDMgrid, zvals,dmvals=get_zdm_grid(new=False,plot=False,method='analytic')
    pcosmic.plot_mean(zvals,'Plots/mean_DM.pdf')
    
    #load the lat50 survey data
    lat50=survey.survey()
    lat50.process_survey_file('Surveys/CRAFT_lat50.dat')
    
    efficiencies=lat50.get_efficiency(dmvals)
    plot_efficiencies(lat50)
    
    # we now do the mean efficiency approximation
    #mean_efficiencies=np.mean(efficiencies,axis=0)
    #Fth=lat50.meta('THRESH')
    
    # create a grid object
    grid=zdm_grid.Grid()
    grid.pass_grid(zDMgrid,zvals,dmvals)
    
    # plots the grid of intrinsic p(DM|z)
    plot_grid_2(grid.grid,grid.zvals,grid.dmvals,zmax=1,DMmax=1000,name='Plots/p_dm_z_grid_image.pdf',norm=1,log=True,label='$\\log_{10}p(DM_{\\rm EG}|z)$',conts=[0.16,0.5,0.88])
    
    # creates a mask of values in DM space to convolve with the DM
    # grid
    # best-fit values-ish from green curves in fig 3 of cosmic dm paper
    mean=125
    sigma=10**0.25
    logmean=np.log10(mean)
    logsigma=np.log10(sigma)
    mask=pcosmic.get_dm_mask(grid.dmvals,(logmean,logsigma),zvals,plot=True)
    
    grid.smear_dm(mask,logmean,logsigma)
    # plots the grid of intrinsic p(DM|z)
    plot_grid_2(grid.smear_grid,grid.zvals,grid.dmvals,zmax=1,DMmax=1000,name='Plots/DMX_grid_image.pdf',norm=1,log=True,label='$\\log_{10}p(DM_{\\rm EG}|z)$')
    #plot_grid_2(grid.smear_grid2,grid.zvals,grid.dmvals,zmax=1,DMmax=1000,name='DMX_grid_image2.pdf',norm=True,log=True,label='$\\log_{10}p(DM_{\\rm EG}|z)$')
    
    # plots grid of effective thresholds
    alpha=1.6
    grid.calc_thresholds(lat50.meta['THRESH'],lat50.mean_efficiencies,alpha=alpha)
    plot_grid_2(grid.thresholds,grid.zvals,grid.dmvals,zmax=1,DMmax=1000,name='Plots/thresholds_dm_z_grid_image.pdf',norm=1,log=True,label='$\\log (E_{\\rm th})$ [erg]')
    
    # calculates rates for given gamma etc
    gamma=-0.7
    Emax=1e42
    Emin=1e30
    grid.calc_dV()
    
    grid.calc_pdv(Emin,Emax,gamma) # calculates volumetric-weighted probabilities
    grid.set_evolution(0) # sets star-formation rate scaling with z - here, no evoltion...
    grid.calc_rates() # calculates rates by multiplying above with pdm plot
    plot_grid_2(grid.pdv,grid.zvals,grid.dmvals,zmax=1,DMmax=1000,name='Plots/pdv.pdf',norm=True,log=True,label='$p(DM_{\\rm EG},z)dV$ [Mpc$^3$]')
    plot_grid_2(grid.rates,grid.zvals,grid.dmvals,zmax=1,DMmax=1000,name='Plots/base_rate_dm_z_grid_image.pdf',norm=2,log=True,label='$f(DM_{\\rm EG},z)p(DM_{\\rm EG},z)dV$ [Mpc$^3$]')
    plot_grid_2(grid.rates,grid.zvals,grid.dmvals,zmax=1,DMmax=1000,name='Plots/project_rate_dm_z_grid_image.pdf',norm=2,log=True,label='$f(DM_{\\rm EG},z)p(DM_{\\rm EG},z)dV$ [Mpc$^3$]',project=True)
    
    
    it.calc_likelihoods_1D(grid.rates,grid.zvals,grid.dmvals,lat50.DMEGs)
    plot_grid_2(grid.rates,grid.zvals,grid.dmvals,zmax=1,DMmax=1000,name='Plots/wFRB_project_rate.pdf',norm=2,log=True,label='$f(DM_{\\rm EG},z)p(DM_{\\rm EG},z)dV$ [Mpc$^3$]',project=True,FRBDM=lat50.DMEGs)
    
    ###### shows how to do a 1D scan of parameter values #######
    pset=it.set_defaults(grid)
    it.print_pset(pset)
    # define set of values to scan over
    lEmaxs=np.linspace(40,44,21)
    likes=it.scan_likelihoods_1D(grid,pset,lat50,1,lEmaxs,norm=True)
    plot_1d(lEmaxs,likes,'$E_{\\rm max}$','Plots/test_lik_fn_emax.pdf')
        

def plot_1d(pvec,lset,xlabel,savename,showplot=False):
    plt.figure()
    plt.xlabel(xlabel)
    plt.ylabel('$\\ell($'+xlabel+'$)$')
    plt.plot(pvec,lset)
    plt.tight_layout()
    plt.savefig(savename)
    if showplot:
        plt.show()
    plt.close()
    
# generates grid based on Monte Carlo model
def get_zdm_grid(state:parameters.State, new=True,plot=False,method='analytic',
                 nz=500,zmax=5,ndm=1400,dmmax=7000.,
                 datdir='GridData',tag="", orig=False,
                 verbose=False):
    """Generate a grid of z vs. DM for an assumed F value
    for a specified z range and DM range.

    Args:
        state (parameters.State): Object holding all the key parameters for the analysis
        new (bool, optional): [description]. Defaults to True.
        plot (bool, optional): [description]. Defaults to False.
        method (str, optional): [description]. Defaults to 'analytic'.
        nz (int, optional): Size of grid in redshift. Defaults to 500.
        zmax (int, optional): [description]. Defaults to 5.
        ndm (int, optional): Size of grid in DM.  Defaults to 1400.
        dmmax ([type], optional): Maximum DM of grid. Defaults to 7000..
        datdir (str, optional): [description]. Defaults to 'GridData'.
        tag (str, optional): [description]. Defaults to "".
        orig (bool, optional): Use original calculations for 
            things like C0. Defaults to False.

    Returns:
        tuple: zDMgrid, zvals, dmvals
    """
    # no action in fail case - it will already exist
    try:
        os.mkdir(datdir)
    except:
        pass
    if method=='MC':
        savefile=datdir+'/'+tag+'zdm_MC_grid_'+str(state.IGM.F)+'.npy'
        datfile=datdir+'/'+tag+'zdm_MC_data_'+str(state.IGM.F)+'.npy'
        zfile=datdir+'/'+tag+'zdm_MC_z_'+str(state.IGM.F)+'.npy'
        dmfile=datdir+'/'+tag+'zdm_MC_dm_'+str(state.IGM.F)+'.npy'
    elif method=='analytic':
        savefile=datdir+'/'+tag+'zdm_A_grid_'+str(state.IGM.F)+'H0_'+str(state.cosmo.H0)+'.npy'
        datfile=datdir+'/'+tag+'zdm_A_data_'+str(state.IGM.F)+'H0_'+str(state.cosmo.H0)+'.npy'
        zfile=datdir+'/'+tag+'zdm_A_z_'+str(state.IGM.F)+'H0_'+str(state.cosmo.H0)+'.npy'
        dmfile=datdir+'/'+tag+'zdm_A_dm_'+str(state.IGM.F)+'H0_'+str(state.cosmo.H0)+'.npy'
        C0file=datdir+'/'+tag+'zdm_A_C0_'+str(state.IGM.F)+'H0_'+str(state.cosmo.H0)+'.npy'
    #labelled pickled files with H0
    if new:
        
        #nz=500
        #zmax=5
        dz=zmax/nz
        
        #ndm=1400
        #dmmax=7000.
        ddm=dmmax/ndm
        
        zvals=(np.arange(nz)+1)*dz
        dmvals=(np.arange(ndm)+1)*ddm
        
        dmmeans=dmvals[1:] - (dmvals[1]-dmvals[0])/2.
        zdmgrid=np.zeros([nz,ndm])
        
        if method=='MC':
            # generate DM grid from the models
            if verbose:
                print("Generating the zdm Monte Carlo grid")
            nfrb=10000
            t0=time.process_time()
            DMs = dlas.monte_DM(np.array(zvals)*3000, nrand=nfrb)
            #DMs *= 200000 #seems to be a good fit...
            t1=time.process_time()
            dt=t1-t0
            print("Done. Took ",dt," seconds")
            hists=[]
            for i,z in enumerate(zvals):
                print("first 10 DM list for z=",z," is ",DMs[0:10,i])
                hist,bins=np.histogram(DMs[:,i],bins=dmvals)
                hists.append(hist)
            all_hists=np.array(hists)
            #all_hists /= float(nfrb)
            print(all_hists.shape)
        elif method=='analytic':
            if verbose:
                print("Generating the zdm analytic grid")
            t0=time.process_time()
            # calculate constants for p_DM distribution
            if orig:
                C0s=pcosmic.make_C0_grid(zvals,state.IGM.F)
            else:
                f_C0_3 = cosmic.grab_C0_spline()
                sigma = state.IGM.F / np.sqrt(zvals)
                C0s = f_C0_3(sigma)
            # generate pDM grid using those COs
            #zDMgrid=pcosmic.get_pDM_grid(H0,F,dmvals,zvals,C0s)
            zDMgrid=pcosmic.get_pDM_grid(state,dmvals,zvals,C0s)
            t1=time.process_time()
            dt=t1-t0
            if verbose:
                print("Done. Took ",dt," seconds")
        
        np.save(savefile,zDMgrid)
        metadata=np.array([nz,ndm,state.IGM.F])
        np.save(datfile,metadata)
        np.save(zfile,zvals)
        np.save(dmfile,dmvals)
    else:
        zDMgrid=np.load(savefile)
        zvals=np.load(zfile)
        dmvals=np.load(dmfile)
        metadata=np.load(datfile)
        nz,ndm,F=metadata
    
    if plot:
        plt.figure()
        plt.xlabel('DM_{\\rm EG} [pc cm$^{-3}$]')
        plt.ylabel('p(DM_{\\rm EG})')
        
        nplot=int(zvals.size/10)
        for i,z in enumerate(zvals):
            if i%nplot==0:
                plt.plot(dmvals,zDMgrid[i,:],label='z='+str(z)[0:4])
        plt.legend()
        plt.tight_layout()
        plt.savefig('p_dm_slices.pdf')
        plt.close()

    return zDMgrid, zvals,dmvals

def plot_zdm_basic_paper(zDMgrid,zvals,dmvals,zmax=1,DMmax=1000,
                         norm=0,log=True,name='temp.pdf',ylabel=None,
                         label='$\\log_{10}p(DM_{\\rm EG},z)$',project=False,conts=False,FRBZ=None,
                         FRBDM=None,title='Plot',H0=None,showplot=False):
    ''' Plots basic distributions of z and dm for the paper '''
    if H0 is None:
        H0 = cos.cosmo.H0
    cmx = plt.get_cmap('cubehelix')
    
    ##### imshow of grid #######
    
    # we protect these variables
    zDMgrid=np.copy(zDMgrid)
    zvals=np.copy(zvals)
    dmvals=np.copy(dmvals)
    
    plt.figure()
    #rect_2D=[0,0,1,1]
    ax1=plt.axes()
    
    plt.sca(ax1)
    
    plt.xlabel('z')
    if ylabel is not None:
        plt.ylabel(ylabel)
    else:
        plt.ylabel('${\\rm DM}_{\\rm EG}$')
    
    nz,ndm=zDMgrid.shape
    
    
    ixmax=np.where(zvals > zmax)[0]
    if len(ixmax) >0:
        zvals=zvals[:ixmax[0]]
        nz=zvals.size
        zDMgrid=zDMgrid[:ixmax[0],:]
    
    ### generates contours *before* cutting array in DM ###
    ### might need to normalise contours by integer lengths, oh well! ###
    if conts:
        nc = len(conts)
        carray=np.zeros([nc,nz])
        for i in np.arange(nz):
            cdf=np.cumsum(zDMgrid[i,:])
            cdf /= cdf[-1]
            
            for j,c in enumerate(conts):
                less=np.where(cdf < c)[0]
                
                if len(less)==0:
                    carray[j,i]=0.
                    dmc=0.
                    il1=0
                    il2=0
                else:
                    il1=less[-1]
                    
                    if il1 == ndm-1:
                        il1=ndm-2
                    
                    il2=il1+1
                    k1=(cdf[il2]-c)/(cdf[il2]-cdf[il1])
                    dmc=k1*dmvals[il1]+(1.-k1)*dmvals[il2]
                    carray[j,i]=dmc
                
        ddm=dmvals[1]-dmvals[0]
        carray /= ddm # turns this into integer units for plotting
        
    iymax=np.where(dmvals > DMmax)[0]
    if len(iymax)>0:
        dmvals=dmvals[:iymax[0]]
        zDMgrid=zDMgrid[:,:iymax[0]]
        ndm=dmvals.size
    
    # currently this is "per cell" - now to change to "per DM"
    # normalises the grid by the bin width, i.e. probability per bin, not probability density
    ddm=dmvals[1]-dmvals[0]
    dz=zvals[1]-zvals[0]
    
    zDMgrid /= ddm # norm=1
    
    # checks against zeros for a log-plot
    orig=np.copy(zDMgrid)
    zDMgrid=zDMgrid.reshape(zDMgrid.size)
    setzero=np.where(zDMgrid==0.)
    zDMgrid=np.log10(zDMgrid)
    zDMgrid[setzero]=-100
    zDMgrid=zDMgrid.reshape(nz,ndm)
    
    # gets a square plot
    aspect=nz/float(ndm)
    
    # sets the x and y tics	
    xtvals=np.arange(zvals.size)
    everx=int(zvals.size/5)
    plt.xticks(xtvals[everx-1::everx],zvals[everx-1::everx])

    ytvals=np.arange(dmvals.size)
    every=int(dmvals.size/5)
    plt.yticks(ytvals[every-1::every],dmvals[every-1::every])
    
    im=plt.imshow(zDMgrid.T,cmap=cmx,origin='lower', interpolation='None',aspect=aspect)
    
    ###### gets decent axis labels, down to 1 decimal place #######
    ax=plt.gca()
    labels = [item.get_text() for item in ax.get_xticklabels()]
    for i in np.arange(len(labels)):
        labels[i]=labels[i][0:4]
    ax.set_xticklabels(labels)
    labels = [item.get_text() for item in ax.get_yticklabels()]
    for i in np.arange(len(labels)):
        if '.' in labels[i]:
            labels[i]=labels[i].split('.')[0]
    ax.set_yticklabels(labels)
    ax.yaxis.labelpad = 0
    
    # plots contours i there
    cls=[":","--","-","--",":"]
    if conts:
        plt.ylim(0,ndm-1)
        for i in np.arange(nc):
            j=int(nc-i-1)
            plt.plot(np.arange(nz),carray[j,:],label=str(conts[j]),color='white',linestyle=cls[i])
        l=plt.legend(loc='lower right',fontsize=12)
        #l=plt.legend(bbox_to_anchor=(0.2, 0.8),fontsize=8)
        for text in l.get_texts():
                text.set_color("white")
    
    
    # limit to a reasonable range if logscale
    themax=zDMgrid.max()
    themin=int(themax-4)
    themax=int(themax)
    plt.clim(themin,themax)
    
    cbar=plt.colorbar(im,fraction=0.046, shrink=1.2,aspect=15,pad=0.05)
    cbar.set_label(label)
    plt.clim(-4,0)
    plt.tight_layout()
    
    plt.savefig(name)
    plt.title(title+str(H0))
    if showplot:
        plt.show()
    plt.close()	

def plot_grid_2(zDMgrid,zvals,dmvals,
                zmax=1,DMmax=1000,norm=0,log=True,name='temp.pdf',label='$\\log_{10}p(DM_{\\rm EG},z)$',project=False,conts=False,
                FRBZ=None,FRBDM=None,Aconts=False,
                Macquart=None,title="Plot",
                H0=None,showplot=False):
    """
    Very complicated routine for plotting 2D zdm grids 

    Args:
        zDMgrid ([type]): [description]
        zvals ([type]): [description]
        dmvals ([type]): [description]
        zmax (int, optional): [description]. Defaults to 1.
        DMmax (int, optional): [description]. Defaults to 1000.
        norm (int, optional): [description]. Defaults to 0.
        log (bool, optional): [description]. Defaults to True.
        name (str, optional): [description]. Defaults to 'temp.pdf'.
        label (str, optional): [description]. Defaults to '$\log_{10}p(DM_{\rm EG},z)$'.
        project (bool, optional): [description]. Defaults to False.
        conts (bool, optional): [description]. Defaults to False.
        FRBZ ([type], optional): [description]. Defaults to None.
        FRBDM ([type], optional): [description]. Defaults to None.
        Aconts (bool, optional): [description]. Defaults to False.
        Macquart (state, optional): state object.  Used to generat the Maquart relation.
            Defaults to None.
        title (str, optional): [description]. Defaults to "Plot".
        H0 ([type], optional): [description]. Defaults to None.
        showplot (bool, optional): [description]. Defaults to False.
    """
    if H0 is None:
        H0 = cos.cosmo.H0
    cmx = plt.get_cmap('cubehelix')
    
    ##### imshow of grid #######
    
    # we protect these variables
    zDMgrid=np.copy(zDMgrid)
    zvals=np.copy(zvals)
    dmvals=np.copy(dmvals)
    
    if (project):
        plt.figure(1, figsize=(8, 8))
        left, width = 0.1, 0.65
        bottom, height = 0.1, 0.65
        gap=0.02
        woff=width+gap+left
        hoff=height+gap+bottom
        dw=1.-woff-gap
        dh=1.-hoff-gap
        
        delta=1-height-bottom-0.05
        gap=0.11
        rect_2D = [left, bottom, width, height]
        rect_1Dx = [left, hoff, width, dh]
        rect_1Dy = [woff, bottom, dw, height]
        rect_cb = [woff,hoff,dw*0.5,dh]
        ax1=plt.axes(rect_2D)
        axx=plt.axes(rect_1Dx)
        axy=plt.axes(rect_1Dy)
        acb=plt.axes(rect_cb)
        #axx.xaxis.set_major_formatter(NullFormatter())
        #axy.yaxis.set_major_formatter(NullFormatter())
    else:
        plt.figure()
        #rect_2D=[0,0,1,1]
        ax1=plt.axes()
    
    plt.sca(ax1)
    
    plt.xlabel('z')
    plt.ylabel('${\\rm DM}_{\\rm EG}$')
    plt.title(title+str(H0))
    
    nz,ndm=zDMgrid.shape
    
    
    ixmax=np.where(zvals > zmax)[0]
    if len(ixmax) >0:
        zvals=zvals[:ixmax[0]]
        nz=zvals.size
        zDMgrid=zDMgrid[:ixmax[0],:]
    
    # sets contours according to norm
    if Aconts:
        slist=np.sort(zDMgrid.flatten())
        cslist=np.cumsum(slist)
        cslist /= cslist[-1]
        nAc=len(Aconts)
        alevels=np.zeros([nAc])
        for i,ac in enumerate(Aconts):
            # cslist is the cumulative probability distribution
            # Where cslist > ac determines the integer locations
            #    of all cells exceeding the threshold
            # The first in this list is the first place exceeding
            #    the threshold
            # The value of slist at that point is the
            #    level of the countour to draw
            iwhich=np.where(cslist > ac)[0][0]
            alevels[i]=slist[iwhich]
        
    ### generates contours *before* cutting array in DM ###
    ### might need to normalise contours by integer lengths, oh well! ###
    if conts:
        nc = len(conts)
        carray=np.zeros([nc,nz])
        for i in np.arange(nz):
            cdf=np.cumsum(zDMgrid[i,:])
            cdf /= cdf[-1]
            
            for j,c in enumerate(conts):
                less=np.where(cdf < c)[0]
                
                if len(less)==0:
                    carray[j,i]=0.
                    dmc=0.
                    il1=0
                    il2=0
                else:
                    il1=less[-1]
                    
                    if il1 == ndm-1:
                        il1=ndm-2
                    
                    il2=il1+1
                    k1=(cdf[il2]-c)/(cdf[il2]-cdf[il1])
                    dmc=k1*dmvals[il1]+(1.-k1)*dmvals[il2]
                    carray[j,i]=dmc
                
        ddm=dmvals[1]-dmvals[0]
        carray /= ddm # turns this into integer units for plotting
        
    iymax=np.where(dmvals > DMmax)[0]
    if len(iymax)>0:
        dmvals=dmvals[:iymax[0]]
        zDMgrid=zDMgrid[:,:iymax[0]]
        ndm=dmvals.size
    
    # currently this is "per cell" - now to change to "per DM"
    # normalises the grid by the bin width, i.e. probability per bin, not probability density
    ddm=dmvals[1]-dmvals[0]
    dz=zvals[1]-zvals[0]
    if norm==1:
        zDMgrid /= ddm
        if Aconts:
            alevels /= ddm
    if norm==2:
        xnorm=np.sum(zDMgrid)
        zDMgrid /= xnorm
        if Aconts:
            alevels /= xnorm
    
    if log:
        # checks against zeros for a log-plot
        orig=np.copy(zDMgrid)
        zDMgrid=zDMgrid.reshape(zDMgrid.size)
        setzero=np.where(zDMgrid==0.)
        zDMgrid=np.log10(zDMgrid)
        zDMgrid[setzero]=-100
        zDMgrid=zDMgrid.reshape(nz,ndm)
        if Aconts:
            alevels=np.log10(alevels)
    else:
        orig=zDMgrid
    
    # gets a square plot
    aspect=nz/float(ndm)
    
    # sets the x and y tics	
    xtvals=np.arange(zvals.size)
    everx=int(zvals.size/5)
    plt.xticks(xtvals[everx-1::everx],zvals[everx-1::everx])

    ytvals=np.arange(dmvals.size)
    every=int(dmvals.size/5)
    plt.yticks(ytvals[every-1::every],dmvals[every-1::every])
    
    im=plt.imshow(zDMgrid.T,cmap=cmx,origin='lower', interpolation='None',aspect=aspect)
    
    if Aconts:
        styles=['--','-.',':']
        ax=plt.gca()
        cs=ax.contour(zDMgrid.T,levels=alevels,origin='lower',colors="white",linestyles=styles)
        #plt.clim(0,2e-5)
        #ax.clabel(cs, cs.levels, inline=True, fontsize=10,fmt=['0.5','0.1','0.01'])
    ###### gets decent axis labels, down to 1 decimal place #######
    ax=plt.gca()
    labels = [item.get_text() for item in ax.get_xticklabels()]
    for i in np.arange(len(labels)):
        labels[i]=labels[i][0:4]
    ax.set_xticklabels(labels)
    labels = [item.get_text() for item in ax.get_yticklabels()]
    for i in np.arange(len(labels)):
        if '.' in labels[i]:
            labels[i]=labels[i].split('.')[0]
    ax.set_yticklabels(labels)
    ax.yaxis.labelpad = 0
    
    # plots contours i there
    if conts:
        plt.ylim(0,ndm-1)
        for i in np.arange(nc):
            j=int(nc-i-1)
            plt.plot(np.arange(nz),carray[j,:],label=str(conts[j]),color='white')
        l=plt.legend(loc='upper left',fontsize=8)
        #l=plt.legend(bbox_to_anchor=(0.2, 0.8),fontsize=8)
        for text in l.get_texts():
                text.set_color("white")

    if Macquart is not None:
        # Note this is the Median for the lognormal, not the mean
        muDMhost=np.log(10**Macquart.host.lmean)
        sigmaDMhost=np.log(10**Macquart.host.lsigma)
        meanHost = np.exp(muDMhost + sigmaDMhost**2/2.)
        medianHost = np.exp(muDMhost) 
        print(f"Host: mean={meanHost}, median={medianHost}")
        plt.ylim(0,ndm-1)
        plt.xlim(0,nz-1)
        zmax=zvals[-1]
        nz=zvals.size
        #DMbar, zeval = igm.average_DM(zmax, cumul=True, neval=nz+1)
        DM_cosmic = pcosmic.get_mean_DM(zvals, Macquart)
        
        #idea is that 1 point is 1, hence...
        zeval = zvals/dz
        DMEG_mean = (DM_cosmic+meanHost)/ddm
        DMEG_median = (DM_cosmic+medianHost)/ddm
        plt.plot(zeval,DMEG_mean,color='blue',linewidth=2,
                 label='Macquart relation (mean)')
        plt.plot(zeval,DMEG_median,color='blue',
                 linewidth=2, ls='--',
                 label='Macquart relation (median)')
        l=plt.legend(loc='lower right',fontsize=12)
        #l=plt.legend(bbox_to_anchor=(0.2, 0.8),fontsize=8)
        #for text in l.get_texts():
            #	text.set_color("white")
    
    # limit to a reasonable range if logscale
    if log:
        themax=zDMgrid.max()
        themin=int(themax-4)
        themax=int(themax)
        plt.clim(themin,themax)
    
    ##### add FRB host galaxies at some DM/redshift #####
    if FRBZ is not None:
        iDMs=FRBDM/ddm
        iZ=FRBZ/dz
        plt.plot(iZ,iDMs,'ro',linestyle="")
        
    # do 1-D projected plots
    if project:
        plt.sca(acb)
        cbar=plt.colorbar(im,fraction=0.046, shrink=1.2,aspect=20,pad=0.00,cax = acb)
        cbar.ax.tick_params(labelsize=6)
        cbar.set_label(label,fontsize=8)
        
        axy.set_yticklabels([])
        #axy.set_xticklabels([])
        #axx.set_yticklabels([])
        axx.set_xticklabels([])
        yonly=np.sum(orig,axis=0)
        xonly=np.sum(orig,axis=1)
        
        axy.plot(yonly,dmvals) # DM is the vertical axis now
        axx.plot(zvals,xonly)
        
        # if plotting DM only, put this on the axy axis showing DM distribution
        if FRBDM is not None:
            hvals=np.zeros(FRBDM.size)
            for i,DM in enumerate(FRBDM):
                hvals[i]=yonly[np.where(dmvals > DM)[0][0]]
            
            axy.plot(hvals,FRBDM,'ro',linestyle="")
            for tick in axy.yaxis.get_major_ticks():
                        tick.label.set_fontsize(6)
            
        if FRBZ is not None:
            hvals=np.zeros(FRBZ.size)
            for i,Z in enumerate(FRBZ):
                hvals[i]=xonly[np.where(zvals > Z)[0][0]]
            axx.plot(FRBZ,hvals,'ro',linestyle="")
            for tick in axx.xaxis.get_major_ticks():
                        tick.label.set_fontsize(6)
    else:
        cbar=plt.colorbar(im,fraction=0.046, shrink=1.2,aspect=15,pad=0.05)
        cbar.set_label(label)
        plt.tight_layout()
    
    plt.savefig(name)
    if showplot:
        plt.show()
    plt.close()

def plot_grid(grid,zvals,dmvals,showplot=False):
    ''' Plots a simple 2D grid '''
    cmx = plt.get_cmap('cubehelix')
    
    plt.figure()
    plt.zlabel('z')
    plt.ylabel('DM_{\\rm EG}')
    
    plt.xtics(np.arange(zvals.size)[::10],zvals[::10])
    plt.ytics(np.arange(dmvals.size)[::10],dmvals[::10])
    plt.imshow(grid,extent=(zvals[0],zvals[-1],dmvals[0],dmvals[-1]),origin='lower',cmap=cmx)
    cbar=plt.colorbar()
    cbar.set_label('$p(DM_{\\rm EG}|z)')
    if showplot:
        plt.show()

def plot_efficiencies_paper(survey,savename,label):
    ''' Plots a final version of efficiencies for the purpose of the paper '''
    dm=survey.DMlist
    eff1s=survey.get_efficiency(dm,model="Quadrature",dsmear=True)
    eff1mean=np.copy(survey.mean_efficiencies)
    eff2s=survey.get_efficiency(dm,model="Sammons",dsmear=True)
    eff2mean=np.copy(survey.mean_efficiencies)
    DMobs=survey.DMs
    NDM=DMobs.size
    DMx=np.zeros([NDM])
    DMy=np.zeros([NDM])
    for i,DMo in enumerate(DMobs):
        pos=np.where(dm > DMo)[0][0]
        DMy[i]=eff1s[i,pos]
        DMx[i]=dm[pos]
    
    
    plt.figure()
    plt.ylim(0,1)
    
    plt.text(1500,0.9,label)
    
    eff=survey.efficiencies
    if "ID" in survey.frbs:
        labels=survey.frbs["ID"]
    else:
        labels=np.arange(survey.NFRB)
    
    plt.xlabel('DM [pc cm$^{-3}$]')
    plt.ylabel('Efficiency $\\epsilon$')#\\dfrac{F_{\\rm 1\,ms}}{F_{\\rm th}}$')
    
    ls=['-',':','--','-.']
    
    for i in np.arange(survey.NFRB):
        ils=int(i/10)
        if i==0:
            plt.plot(dm,eff1s[i],linestyle=':',color='black',label='$\\epsilon_i$')
        else:
            plt.plot(dm,eff1s[i],linestyle=':',color='black')
        #plt.plot(dm,eff2s[i],linestyle=ls[ils],lw=1)
    plt.plot(dm,eff1mean,linewidth=3,color='blue',ls='-',label='$\\bar{\\epsilon}$')
    plt.plot(DMx,DMy,'ro',label='${\\rm DM}_i$')
    
    #plt.plot(dm,eff2mean,linewidth=1,color='black',ls='-')
    ncol=int((survey.NFRB+9)/10)
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(savename)
    plt.close()

def plot_efficiencies(survey,savename='Plots/efficiencies.pdf',showplot=False):
    """ Plots efficiency as function of DM """
    plt.figure()
    
    eff=survey.efficiencies
    dm=survey.DMlist
    if "ID" in survey.frbs:
        labels=survey.frbs["ID"]
    else:
        labels=np.arange(survey.NFRB)
    
    plt.xlabel('DM [pc cm$^{-3}$]')
    plt.ylabel('Efficiency $\\epsilon$')
    
    ls=['-',':','--','-.']
    
    for i in np.arange(survey.NFRB):
        ils=int(i/10)
        plt.plot(dm,eff[i],label=labels[i],linestyle=ls[ils])
    plt.plot(dm,survey.mean_efficiencies,linewidth=2,color='black',ls='-')
    ncol=int((survey.NFRB+9)/10)
    plt.legend(loc='upper right',fontsize=min(14,200./survey.NFRB),ncol=ncol)
    plt.tight_layout()
    plt.savefig(savename)
    if showplot:
        plt.show()
    plt.close()


def plot_beams(prefix):
    ''' Plots something to do with beams '''
    logb,omega_b=beams.load_beam(prefix)
    total=np.sum(omega_b)
    print("Total length of histogram is ",omega_b.size)
    
    # rate of -1.5
    b=10**logb
    rate=omega_b*b**1.5
    nbins=10
    b2,o2=beams.simplify_beam(logb,omega_b,nbins)
    print(b2,o2)
    
    # note that omega_b is just unscaled total solid angle
    plt.figure()
    plt.xlabel('$B$')
    plt.ylabel('$\\Omega(B)$/bin')
    plt.yscale('log')
    plt.xscale('log')
    plt.plot(b,omega_b,label='original_binning')
    plt.plot(b2,o2,'ro',label='simplified',linestyle=':')
    plt.plot(b,rate,label='Relative rate')
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig('Plots/lat50_beam.pdf')
    plt.close()
    
    
    
    crate=np.cumsum(rate)
    crate /= crate[-1]
    plt.figure()
    plt.xlabel('$\\log_{10}(B)$')
    plt.ylabel('cumulative rate')
    #plt.yscale('log')
    plt.plot(logb,crate)
    plt.tight_layout()
    plt.savefig('Plots/crate_lat50_beam.pdf')
    plt.close()
    crate=np.cumsum(rate)

def process_missing_pfile(pfile,number,howmany):
    ''' searches for missing data in cube output and iterates for that only '''
    NPARAMS=8
    current=np.zeros([1,NPARAMS])
    last=np.zeros([NPARAMS])
    this=np.zeros([NPARAMS])
    n=0 # counts number of lines in this calculation
    new=0 # counts new significant calculations
    count=0 # counts total lines
    
    # the range of calculations to do. These are *inclusive* values
    start=(number-1)*howmany+1
    stop=number*howmany
    print("Calculated start and stop as ",start,stop)
    
    max_number=np.zeros([howmany*11,NPARAMS])
    
    #I am now testing how many in the file in total
    
    with open(pfile) as pf:
        
        for line in pf:
            #if count==NPARAMS:
            #    break
            vals=line.split()
            for j,v in enumerate(vals):
                this[j]=float(v)
                
            # tests to see if this is a serious calculation
            for j in np.arange(NPARAMS-1):
                if j==4:
                    continue
                if this[j] != last[j]:
                    new += 1
                    break
            
            # tests to see if we have gone far enough
            if new > stop:
                break
            
            # test to see if we now do this
            if new >= start:
                max_number[n,:]=this[:]
                n += 1
            
            
            # sets last to this one
            last[:]=this[:]
            
            count += 1
            
    if n==0:
        print("Reached the end of the file, exiting")
        exit()
    # concatenate max number to true size
    todo=max_number[0:n,:]
    starti=count-n+1
    return todo,starti

def process_pfile(pfile):
    ''' used for cube.py to input multi-dimensional grid to iterate over'''
    NPARAMS=8
    mins=np.zeros([NPARAMS])
    maxs=np.zeros([NPARAMS])
    Nits=np.zeros([NPARAMS],dtype='int')
    with open(pfile) as pf:
        count=0
        for line in pf:
            if count==NPARAMS:
                break
            vals=line.split()
            mins[count]=float(vals[0])
            maxs[count]=float(vals[1])
            Nits[count]=int(vals[2])
            count += 1
    return mins,maxs,Nits
