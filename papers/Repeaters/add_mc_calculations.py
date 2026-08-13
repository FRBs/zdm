""" 
This script reads in optimum combinations of Rmax, Rmin, gamma
and generates O~1000 MC iterations to determine how may rapid repeaters
we find.

"""
import os
from pkg_resources import resource_filename
from zdm import cosmology as cos
from zdm import misc_functions
from zdm import parameters
from zdm import survey
from zdm import pcosmic
from zdm import iteration as it
from zdm.MC_sample import loading
from zdm import io
from zdm import repeat_grid as rep

import utilities as ute
import states as st

import pickle
import numpy as np
from zdm import survey
from matplotlib import pyplot as plt

import scipy as sp
from scipy.stats import poisson

import matplotlib
import time
from zdm import beams

Planck_H0=67.4

matplotlib.rcParams['image.interpolation'] = None

defaultsize=14
ds=4
font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : defaultsize}
matplotlib.rc('font', **font)

def main(FC=1.0):
    
    # gets the possible states for evaluation
    states,names=st.get_states()
    
    sdir = os.path.join(resource_filename('zdm','data/Surveys/'),'CHIME/')
    
    outdir = 'Rfitting39_'+str(FC)+'/'
    for i,state in enumerate(states):
        if i==1:
            break
        outfile = outdir+'FC39'+str(FC)+'converge_set_'+str(i)+'_output.npz'
        outfile2 = outdir+'mc_FC39'+str(FC)+'converge_set_'+str(i)+'_output.npz'
        
        add_mc(state,outfile,outfile2)

def add_mc(state,outfile,outfile2,Nbin=6,verbose=False,Rstar = 0.3,\
    Nacc = 0.1):
    """
    Find the best-fitting parameters in an intelligent way
    
    Steps:
    #1: Find R*, the critical rate producing repeat distributions
        Rstar begins at 0.3 as a guess, but should be set from previous state as a best guess
        
        Nacc is the accuracy in Nrepeaters predicted. About 10% of 1 sigma, i.e. 0.1 * sqrt(17)
        
    
    """
    
    
    bdir='Nbounds'+str(Nbin)+'/'
    beams.beams_path = os.path.join(resource_filename('zdm','data/BeamData/CHIME/'),bdir)
    
    ############## loads CHIME surveys and grids #############
    
    # old implementation
    # defines list of surveys to consider, together with Tpoint
    sdir = os.path.join(resource_filename('zdm','data/Surveys/'),'CHIME/')
    
    
    # loads beam data
    bdir='Nbounds'+str(Nbin)+'/'
    beams.beams_path = os.path.join(resource_filename('zdm','data/BeamData/CHIME/'),bdir)
    bounds = np.load(beams.beams_path+'bounds.npy')
    solids = np.load(beams.beams_path+'solids.npy')
    
    
    ss=[]
    gs=[]
    nrs=[]
    nss=[]
    irs=[]
    iss=[]
    NR=0
    NS=0
    # we initialise surveys and grids
    Cnreps = np.array([])
    for ibin in np.arange(Nbin):
        
        name = "CHIME_decbin_"+str(ibin)+"_of_"+str(Nbin)
        
        s,g = ute.survey_and_grid(survey_name=name,NFRB=None,sdir=sdir,init_state=state)
            #with open(savefile, 'wb') as output:
            #    pickle.dump(s, output, pickle.HIGHEST_PROTOCOL)
            #    pickle.dump(g, output, pickle.HIGHEST_PROTOCOL)
        
        ss.append(s)
        gs.append(g)
        
        ir = np.where(s.frbs['NREP']>1)[0]
        nr=len(ir)
        irs.append(ir)
        nreps = s.frbs['NREP'][ir]
        Cnreps = np.concatenate((Cnreps,nreps))
        i_s = np.where(s.frbs['NREP']==1)[0]
        ns=len(i_s)
        iss.append(i_s)
        
        NR += nr
        NS += ns
        
        nrs.append(nr)
        nss.append(ns)
    
    if verbose:
        print("We have calculated a total of ",NR,NS," repeat/single bursts")
        print("Per declination, this is ",nrs," reps and ",nss," singles")
    
    
    ############## iterates to find Rstar ########
    rgs=[None]*Nbin # 6 is number of dec bins
    
    # gets CHIME declination histograms
    Csxvals,Csyvals,Crxvals,Cryvals = ute.get_chime_rs_dec_histograms(newdata=False)
    sdmegs,rdmegs,sdecs,rdecs = ute.get_chime_dec_dm_data(newdata=False,sort=True)
    
    
    ############## Set up results grid ##############
    
    # we now have an initial estmate for Rstar. From hereon, we make a grid in Rmin, Rmax, and
    # optimise gamma for each one
    
    # furthermore, Rmaxes and Rmins are meant to be symmetrical about Rstar
    data = np.load(outfile)
    lps=data['arr_0']
    lns=data['arr_1']
    ldms=data['arr_2']
    lpNs=data['arr_3']
    Nrs=data['arr_4']
    Rmins=data['arr_5']
    Rmaxes=data['arr_6']
    Rgammas=data['arr_7']
    lskps=data['arr_8']
    lrkps=data['arr_9']
    ltdm_kss=data['arr_10']
    ltdm_krs=data['arr_11']
    lprod_bin_dm_krs=data['arr_12']
    lprod_bin_dm_kss=data['arr_13']
    Rstar = data['arr_14']
    Cprimes = data['arr_15']
    
    # Here, Rmin *and* Rmax are both increasing. Which means we begin very heavily weighted to Rmin
    # Thus we begin with a very flat Rgamma. We also set up initial increments
    
    NMChist = 100
    mcrs=np.linspace(1.5,0.5+NMChist,NMChist) # from 1.5 to 100.5 inclusive
    Chist,bins = np.histogram(Cnreps,bins=mcrs)
    
    MChs=np.zeros([Rgammas.size,Rmaxes.size,NMChist-1])
    ranks=np.zeros([Rgammas.size,Rmaxes.size])
    
    t0=time.time()
    for i,Rgamma in enumerate(Rgammas):
        
        for j,Rmax in enumerate(Rmaxes):
            
            Rmin = Rmins[i,j]
            
            #if not Rmin == 1e-8:
            #    print("Skipping Rmin of ",Rmin)
            #    continue
            #else:
            #    print("Found ",i,j,Rmin," analysing...")
            if Nrs[i,j] > 20:
                Rmult = 1000.*17./Nrs[i,j]
            else:
                Rmult = 1000.
            print("Found ",i,j," using Rmult of ",Rmult)
            verbose = True
            rank,mch=perform_mc_simulation(ss,gs,Rmin,Rmax,Rgamma,\
                    rgs,bounds,\
                    mcrs,Chist,Cnreps,Rmult=Rmult,verbose=False)
            
            MChs[i,j,:] = mch
            ranks[i,j]=rank
            t1=time.time()
            print("Result ",rank," Iteration ",i,j," time taken is ",t1-t0)
        
    np.savez(outfile2,lps,lns,ldms,lpNs,Nrs,Rmins,Rmaxes,Rgammas,lskps,lrkps,\
                    ltdm_kss,ltdm_krs,lprod_bin_dm_krs,lprod_bin_dm_kss,Rstar,\
                    Cprimes,mcrs,MChs,ranks)
       
      
    ####### now inside repeater loop ###### - new function to loop over rep properties
def perform_mc_simulation(ss,gs,Rmin,Rmax,Rgamma,\
        rgs,bounds,mcrs,Chist,Cnreps,Rmult=1.,FC=1.,verbose=False,Nbin=6,
        doresample=True):
    """
    
    FC is fraction of CHIME singles explained by repeaters. <= 1.0.
    Don't set this to less than 1., we can't reproduce the repeaters we have!
    
    mcrs are the rates for histogram binning of results
    Chist is a histogram of CHIME repetition rates
    Cnreps are the actual repetition rates of CHIME bursts
    
    
    Resample produces a distribution of likelihoods based on the MC itself
    It does not re-calculate the overall histogram however, so the evaluation
    and MC sampels are not independent, but they should be correlated by 0.1%
    at most.
    """
    
    numbers = np.array([])
    for ibin in np.arange(Nbin):
        # calculates repetition info
        g=gs[ibin]
        s=ss[ibin]
        g.state.rep.Rmin = Rmin
        g.state.rep.Rmax = Rmax
        g.state.rep.Rgamma = Rgamma
        
        savefile = 'Rfitting/rg_'+str(Nbin)+'_'+str(ibin)+'.pkl'
        
        if rgs[ibin] is not None:
            rg = rgs[ibin]
            rg.update(Rmin=Rmin,Rmax=Rmax,Rgamma=Rgamma)
        #elif os.path.exists(savefile):
        #    with open(savefile, 'rb') as infile:
        #        rg=pickle.load(infile)
        #        rgs[ibin]=rgs
        else:
            rg = rep.repeat_Grid(g,Tfield=s.TOBS,Nfields=1,opdir=None,bmethod=2,\
                Exact=False,MC=Rmult)
            numbers = np.append(numbers,rg.MC_numbers)
            # I have turned off saving, it produces 1GB of output per loop!
            #with open(savefile, 'wb') as output:
            #    pickle.dump(rg, output, pickle.HIGHEST_PROTOCOL)
        
    
    # h is the histogram of repetition rates (length Nbins)
    # cs is the cumulative sum (length Nbins)
    # mcrs are the bins: length Nbins+1
    
    nM = len(numbers) # number of repetitions
    nC = len(Cnreps)
    Mh,b=np.histogram(numbers,bins=mcrs)
    bcs = b[:-1]+(b[1]-b[0])/2.
    # find the max set below which there are no MC zeros
    firstzero = np.where(Mh==0)[0]
    if len(firstzero) > 0:
        firstzero = firstzero[0]
        f = np.polyfit(np.log10(bcs[:firstzero]),np.log10(Mh[:firstzero]),1,\
            w=1./np.log10(bcs[:firstzero]**0.5))
        #    error = np.log10(bcs[:firstzero]**0.5))
    else:
        f = np.polyfit(np.log10(bcs),np.log10(Mh),1,\
            w=1./np.log10(bcs**0.5))
    
    fitv = 10**np.polyval(f,np.log10(bcs)) *nC/nM
    
    # normalise h to number of CHIME repeaters
    # Note: for overflow of histogram, the actual sum will be reduced
    Mh = Mh*nC/nM
    Nmissed = nC - np.sum(Mh) # expectation value for missed fraction
    
    # Above certain value, replace with polyval expectation
    tooLow = np.where(Mh < 10)[0]
    copyMh = np.copy(Mh)
    copyMh[tooLow] = fitv[tooLow]
    
    plot=False
    if plot:
        plt.figure()
        plt.hist(numbers,bins=mcrs,weights=np.full([nM],nC/nM),label='MC')
        plt.hist(Cnreps,bins=mcrs,label='CHIME',alpha=0.5)
        plt.plot(bcs,fitv,label='polyfit')
        plt.scatter(bcs[tooLow],Mh[tooLow],marker='x',s=10.)
        plt.legend()
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('$N_{\\rm reps}$')
        plt.ylabel('$N_{\\rm frb}$')
        plt.tight_layout()
        plt.savefig('MChistogram.pdf')
        plt.close()
    
        # normalise cumsum to *total* number of bursts.
        Mch=np.cumsum(Mh)
        Cch=np.cumsum(Chist)
        polyMch=np.cumsum(fitv)
        
        plt.figure()
        plt.plot(mcrs[:-1]+0.5,Mch,label='MC')
        plt.plot(mcrs[:-1]+0.5,Cch,label='CHIME')
        plt.plot(mcrs[:-1]+0.5,polyMch,label='polyfit')
        
        plt.xlabel('$N_{\\rm reps}$')
        plt.ylabel('Cumulative $N_{\\rm frb}$')
        plt.legend()
        plt.xscale('log')
        #plt.yscale('log')
        plt.tight_layout()
        plt.savefig('cumulativeMC.pdf')
        plt.close()
    
    
    long_slPN = poisson_expectations(Chist,copyMh)
    
    # now does this over 3 bins only: 2 reps, 3-10 reps, more than 10
    shortC = np.array([Chist[0],np.sum(Chist[1:8]),np.sum(Chist[9:])])
    shortM = np.array([Mh[0],np.sum(Mh[1:8]),np.sum(Mh[9:])+Nmissed])
    short_slPN = poisson_expectations(shortC,shortM)
    
    
    # now does this over 3 bins only: 2-3, 4-8, 9-20 reps, 3-10 reps, more than 10
    mediumC = np.array([np.sum(Chist[0:2]),np.sum(Chist[2:7]),np.sum(Chist[7:20]),
        np.sum(Chist[20:50]),np.sum(Chist[50:])])
    mediumM = np.array([np.sum(Mh[0:2]),np.sum(Mh[2:7]),np.sum(Mh[7:20]),
        np.sum(Mh[20:50]),np.sum(Mh[50:])+Nmissed])
    medium_slPN = poisson_expectations(mediumC,mediumM)
    
    # does chi-square test
    if doresample:
        MCrank = resample(nC,mcrs,numbers,copyMh,long_slPN,plot=True)
    
    #return short_slPN,medium_slPN,long_slPN,Mh
    return MCrank,Mh


def resample(Nrep, bins, nbursts, copyMh, realll,plot=False):
    """
    Samples Nrep repetitions from numbers,
    and evaluates the likelihood against copyMh (the histogram)
    
    Inputs:
        Nrep (int): number of repeaters in the real CHIME sample
        
        bins (np: float array): MC bins over which the number of repeaters is evaluated
        
        Nbursts (np.int array) a numpy array of the number of bursts for MC
            simulated repeaters
        
    
        copyMh is the smoothed histogram of burst number
    
    realll is the actual log-likelihood of the observed CHIME sample
    
    
    """
    NMC = len(nbursts)
    Nsamps = int(len(nbursts)/Nrep)
    # assigns a random number to each FRB, in order to choose
    # Nrep from them
    rands = np.random.rand(NMC)
    # assigns 0th rand to 50th place, 1st to 17th, 3rd to 1000th etc
    order = np.argsort(rands)
    # we now have randomised the rep numbers
    newnum = nbursts[order]
    #print("Found ",NMC," MC vals, Nrep = ",Nrep," hence sampling ",Nsamps," times")
    #### generates log-likelihood distribution
    mclls = np.zeros([Nsamps])
    for i in np.arange(Nsamps):
        this_samp = newnum[i*Nrep:(i+1)*Nrep]
        h,b = np.histogram(this_samp,bins=bins)
        mclls[i] = poisson_expectations(h,copyMh)
        #print("Sample ",i,"Reps ",this_samp," ll ",mclls[i])
    
    # orders the distribution
    mclls = np.sort(mclls)
    
    # calculates the rank, i.e. what fraction of samples
    # have a lower ll than the real evaluation
    if realll > mclls[-1]:
        rank=1.
    else:
        rank = np.where(realll < mclls)[0][0]
        rank /= Nsamps
    
    #print("Calculated rank to be ",rank)
    
    
    if plot:
        plt.figure()
        plt.hist(mclls,label='MC distribution')
        plt.xlabel('log ll')
        plt.ylabel('pll')
        plt.plot([realll,realll],[0,10],color='red',linewidth=3,label='Observed')
        plt.legend()
        plt.tight_layout()
        plt.savefig('resampleMC.pdf')
    
    return rank
    
def poisson_expectations(Creps,Mreps):
    """
    Computes a likelihood value for observing Creps CHIME
    repeaters against a simulation of Mreps Monte Carlo
    values.
    Creps are the counts for CHIME repeaters
    """
    
    # scale MC repeaters to number observed in CHIME - normalisation
    Mreps *= np.sum(Creps)/np.sum(Mreps)
    
    # calculates Poissonian likelihoods for observations
    # of Creps against expectations of Mreps
    PN = poisson.pmf(Creps,Mreps)
    lPN = np.log10(PN)
    slPN = np.sum(lPN)
    
    return slPN
    # is this dominated by... cats?
    
def int_cdf(x,cdf):
    """
    Returns values of cdf for x
    Relies on x being an integer
    0.001 ensures we are immune to minor negative fluctuations
    """
    ix = (x+0.001-2) # because a repeater with value 2 is in the 0th bin
    ix = ix.astype(int)
    vals = cdf[ix]
    return vals





main()
#for FC in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
#    main(FC=FC)

