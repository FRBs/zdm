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
from zdm.craco import loading
from zdm import io
from zdm import repeat_grid as rep

import utilities as ute

import pickle
import numpy as np
from zdm import survey
from matplotlib import pyplot as plt

import scipy as sp
from scipy.stats import poisson

import matplotlib
import time
from zdm import beams
beams.beams_path = '/Users/cjames/CRAFT/FRB_library/Git/H0paper/papers/Repeaters/BeamData/'

Planck_H0=67.4

matplotlib.rcParams['image.interpolation'] = None

defaultsize=14
ds=4
font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : defaultsize}
matplotlib.rc('font', **font)

def main():
    
    # gets the possible states for evaluation
    states,names=get_states()
    
    sdir = os.path.join(resource_filename('zdm','../'),'papers/Repeaters/Surveys')
    for i,state in enumerate(states):
        
        outfile = 'Rfitting/converge_set_'+str(i)+'_'+'_output.npz'
        outfile2 = 'Rfitting/mc_converge_set_'+str(i)+'_'+'_output.npz'
        if i==1:
            exit()
        add_mc(state,outfile,outfile2)
        exit() #only do this for one

def add_mc(state,outfile,outfile2,Nbin=6,verbose=False,Rstar = 0.3,\
    Nacc = 0.1):
    """
    Find the best-fitting parameters in an intelligent way
    
    Steps:
    #1: Find R*, the critical rate producing repeat distributions
        Rstar begins at 0.3 as a guess, but should be set from previous state as a best guess
        
        Nacc is the accuracy in Nrepeaters predicted. About 10% of 1 sigma, i.e. 0.1 * sqrt(17)
        
    
    """
    
    
    ############## loads CHIME surveys and grids #############
    
    # old implementation
    # defines list of surveys to consider, together with Tpoint
    sdir = os.path.join(resource_filename('zdm','../'),'papers/Repeaters/Surveys')
    
    
    bdir='Nbounds'+str(Nbin)+'/'
    beams.beams_path = os.path.join(resource_filename('zdm','../'),'papers/Repeaters/'+bdir)
    bounds = np.load(bdir+'bounds.npy')
    solids = np.load(bdir+'solids.npy')
    
    
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
        
        s,g = survey_and_grid(survey_name=name,NFRB=None,sdir=sdir,init_state=state)
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
    
    #data2 = np.load(outfile2)
    
    #mcrs=data2['arr_15']
    #MChs=data2['arr_16']
    #shortlPs=data2['arr_17']
    #mediumlPs=data2['arr_18']
    #longlPs = data2['arr_19']
    
    
    print("Rstar is ",Rstar)
    # Here, Rmin *and* Rmax are both increasing. Which means we begin very heavily weighted to Rmin
    # Thus we begin with a very flat Rgamma. We also set up initial increments
    
    NMChist = 100
    mcrs=np.linspace(1.5,0.5+NMChist,NMChist) # from 1.5 to 100.5 inclusive
    Chist,bins = np.histogram(Cnreps,bins=mcrs)
    
    MChs=np.zeros([Rgammas.size,Rmaxes.size,NMChist-1])
    shortlPs=np.zeros([Rgammas.size,Rmaxes.size])
    mediumlPs=np.zeros([Rgammas.size,Rmaxes.size])
    longlPs=np.zeros([Rgammas.size,Rmaxes.size])
    
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
            lS,lM,lL,mch=perform_mc_simulation(ss,gs,Rmin,Rmax,Rgamma,\
                    rgs,bounds,\
                    mcrs,Chist,Cnreps,Rmult=Rmult,verbose=False)
            
            MChs[i,j,:] = mch
            shortlPs[i,j]=lS
            mediumlPs[i,j]=lM
            longlPs[i,j]=lL
            t1=time.time()
            print("Result ",lS,lM,lL," Iteration ",i,j," time taken is ",t1-t0)
        
    np.savez(outfile2,lps,lns,ldms,lpNs,Nrs,Rmins,Rmaxes,Rgammas,lskps,lrkps,ltdm_kss,\
                    ltdm_krs,lprod_bin_dm_krs,lprod_bin_dm_kss,Rstar,mcrs,MChs,shortlPs,\
                    mediumlPs,longlPs)
       
      
    ####### now inside repeater loop ###### - new function to loop over rep properties
def perform_mc_simulation(ss,gs,Rmin,Rmax,Rgamma,\
        rgs,bounds,mcrs,Chist,Cnreps,Rmult=1.,FC=1.,verbose=False,Nbin=6):
    """
    
    FC is fraction of CHIME singles explained by repeaters. <= 1.0.
    Don't set this to less than 1., we can't reproduce the repeaters we have!
    
    mcrs are the rates for histogram binning of results
    Chist is a histogram of CHIME repetition rates
    Cnreps are the actual repetition rates of CHIME bursts
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
    
    nM = len(numbers)
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
    
    return short_slPN,medium_slPN,long_slPN,Mh


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

def shin_fit():
    """
    Returns best-fit parameters from Shin et al.
    https://arxiv.org/pdf/2207.14316.pdf
    
    """
    
    pset={}
    pset["lEmax"] = np.log10(2.38)+41.
    pset["alpha"] = -1.39
    pset["gamma"] = -1.3
    pset["sfr_n"] = 0.96
    pset["lmean"] = 1.93
    pset["lsigma"] = 0.41
    pset["lC"] = np.log10(7.3)+4.
    
    return pset

def james_fit():
    """
    Returns best-fit parameters from James et al 2022 (Hubble paper)
    """
    
    pset={}
    pset["lEmax"] = 41.63
    pset["alpha"] = -1.03
    pset["gamma"] = -0.948
    pset["sfr_n"] = 1.15
    pset["lmean"] = 2.22
    pset["lsigma"] = 0.57
    pset["lC"] = 1.963
    
    return pset



def read_extremes(infile='planck_extremes.dat',H0=Planck_H0):
    """
    reads in extremes of parameters from a get_extremes_from_cube
    """
    f = open(infile)
    
    sets=[]
    
    for pset in np.arange(6):
        # reads the 'getting' line
        line=f.readline()
        
        pdict={}
        # gets parameter values
        for i in np.arange(7):
            line=f.readline()
            words=line.split()
            param=words[0]
            val=float(words[1])
            pdict[param]=val
        pdict["H0"]=H0
        pdict["alpha"] = -pdict["alpha"] # alpha is reversed!
        sets.append(pdict)
        
        pdict={}
        # gets parameter values
        for i in np.arange(7):
            line=f.readline()
            words=line.split()
            param=words[0]
            val=float(words[1])
            pdict[param]=val
        pdict["H0"]=H0
        pdict["alpha"] = -pdict["alpha"] # alpha is reversed!
        sets.append(pdict)
    return sets

def set_state(pset,chime_response=True):
    """
    Sets the state parameters
    """
    
    state = loading.set_state(alpha_method=1)
    state_dict = dict(cosmo=dict(fix_Omega_b_h2=True))
    state.energy.luminosity_function = 2 # this is Schechter
    state.update_param_dict(state_dict)
    # changes the beam method to be the "exact" one, otherwise sets up for FRBs
    state.beam.Bmethod=3
    
    
    # updates to most recent best-fit values
    state.cosmo.H0 = 67.4
    
    if chime_response:
        state.width.Wmethod=0 #only a single width bin
        state.width.Wbias="CHIME"
    
    state.energy.lEmax = pset['lEmax']
    state.energy.gamma = pset['gamma']
    state.energy.alpha = pset['alpha']
    state.FRBdemo.sfr_n = pset['sfr_n']
    state.host.lsigma = pset['lsigma']
    state.host.lmean = pset['lmean']
    state.FRBdemo.lC = pset['lC']
    
    return state


def survey_and_grid(survey_name:str='CRAFT/CRACO_1_5000',
            init_state=None,
            state_dict=None, iFRB:int=0,
               alpha_method=1, NFRB:int=100, 
               lum_func:int=2,sdir=None):
    """ Load up a survey and grid for a CRACO mock dataset

    Args:
        init_state (State, optional):
            Initial state
        survey_name (str, optional):  Defaults to 'CRAFT/CRACO_1_5000'.
        NFRB (int, optional): Number of FRBs to analyze. Defaults to 100.
        iFRB (int, optional): Starting index for the FRBs.  Defaults to 0
        lum_func (int, optional): Flag for the luminosity function. 
            0=power-law, 1=gamma.  Defaults to 0.
        state_dict (dict, optional):
            Used to init state instead of alpha_method, lum_func parameters

    Raises:
        IOError: [description]

    Returns:
        tuple: Survey, Grid objects
    """
    
    # Init state
    if init_state is None:
        state = loading.set_state(alpha_method=alpha_method)
        # Addiitonal updates
        if state_dict is None:
            state_dict = dict(cosmo=dict(fix_Omega_b_h2=True))
            state.energy.luminosity_function = lum_func
        state.update_param_dict(state_dict)
    else:
        state = init_state
    
    # Cosmology
    cos.set_cosmology(state)
    cos.init_dist_measures()
    
    # get the grid of p(DM|z)
    zDMgrid, zvals,dmvals = misc_functions.get_zdm_grid(
        state, new=True, plot=False, method='analytic',
        datdir=resource_filename('zdm', 'GridData'),
        zlog=False,nz=500)

    ############## Initialise surveys ##############
    if sdir is not None:
        print("Searching for survey in directory ",sdir)
    else:
        sdir = os.path.join(resource_filename('zdm', 'craco'), 'MC_Surveys')
    
    
    isurvey = survey.load_survey(survey_name, state, dmvals,
                                 NFRB=NFRB, sdir=sdir, Nbeams=5,
                                 iFRB=iFRB)
    
    # generates zdm grid
    grids = misc_functions.initialise_grids(
        [isurvey], zDMgrid, zvals, dmvals, state, wdist=True)
    print("Initialised grid")

    # Return Survey and Grid
    return isurvey, grids[0]



def get_states():  
    """
    Gets the states corresponding to plausible fits to single CHIME data
    """
    psets=read_extremes()
    psets.insert(0,shin_fit())
    psets.insert(1,james_fit())
    
    
    # gets list of psets compatible (ish) with CHIME
    chime_psets=[4]
    chime_names = ["CHIME min $\\alpha$"]
    
    # list of psets compatible (ish) with zdm
    zdm_psets = [1,2,7,12]
    zdm_names = ["zDM best fit","zDM min $\\E_{\\rm max}$","zDM max $\\gamma$","zDM min $\sigma_{\\rm host}$"]
    
    names=[]
    # loop over chime-compatible state
    for i,ipset in enumerate(chime_psets):
        
        state=set_state(psets[ipset],chime_response=True)
        if i==0:
            states=[state]
        else:
            states.append(states)
        names.append(chime_names[i])
    
    for i,ipset in enumerate(zdm_psets):
        state=set_state(psets[ipset],chime_response=False)
        states.append(state)
        names.append(zdm_names[i])
    
    return states,names
    
main()
