
"""
This script generates MC samples for CRAFT CRACO.

"""


from pkg_resources import resource_filename
import os
import copy
import pickle
import sys
import scipy as sp
import numpy as np
import time
import argparse

from astropy.cosmology import Planck18

from zdm import cosmology as cos
from zdm import misc_functions
from zdm import parameters
from zdm import survey
from zdm import iteration as it
from zdm import beams
from zdm import misc_functions

from IPython import embed

#from zdm import zdm
#import pcosmic
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm
import matplotlib

from matplotlib.ticker import NullFormatter


matplotlib.rcParams['image.interpolation'] = None

defaultsize=14
ds=4
font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : defaultsize}
matplotlib.rc('font', **font)

def main():
        
    ############## Initialise parameters ##############
    state = parameters.State()

    # Variable parameters
    vparams = {}
    vparams['cosmo'] = {}
    vparams['cosmo']['H0'] = 67.74
    vparams['cosmo']['Omega_lambda'] = 0.685
    vparams['cosmo']['Omega_m'] = 0.315
    vparams['cosmo']['Omega_b'] = 0.044
    
    vparams['FRBdemo'] = {}
    vparams['FRBdemo']['alpha_method'] = 1
    vparams['FRBdemo']['source_evolution'] = 0
    
    vparams['beam'] = {}
    vparams['beam']['thresh'] = 0
    vparams['beam']['method'] = 2
    
    vparams['width'] = {}
    vparams['width']['logmean'] = 1.70267
    vparams['width']['logsigma'] = 0.899148
    vparams['width']['Wbins'] = 10
    vparams['width']['Wscale'] = 2
    
     # constants of intrinsic width distribution
    vparams['MW']={}
    vparams['MW']['DMhalo']=50
    
    vparams['host']={}
    vparams['energy'] = {}
    
    if vparams['FRBdemo']['alpha_method'] == 0:
        vparams['energy']['lEmin'] = 30
        vparams['energy']['lEmax'] = 41.7
        vparams['energy']['alpha'] = 1.55
        vparams['energy']['gamma'] = -1.09
        vparams['FRBdemo']['sfr_n'] = 1.67
        vparams['FRBdemo']['lC'] = 3.15
        vparams['host']['lmean'] = 2.11
        vparams['host']['lsigma'] = 0.53
    elif  vparams['FRBdemo']['alpha_method'] == 1:
        vparams['energy']['lEmin'] = 30
        vparams['energy']['lEmax'] = 41.4
        vparams['energy']['alpha'] = 0.65
        vparams['energy']['gamma'] = -1.01
        vparams['FRBdemo']['sfr_n'] = 0.73
        # NOTE: I have not checked what the best-fit value
        # of lC is for alpha method=1
        vparams['FRBdemo']['lC'] = 1 #not best fit, OK for a once-off
        
        vparams['host']['lmean'] = 2.18
        vparams['host']['lsigma'] = 0.48
        
    state.update_param_dict(vparams)
    
    ############## Initialise cosmology ##############
    cos.set_cosmology(state)
    cos.init_dist_measures()
    
      # get the grid of p(DM|z). See function for default values.
    # set new to False once this is already initialised
    zDMgrid, zvals,dmvals = misc_functions.get_zdm_grid(
        state, new=True, plot=False, method='analytic')
    
    ############## Initialise surveys ##############
    
   
    
    #These surveys combine time-normalised and time-unnormalised samples 
    NewSurveys=True
    #sprefix='Full' # more detailed estimates. Takes more space and time
    
    sprefix='craco_alpha' # faster - fine for max likelihood calculations, not as pretty
    
    if NewSurveys:
        surveys=[survey.load_survey('CRAFT_CRACO_MC_frbs_alpha1', state, dmvals)]
        
        if not os.path.isdir('Pickle'):
            os.mkdir('Pickle')
        with open('Pickle/'+sprefix+'surveys.pkl', 'wb') as output:
            pickle.dump(surveys, output, pickle.HIGHEST_PROTOCOL)
    else:
        with open('Pickle/'+sprefix+'surveys.pkl', 'rb') as infile:
            surveys=pickle.load(infile)
    craco=surveys[0]
    
    dirnames=['CRACO']
    gprefix=sprefix
    
    NewGrids=False
    
    if NewGrids:
        print("Generating new grids, set NewGrids=False to save time later")
        grids=misc_functions.initialise_grids(
            surveys,zDMgrid, zvals, dmvals, state, wdist=True)#, source_evolution=source_evolution, alpha_method=alpha_method)
        with open('Pickle/'+gprefix+'grids.pkl', 'wb') as output:
            pickle.dump(grids, output, pickle.HIGHEST_PROTOCOL)
    else:
        print("Loading grid ",'Pickle/'+gprefix+'grids.pkl')
        with open('Pickle/'+gprefix+'grids.pkl', 'rb') as infile:
            grids=pickle.load(infile)
    
    
    muDM=10**state.host.lmean
    plot_grid=True
    if plot_grid:
        misc_functions.plot_grid_2(grids[0].rates,grids[0].zvals,grids[0].dmvals,FRBZ=craco.frbs["Z"],FRBDM=craco.DMEGs,zmax=1.8,DMmax=2000,name='CRACOalpha/alpha1_mc_frbs_best_zdm_grid.pdf',norm=2,log=True,label='$\\log_{10} p({\\rm DM}_{\\rm EG},z)$',project=False,Aconts=[0.01,0.1,0.5],Macquart=muDM)
    
    #testing_MC!!! Generate pseudo samples from lat50
    which=0
    g=grids[which]
    s=surveys[which]
    
    savefile='CRACOalpha/alpha_mc_sample.npy'
    
    try:
        sample=np.load(savefile)
        print("Loading ",sample.shape[0]," samples from file ",savefile)
    except:
        Nsamples=10000
        print("Generating ",Nsamples," samples from survey/grid ",which)
        sample=g.GenMCSample(Nsamples)
        sample=np.array(sample)
        np.save(savefile,sample)
    
    
    #sample
    # 0: z
    # 1: DM
    # 2: b
    # 3: w
    # 4: s
    # plot some sample plots
    do_basic_sample_plots(sample,opdir='CRACOalpha')
    NFRB=5000
    for i in np.arange(NFRB):
        DMG=35
        DMEG=sample[i,1]
        DMtot=DMEG+DMG+state.MW.DMhalo
        SNRTHRESH=9.5
        SNR=SNRTHRESH*sample[i,4]
        z=sample[i,0]
        w=sample[i,3]
        
        string="FRB "+str(i)+'  {:6.1f}  35   {:6.1f}  {:5.3f}   {:5.1f}  {:5.1f}'.format(DMtot,DMEG,z,SNR,w)
        #print("FRB ",i,DMtot,SNR,DMEG,w)
        print (string)
    exit()
    #evaluate_mc_sample_v1(g,s,pset,sample)
    evaluate_mc_sample_v2(g,s,pset,sample)
    
    
def evaluate_mc_sample_v1(grid,survey,pset,sample,opdir='Plots'):
    """
    Evaluates the likelihoods for an MC sample of events
    Simply replaces individual sets of z, DM, s with MC sets
    Will produce a plot of Nsamples/NFRB pseudo datasets.
    """
    t0=time.process_time()
    
    nsamples=sample.shape[0]
    
    # get number of FRBs per sample
    Npersurvey=survey.NFRB
    # determines how many false surveys we have stats for
    Nsurveys=int(nsamples/Npersurvey)
    
    print("We can evaluate ",Nsurveys,"MC surveys given a total of ",nsamples," and ",Npersurvey," FRBs in the original data")
    
    # makes a deep copy of the survey
    s=copy.deepcopy(survey)
    
    lls=[]
    #Data order is DM,z,b,w,s
    # we loop through, artificially altering the survey with the composite values.
    for i in np.arange(Nsurveys):
        this_sample=sample[i*Npersurvey:(i+1)*Npersurvey,:]
        s.DMEGs=this_sample[:,0]
        s.Ss=this_sample[:,4]
        if s.nD==1: # DM, snr only
            ll=it.calc_likelihoods_1D(grid,s,pset,psnr=True,Pn=True,dolist=0)
        else:
            s.Zs=this_sample[:,1]
            ll=it.calc_likelihoods_2D(grid,s,pset,psnr=True,Pn=True,dolist=0)
        lls.append(ll)
    t1=time.process_time()
    dt=t1-t0
    print("Finished after ",dt," seconds")
    
    lls=np.array(lls)
    
    plt.figure()
    plt.hist(lls,bins=20)
    plt.xlabel('log likelihoods [log10]')
    plt.ylabel('p(ll)')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(opdir+'/ll_histogram.pdf')
    plt.close()


def evaluate_mc_sample_v2(grid,survey,pset,sample,opdir='Plots',Nsubsamp=1000):
    """
    Evaluates the likelihoods for an MC sample of events
    First, gets likelihoods for entire set of FRBs
    Then re-samples as needed, a total of Nsubsamp times
    """
    t0=time.process_time()
    
    nsamples=sample.shape[0]
    
    # makes a deep copy of the survey
    s=copy.deepcopy(survey)
    NFRBs=s.NFRB
    
    s.NFRB=nsamples # NOTE: does NOT change the assumed normalised FRB total!
    s.DMEGs=sample[:,1]
    s.Ss=sample[:,4]
    if s.nD==1: # DM, snr only
        llsum,lllist,expected,longlist=it.calc_likelihoods_1D(grid,s,pset,psnr=True,Pn=True,dolist=2)
    else:
        s.Zs=sample[:,0]
        llsum,lllist,expected,longlist=it.calc_likelihoods_2D(grid,s,pset,psnr=True,Pn=True,dolist=2)
    
    # we should preserve the normalisation factor for Tobs from lllist
    Pzdm,Pn,Psnr=lllist
    
    # plots histogram of individual FRB likelihoods including Psnr and Pzdm
    plt.figure()
    plt.hist(longlist,bins=100)
    plt.xlabel('Individual Psnr,Pzdm log likelihoods [log10]')
    plt.ylabel('p(ll)')
    plt.tight_layout()
    plt.savefig(opdir+'/individual_ll_histogram.pdf')
    plt.close()
    
    # generates many sub-samples of the data
    lltots=[]
    for i in np.arange(Nsubsamp):
        thislist=np.random.choice(longlist,NFRBs) # samples with replacement, by default
        lltot=Pn+np.sum(thislist)
        lltots.append(lltot)
    
    plt.figure()
    plt.hist(lltots,bins=20)
    plt.xlabel('log likelihoods [log10]')
    plt.ylabel('p(ll)')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(opdir+'/sampled_ll_histogram.pdf')
    plt.close()
    
    t1=time.process_time()
    dt=t1-t0
    print("Finished after ",dt," seconds")
    
    
def do_basic_sample_plots(sample,opdir='Plots'):
    """
    Data order is DM,z,b,w,s
    
    """
    if not os.path.exists(opdir):
        os.mkdir(opdir)
    zs=sample[:,0]
    DMs=sample[:,1]
    plt.figure()
    plt.hist(DMs,bins=100)
    plt.xlabel('DM')
    plt.ylabel('Sampled DMs')
    plt.tight_layout()
    plt.savefig(opdir+'/DM_histogram.pdf')
    plt.close()
    
    plt.figure()
    plt.hist(zs,bins=100)
    plt.xlabel('z')
    plt.ylabel('Sampled redshifts')
    plt.tight_layout()
    plt.savefig(opdir+'/z_histogram.pdf')
    plt.close()
    
    bs=sample[:,2]
    plt.figure()
    plt.hist(np.log10(bs),bins=5)
    plt.xlabel('log10 beam value')
    plt.yscale('log')
    plt.ylabel('Sampled beam bin')
    plt.tight_layout()
    plt.savefig(opdir+'/b_histogram.pdf')
    plt.close()
    
    ws=sample[:,3]
    plt.figure()
    plt.hist(ws,bins=5)
    plt.xlabel('width bin (not actual width!)')
    plt.ylabel('Sampled width bin')
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig(opdir+'/w_histogram.pdf')
    plt.close()
    
    s=sample[:,4]
    plt.figure()
    plt.hist(np.log10(s),bins=100)
    plt.xlabel('$\\log_{10} (s={\\rm SNR}/{\\rm SNR}_{\\rm th})$')
    plt.yscale('log')
    plt.ylabel('Sampled $s$')
    plt.tight_layout()
    plt.savefig(opdir+'/s_histogram.pdf')
    plt.close()
    
main()
