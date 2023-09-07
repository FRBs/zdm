""" 
This script finds Rstar as a function of f, the fraction of single bursts that
come from repeaters

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
    
    ##### defines parameter range of search
    Rmax0 = -1
    Rmax1 = 3
    nRmax = 17
    #nRmax = 7
    Rmaxes = np.logspace(Rmax0,Rmax1,nRmax)
    
    Rgamma0 = -1.2
    Rgamma1 = -3
    #nRgamma = 37
    nRgamma = 19
    Rgammas = np.linspace(Rgamma0,Rgamma1,nRgamma)+0.001 # avoids magic number of -2
    
    Rmin0 = -5
    Rmin1 = 0.
    nRmin = 21
    #nRmin = 11
    Rmins = np.logspace(Rmin0,Rmin1,nRmin)
    
    
    fgrid = np.linspace(0.1,1.0,10)
    
    # gets the possible states for evaluation
    states,names=st.get_states()
    
    outdir='Rstar/'
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    
    # For each possible state, calculates Rstar
    sdir = os.path.join(resource_filename('zdm','data/Surveys/'),'CHIME/')
    for i,state in enumerate(states):
        outfile = outdir+'rstar_set_'+str(i)+'_'+'_output.npz'
        oldoutfile=None
        converge_state(state,Rmins,Rmaxes,Rgammas,outfile,fgrid,oldoutfile)
        exit()

def converge_state(state,Rmins,Rmaxes,Rgammas,outfile,fgrid,oldoutfile=None,Nbin=6,verbose=False,Rstar = 0.3,\
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
    sdir = os.path.join(resource_filename('zdm','data/Surveys/'),'CHIME/')
    
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
    
    # initial rate change in dRstar
    
    # the below only needs to be accurate compared to the grid in Rmax
    Rstar=1.
    Rstars = np.zeros([fgrid.size])
    for i,f in enumerate(fgrid):
        Rstar = find_Rstar(Rstar,Nacc,ss,gs,NR,NS,nrs,nss,irs,iss,\
            rgs,sdmegs,rdmegs,sdecs,rdecs,bounds,f=f,verbose=False)
        Rstars[i] = Rstar
        print("For f ",f," found Rstar ",Rstar)
    np.savez(outfile,fgrid,Rstars)
       

def find_Rstar(Rstar,Nacc,ss,gs,NR,NS,nrs,nss,irs,iss,rgs,\
        sdmegs,rdmegs,sdecs,rdecs,bounds,f=1.,verbose = False):
    """
    Finds critical value of repeater rate R
    """
    verbose = True
    ddRstar = 0.3
    dRstar = 3.
    last = 0 # will be -1 for negative, +1 for positive
    count=0
    #Rstar = 1.
    Rgamma=-1.5
    while True:
        Rmin = Rstar * 0.99
        Rmax = Rstar * 1.01
        Nr=calc_repetition_statistics(ss,gs,Rmin,Rmax,Rgamma,\
                NR,NS,nrs,nss,irs,iss,rgs,sdmegs,rdmegs,sdecs,rdecs,bounds,\
                FC=f,verbose=False)
        if np.abs(Nr - NR) < Nacc:
            # we have a sufficiently accurate Rstar
            if verbose:
                print("Found Rstar of ",Rstar," got ",Nr," needing ",NR)
            break
        elif Nr > NR:
            # reduce Rstar, too many repeaters
            if last == 1:
                last = -1 # we are now reducing Rstar
                dRstar = dRstar ** ddRstar # smaller increments
                Rstar /= dRstar
            else:
                Rstar /= dRstar # was already reducing, keep it going
                last = -1
        elif Nr < NR:
            # increase Rstar, too few repeaters
            if last == -1:
                last = 1 # now increasing Rstar
                dRstar = dRstar ** ddRstar #smller increments
                Rstar *= dRstar # was already reducing, keep it going
            else:
                # was already increasing or neutral, keep going
                Rstar *= dRstar
                last = 1
        else:
            print("Why did we do nothing while finding Rstar?")
        if verbose:
            print(count, "Rstar ",Rstar,Nr,NR)
        count += 1
        
    if verbose:
        print("After ",count, "repetitions, found Rstar ",Rstar," predicting ",Nr, " repeaters, c.f. CHIME: ",NR)
    return Rstar

       
    ####### now inside repeater loop ###### - new function to loop over rep properties
def calc_repetition_statistics(ss,gs,Rmin,Rmax,Rgamma,CNR,CNS,Cnrs,Cnss,irs,iss,rgs,\
        Csdmegs,Crdmegs,Csdecs,Crdecs,bounds,FC=1.,verbose=False):
    """
    
    FC is fraction of CHIME singles explained by repeaters. <= 1.0.
    Don't set this to less than 1., we can't reproduce the repeaters we have!
    """
    Nbin=6 # should make this global... or via an input file!
    
    # sum totals over declination
    tr=0
    ts=0
    ndm = gs[0].dmvals.size
    ddm = gs[0].dmvals[1]-gs[0].dmvals[0]
    # contains totals of z-distributions over all decs
    tdmr = np.zeros([ndm])
    tdms = np.zeros([ndm])
    dmrs=[]
    dmss=[]
    nrs=[]
    nss=[]
    
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
            rg = rep.repeat_Grid(g,Tfield=s.TOBS,Nfields=1,MC=False,opdir=None,bmethod=2)
            rgs[ibin]=rg
            # I have turned off saving, it produces 1GB of output per loop!
            #with open(savefile, 'wb') as output:
            #    pickle.dump(rg, output, pickle.HIGHEST_PROTOCOL)
        
        
        # collapses CHIME dm distribution for repeaters and once-off burts
        dmr = np.sum(rg.exact_reps,axis=0)
        dms = np.sum(rg.exact_singles,axis=0)
        nr = np.sum(dmr)
        ns = np.sum(dms)
        
        # adds to running totals
        tdmr += dmr
        tdms += dms
        tr += nr
        ts += ns
        
        dmrs.append(dmr)
        dmss.append(dms)
        nrs.append(nr)
        nss.append(ns)
        rgs.append(rg)
    
    nrs = np.array(nrs)
    
    ############### normalisation ###################
    
    # calculates dechist at histogram points
        
    # fits number of single bursts (total only)
    norm = FC*CNS/ts
    
    # expected total number of repeaters, after scaling
    scaled_tr = tr*norm
    
    
    return scaled_tr




main()
