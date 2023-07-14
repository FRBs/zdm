""" 
This script iterates over a grid in Rgamma and Rmax. For each, it:
- calculates Rstar, such that all FRBs repeating at Rstar (=Rmin=Rmax)
    reproduce the correct repetition rate (Rgamma not important)
- iterates over Rgamma, Rmax, to find the Rmin reproducing the correct rate
- simulates the z/DM and declination distribution of repeating CHIME FRBs,
    and calculates likelihood metrics
- saves the resulting output


By default it assumes that 100% of CHIME bursts come from repeaters.
However, you can set FC to vary this "Fraction" to be less than unity.

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

def main(FC=1.0):
    
    ##### defines parameter range of search
    Rmax0 = -2
    Rmax1 = 3
    nRmax = 17 # should have been 16 - oops! :-)
    Rmaxes = np.logspace(Rmax0,Rmax1,nRmax)
    
    Rgamma0 = -1.2
    Rgamma1 = -3
    nRgamma = 19
    Rgammas = np.linspace(Rgamma0,Rgamma1,nRgamma)+0.001 # avoids magic number of -2
    
    Rmin0 = -5
    Rmin1 = 0.
    nRmin = 21
    Rmins = np.logspace(Rmin0,Rmin1,nRmin)
    
    
    # gets the possible states for evaluation
    states,names=st.get_states()
    
    sdir = os.path.join(resource_filename('zdm','../'),'papers/Repeaters/Surveys')
    
    outdir = 'Rfitting39_'+str(FC)+'/'
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    for i,state in enumerate(states):
        # originally we trialled several states, before realising this was nonsense
        outfile = outdir+'FC39'+str(FC)+'converge_set_'+str(i)+'_output.npz'
        oldoutfile=None
        converge_state(state,Rmins,Rmaxes,Rgammas,outfile,oldoutfile,FC=FC)
        break

def converge_state(state,Rmins,Rmaxes,Rgammas,outfile,oldoutfile=None,Nbin=6,verbose=False,Rstar = 0.3,\
    Nacc = 0.1,load=False,FC=1.0):
    """
    Find the best-fitting parameters in an intelligent way
    
    Steps:
    #1: Find R*, the critical rate producing repeat distributions
        Rstar begins at 0.3 as a guess, but should be set from previous state as a best guess
        
        Nacc is the accuracy in Nrepeaters predicted. About 10% of 1 sigma, i.e. 0.1 * sqrt(16)
        
    
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
    print("We have calculated a total of ",NR,NS," repeat/single bursts")
    
    if verbose:
        
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
    OK = np.where(Rmaxes >= Rstar)[0]
    Rmaxes = Rmaxes[OK]
    if load:
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
    else:
        # holds best-fit value of Rgamma
        Rmins = np.zeros([Rgammas.size,Rmaxes.size])
        
        lps = np.zeros([Rgammas.size,Rmaxes.size])
        lns = np.zeros([Rgammas.size,Rmaxes.size])
        ldms = np.zeros([Rgammas.size,Rmaxes.size])
        lpNs = np.zeros([Rgammas.size,Rmaxes.size])
        Nrs = np.zeros([Rgammas.size,Rmaxes.size])
        
        lskps = np.zeros([Rgammas.size,Rmaxes.size])
        lrkps = np.zeros([Rgammas.size,Rmaxes.size])
        ltdm_kss = np.zeros([Rgammas.size,Rmaxes.size])
        ltdm_krs = np.zeros([Rgammas.size,Rmaxes.size])
        lprod_bin_dm_krs = np.zeros([Rgammas.size,Rmaxes.size])
        lprod_bin_dm_kss = np.zeros([Rgammas.size,Rmaxes.size])
        Cprimes = np.zeros([Rgammas.size,Rmaxes.size])
        Rstar = find_Rstar(Rstar,Nacc,ss,gs,NR,NS,nrs,nss,irs,iss,\
            rgs,sdmegs,rdmegs,sdecs,rdecs,bounds,verbose=False,FC=FC)
    
    # Here, Rmin *and* Rmax are both increasing. Which means we begin
    # very heavily weighted to Rmin. Thus we begin with a very flat Rgamma.
    # We also set up initial increments. The direction begins reducing Rmin
    # due to Rmax increasing.
    Rmin = Rstar
    # ddRmin is how much dRmin changes each iteration, i.e. dRmin ** ddRmin
    ddRmin = 0.3
    
    t0=time.time()
    for i,Rgamma in enumerate(Rgammas):
        
        Rmin = Rstar # reset Rmin to Rstar when Rmax goes back to lowest value
        
        for j,Rmax in enumerate(Rmaxes):
            # init some things
            count = 0
            GiveUp = False # if set to true, cancels all further Rmax integration
            verbose = False
            
            
            # begin by decreasing Rmin
            last = -1
            dRmin = 3
            #Rmin /= dRmin # reduce it; after all, Rmax has increased!
            lastNr = -1 # impossible number by design
            
            # we find that if Rmin goes too low, then there is no point
            # in further calculations. Partly this may be because the sum
            # never converges, or because it is converging extremely slowly.
            # Practically, it does not matter,
            # we don't need this, if Rmin is
            if Rmin < 1e-9:
                GiveUp = True
                #Rmin = 1.e-8 # Evaluate at 1e-8. Is this correct? Probably - once we hit this,
                # keep Rmin constant
                code=5
            else:
                Rmin /= dRmin
            
            
            print("Iterating for ",Rgamma,Rmax," Rmin begins at ",Rmin)
            # we now optimise Rgamma. Note that we are increasing Rmax for constant Rgamma
            # which means that Rmin *must* be decreasing each time
            while True:
                rgs,lp,ln,ldm,lpN,Nr,lskp,lrkp,ltdm_ks,ltdm_kr,lprod_bin_dm_kr,\
                    lprod_bin_dm_ks,Cprime=calc_repetition_statistics(ss,gs,Rmin,Rmax,Rgamma,\
                    NR,NS,nrs,nss,irs,iss,rgs,sdmegs,rdmegs,sdecs,rdecs,bounds,\
                    verbose=False,FC=FC)
                
                # searches for potential problems with convergence
                # Errors if predicting too many repeaters, convergence too slow,
                # or Nreps going in the wrong direction
                if np.abs(Nr - lastNr) < 0.002:
                    GiveUp=True
                    code=1
                elif Nr > 1e4:
                    GiveUp=True
                    code=2
                elif Rmin < 1e-9:
                    GiveUp = True
                    code=3
                if GiveUp:
                    #give up and break
                    print("Giving up...",count,Nr,lastNr,code)
                    GiveUp = True
                    # this also means we skip all future Rmax increases
                    break
                
                if np.abs(Nr - NR) < Nacc:
                    # we have a sufficiently accurate Rmin for this Rmax/Rgamma
                    if verbose:
                        print("For Rmin/max ",Rgamma,Rmax,", found Rmin ",Rmin," got ",Nr," needing ",NR)
                    break
                elif Nr > NR:
                    # reduce Rmin, too many repeaters
                    if last == 1:
                        last = -1 # we are now reducing Rmin
                        dRmin = dRmin ** ddRmin # smaller increments
                        Rmin /= dRmin
                    else:
                        Rmin /= dRmin # was already reducing, keep it going
                        last = -1 # might have been zero in theory
                elif Nr < NR:
                    # increase Rstar, too few repeaters
                    if last == -1:
                        last = 1 # now increasing Rmin
                        dRmin = dRmin ** ddRmin #smller increments
                        Rmin *= dRmin # was already reducing, keep it going
                    else:
                        # was already increasing or neutral, keep going
                        Rmin *= dRmin
                        last = 1
                else:
                    print("Why did we do nothing to Rmin?")
                
                
                count += 1
                lastNr = Nr
                
            print("After ",count," iterations, we find Rmin = ",Rmin)
            
            t1=time.time()
            Rmins[i,j] = Rmin
            lps[i,j] = lp
            lns[i,j] = ln
            ldms[i,j] = ldm
            lpNs[i,j] = lpN
            Nrs[i,j] = Nr
            lskps[i,j] = lskp
            lrkps[i,j] = lrkp
            ltdm_kss[i,j] = ltdm_ks
            ltdm_krs[i,j] = ltdm_kr
            lprod_bin_dm_krs[i,j] = lprod_bin_dm_kr
            lprod_bin_dm_kss[i,j] = lprod_bin_dm_ks
            Cprimes[i,j] = Cprime
            print("Iteration ",i,j," time taken is ",t1-t0)
    
    np.savez(outfile,lps,lns,ldms,lpNs,Nrs,Rmins,Rmaxes,Rgammas,lskps,lrkps,ltdm_kss,\
                    ltdm_krs,lprod_bin_dm_krs,lprod_bin_dm_kss,Rstar,Cprimes)
       

def find_Rstar(Rstar,Nacc,ss,gs,NR,NS,nrs,nss,irs,iss,rgs,\
        sdmegs,rdmegs,sdecs,rdecs,bounds,verbose = False,FC=1.0):
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
        rgs,lp,ln,ldm,lpN,Nr,lskp,lrkp,ltdm_ks,ltdm_kr,lprod_bin_dm_kr,\
            lprod_bin_dm_ks=calc_repetition_statistics(ss,gs,Rmin,Rmax,Rgamma,\
                NR,NS,nrs,nss,irs,iss,rgs,sdmegs,rdmegs,sdecs,rdecs,bounds,\
                verbose=False,FC=FC)
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

def loop_state(state,Rmins,Rmaxes,Rgammas,outfile,oldoutfile=None,Nbin=6,verbose=False):
    """
    Defined to test some corrections to the repeating FRBs method
    """
    
    if oldoutfile is not None:
        data=np.load(oldoutfile)
        
        oldlps=data['arr_0']
        oldlns=data['arr_1']
        oldldms=data['arr_2']
        oldRmins=data['arr_3']
        oldRmaxes=data['arr_4']
        oldRgammas=data['arr_5']
    
    
    
    # old implementation
    # defines list of surveys to consider, together with Tpoint
    sdir = os.path.join(resource_filename('zdm','../'),'papers/Repeaters/Surveys')
    
    
    bdir='Nbounds'+str(Nbin)+'/'
    beams.beams_path = os.path.join(resource_filename('zdm','../'),'papers/Repeaters/'+bdir)
    bounds = np.load(bdir+'bounds.npy')
    solids = np.load(bdir+'solids.npy')
    #bounds=np.array([-11,30,60,70,80,85,90])
    
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
    
    count=0
    lps = np.zeros([Rmins.size,Rmaxes.size,Rgammas.size])
    lns = np.zeros([Rmins.size,Rmaxes.size,Rgammas.size])
    ldms = np.zeros([Rmins.size,Rmaxes.size,Rgammas.size])
    lpNs = np.zeros([Rmins.size,Rmaxes.size,Rgammas.size])
    Nrs = np.zeros([Rmins.size,Rmaxes.size,Rgammas.size])
    
    lskps = np.zeros([Rmins.size,Rmaxes.size,Rgammas.size])
    lrkps = np.zeros([Rmins.size,Rmaxes.size,Rgammas.size])
    ltdm_kss = np.zeros([Rmins.size,Rmaxes.size,Rgammas.size])
    ltdm_krs = np.zeros([Rmins.size,Rmaxes.size,Rgammas.size])
    lprod_bin_dm_krs = np.zeros([Rmins.size,Rmaxes.size,Rgammas.size])
    lprod_bin_dm_kss = np.zeros([Rmins.size,Rmaxes.size,Rgammas.size])
    
    # gets CHIME declination histograms
    Csxvals,Csyvals,Crxvals,Cryvals = ute.get_chime_rs_dec_histograms(newdata=False)
    sdmegs,rdmegs,sdecs,rdecs = ute.get_chime_dec_dm_data(newdata=False,sort=True)
    
    
    rgs=[None]*Nbin # 6 is number of dec bins
    t0=time.time()
    for k,Rgamma in enumerate(Rgammas):
        for i,Rmin in enumerate(Rmins):
            for j,Rmax in enumerate(Rmaxes):
                
                if oldoutfile is not None and Rgamma in oldRgammas \
                    and Rmin in oldRmins and Rmax in oldRmaxs:
                    print("Found!")
                    #fill with old values
                    oldk = np.where(oldRgammas == Rgamma)[0]
                    oldi = np.where(oldRmins == Rmin)[0]
                    oldj = np.where(oldRmaxs == Rmax)[0]
                    
                    lps[i,j,k] = oldlps[oldi,oldj,oldk]
                    lns[i,j,k] = oldlns[oldi,oldj,oldk]
                    ldms[i,j,k] = oldldms[oldi,oldj,oldk]
                else:
                    if Rmin == Rmax:
                        Rmin *= 0.99
                        
                        rgs,lp,ln,ldm,lpN,Nr,lskp,lrkp,ltdm_ks,ltdm_kr,lprod_bin_dm_kr,\
                            lprod_bin_dm_ks,Cprime=calc_repetition_statistics(ss,gs,Rmin,Rmax,Rgamma,\
                            NR,NS,nrs,nss,irs,iss,rgs,sdmegs,rdmegs,sdecs,rdecs,bounds,\
                            verbose=False)
                    elif Rmin > Rmax:
                        lp=-99
                        ln=-99
                        ldm=-99
                        lskp = -99
                        lrkp = -99
                    else:
                        rgs,lp,ln,ldm,lpN,Nr,lskp,lrkp,ltdm_ks,ltdm_kr,lprod_bin_dm_kr,\
                            lprod_bin_dm_ks,Cprime=calc_repetition_statistics(ss,gs,Rmin,Rmax,Rgamma,\
                            NR,NS,nrs,nss,irs,iss,rgs,sdmegs,rdmegs,sdecs,rdecs,bounds,\
                            verbose=False)
                    t1=time.time()
                    lps[i,j,k] = lp
                    lns[i,j,k] = ln
                    ldms[i,j,k] = ldm
                    lpNs[i,j,k] = lpN
                    Nrs[i,j,k] = Nr
                    lskps[i,j,k] = lskp
                    lrkps[i,j,k] = lrkp
                    ltdm_kss[i,j,k] = ltdm_ks
                    ltdm_krs[i,j,k] = ltdm_kr
                    lprod_bin_dm_krs[i,j,k] = lprod_bin_dm_kr
                    lprod_bin_dm_kss[i,j,k] = lprod_bin_dm_ks
                    print("Iteration ",i,j,k," time taken is ",t1-t0)
    
    np.savez(outfile,lps,lns,ldms,lpNs,Nrs,Rmins,Rmaxes,Rgammas,lskps,lrkps,ltdm_kss,\
                    ltdm_krs,lprod_bin_dm_krs,lprod_bin_dm_kss)
       
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
        
        # get total progenitor number
        Cprime = rg.NRtot
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
    
    ############# ks tests on declination ##############
    # constructs cumulative as function of dec
    RModel_decyhist = np.zeros([Nbin+1])
    Model_decxhist = bounds
    RModel_decyhist[1:] = np.cumsum(nrs)
    RModel_decyhist /= RModel_decyhist[-1]
    
    Rdec_kstat=sp.stats.ks_1samp(Crdecs,ute.nu_cdf,args=(Model_decxhist,RModel_decyhist),\
        alternative='two-sided',mode='exact')
    rkp = Rdec_kstat[1]
    lrkp = np.log10(rkp)
    
    # constructs cumulative as function of dec
    SModel_decyhist = np.zeros([Nbin+1])
    SModel_decyhist[1:] = np.cumsum(nss)
    SModel_decyhist /= SModel_decyhist[-1]
    
    Sdec_kstat=sp.stats.ks_1samp(Csdecs,ute.nu_cdf,args=(Model_decxhist,SModel_decyhist),\
        alternative='two-sided',mode='exact')
    skp = Sdec_kstat[1]
    lskp = np.log10(skp)
    
    ############ ks test on dm #############
    # thankfully, dm is a uniform grid, can go faster
    
    # constructs cumulative as function of dec: repeaters
    cs_tdmr = np.cumsum(dmr)
    cs_tdmr /= cs_tdmr[-1]
    result = sp.stats.ks_1samp(Crdmegs,ute.cdf,args=(g.dmvals,cs_tdmr),\
        alternative='two-sided',mode='exact')
    tdm_kr = result[1]
    ltdm_kr = np.log10(tdm_kr)
    
    cs_tdms = np.cumsum(dmr)
    cs_tdms /= cs_tdms[-1]
    result = sp.stats.ks_1samp(Csdmegs,ute.cdf,args=(g.dmvals,cs_tdms),\
        alternative='two-sided',mode='exact')
    tdm_ks = result[1]
    ltdm_ks = np.log10(tdm_ks)
    
    if verbose:
        print("ks tests on entire (s,r) DM distribution give ",tdm_ks, tdm_kr)
    
    ############### normalisation ###################
    
    # calculates dechist at histogram points
        
    # fits number of single bursts (total only)
    # FC*CNS is the CHIME single bursts explained by repeaters.
    # ts is the total singles predicted by the model
    norm = FC*CNS/ts
    
    # expected total number of repeaters, after scaling
    scaled_tr = tr*norm
    
    
    ############# Poisson probability ############
    PN2 = np.exp(-scaled_tr)*scaled_tr**CNR \
            / np.math.factorial(CNR)
    PN = poisson.pmf(CNR,scaled_tr)
    
    lPN = np.log10(PN)
    #print("Total repaters: Expected ",scaled_tr," observed ",CNR, "poisson prob ",PN)
    
    
    ############### calculate bin-by-bin probabilities ##############
    # get Poissonian repeater likelihood (function of declination, and total)
    
    # calc total poisson here
    Pns = np.zeros([Nbin])
    Pdms = np.zeros([Nbin])
    bin_dm_ks = np.zeros([Nbin]) # holding ks stats for singles
    bin_dm_kr = np.zeros([Nbin]) # holding ks stats for repeaters
    
    all_bin_lpdm = 1. # log probability
    all_bin_lpn = 1.
    for ibin in np.arange(Nbin):
        rg = rgs[ibin] # get back repeat grid
        
        # calculate probability of seeing this many repeaters in this interval
        scaled_nr = nrs[ibin]*norm
        #Pn=poisson.pmf(scaled_nr,Cnrs[ibin])
        
        # Poisson probability
        
        Pn = np.exp(-scaled_nr)*scaled_nr**Cnrs[ibin] \
            / np.math.factorial(Cnrs[ibin])
        
        Pns[ibin]=Pn
        if verbose:
            print("For bin ",ibin," expected ",scaled_nr," obs ",Cnrs[ibin],"gives prob ",Pn)
        #calculate likelihood for each repeater. First normalise.
        # dmrs holds lists of p(DM) for repeaters for each bin
        # nrs holds number of repeating frbs for that bin
        scaled_dmr = dmrs[ibin] / nrs[ibin]
        scaled_dms = dmss[ibin] / nss[ibin]
        
        # loop over the dm values
        bin_pdm=1.
        for ifrb in irs[ibin]:
            dm = ss[ibin].DMEGs[ifrb]
            idm1 = int(dm/ddm)
            idm2 = idm1+1
            kdm2 = dm/ddm - idm1
            kdm1 = 1.-kdm2
            
            pdm1 = scaled_dmr[idm1]
            pdm2 = scaled_dmr[idm2]
            
            pdm = pdm1*kdm1 + pdm2*kdm2
            if verbose:
                print("bin ",ibin," frb ",ifrb," prob ",pdm)
            bin_pdm *= pdm
        if verbose:
            print("Total pdm is ",bin_pdm)
        all_bin_lpdm += np.log10(bin_pdm)
        all_bin_lpn += np.log10(Pn)
        
        ####### calculates ks stat for this bin over dm #########
        
        # only proceed with ks if there are FRBs to consider!
        if len(irs[ibin])==0:
            bin_dm_kr[ibin] = 1.
            bin_dm_ks[ibin] = 1.
            continue
        
        # constructs cumulative as function of dec: repeaters
        cs_scaled_dmr = np.cumsum(scaled_dmr)
        cs_scaled_dmr /= cs_scaled_dmr[-1]
        use_these_dms = ss[ibin].DMEGs[irs[ibin]]
        result = sp.stats.ks_1samp(use_these_dms,ute.cdf,args=(g.dmvals,cs_scaled_dmr),\
            alternative='two-sided',mode='exact')
        bin_dm_kr[ibin] = result[1]
        
        # constructs cumulative as function of dec: singles
        cs_scaled_dms = np.cumsum(scaled_dms)
        cs_scaled_dms /= cs_scaled_dms[-1]
        use_these_dms = ss[ibin].DMEGs[iss[ibin]]
        result = sp.stats.ks_1samp(use_these_dms,ute.cdf,args=(g.dmvals,cs_scaled_dms),\
            alternative='two-sided',mode='exact')
        bin_dm_ks[ibin] = result[1]
        # now calculate ks tests bin-by-bin
    if verbose:
        print("Calculated these ks stats for repeaters ",bin_dm_kr)
        print("Calculated these ks stats for singles ",bin_dm_ks)
    lprod_bin_dm_kr = np.sum(np.log10(bin_dm_kr))
    lprod_bin_dm_ks = np.sum(np.log10(bin_dm_ks))
    
    ############### final summation of probabilities ###########
    
    all_bin_lp = all_bin_lpdm + lPN + lrkp +lskp #all_bin_lpn do NOT to bin-by-bin numbers. Dumb idiot.
    if verbose:
        print("Final log-probs are ",all_bin_lp,all_bin_lpn,all_bin_lpdm,CNR,scaled_tr,PN,PN2)
    
    #We here return the probabilities for:
    #- everything (pDM), p(N), p(dec)
    #- pDM
    #- pN
    #- p(dec) [repeaters]
    #- pdec [singles]
    #- ADD: ks-stat for singles (or other similar, could be log-likelihood?)
    
    # rgs: repeat grids, to be re-used
    # all_bin_lp: log probability, best guess of some combination
    # all_bin_lpn: summed log10 Poisson probability of number of reps in each bin
    # all_bin_lpdm: summed log10 p(DM) summed over each bin
    # lPN: log10 Poisson probability of summed reps over all bins
    # scaled_tr: expected number of repeaters, after scaling
    # lskp, lrkp: log10 ks test results for singles and repeater distributions over declination
    # ltdm_ks,ltdm_kr ks test values over entire DM range
    # lprod_bin_dm_kr,lprod_bin_dm_ks ks test values over individual bins, product thereof
    
    # skp, rkp: ks stats for the declination distribution
    
    return rgs,all_bin_lp,all_bin_lpn,all_bin_lpdm,lPN,scaled_tr,lskp,lrkp,\
        ltdm_ks,ltdm_kr,lprod_bin_dm_kr,lprod_bin_dm_ks,Cprime


main()
# uncomment the following to generate this for all F
for FC in [1.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    main(FC=FC)
