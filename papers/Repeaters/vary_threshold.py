""" 
This script uses the Shin et al best-fits, and
iterates over survey threshold to compare
DM distributions, and number of FRBs

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

import numpy as np
from zdm import survey
from matplotlib import pyplot as plt

import utilities as ute

import states as st
import scipy as sp

import matplotlib
import time
from zdm import beams
beams.beams_path = 'BeamData/'
    

matplotlib.rcParams['image.interpolation'] = None

defaultsize=14
ds=4
font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : defaultsize}
matplotlib.rc('font', **font)

def main():
    
    #defines lists of repeater properties, in order Rmin,Rmax,r
    # units of Rmin and Rmax are "per day above 10^39 erg"
    Rset1={"Rmin":1e-4,"Rmax":50,"Rgamma":-2.2}
    #sets=[Rset1,Rset2]
    
    
    ######## We iterate through the different parameter sets, using the CHIME response function #######
    # standard analysis
    extreme_rsets(Rset1,opdir='Threshold/',DMcut=None,chime_response=True,allplot=True,load=True)
    


def MyExp(x, m, t):
    return m * np.exp(-x/t)

        
def extreme_rsets(Rset,opdir='Threshold/',DMcut=None,chime_response=True,allplot=True,load=False):
    """
    This function iterates through 14 parameter sets:
        6 params by min/max in each from James et al
        Shin et al best fit
        James et al best fit
    And compares the predicted number of single FRBs, and their cumulative DM distribution,
    with that predicted from two sets of repeater parameters (Rset1 and Rset2).
    
    It prints normalisation factors in terms of total number of FRBs between the two cases.
    It also calculates values of the ks-statistics for each case
    """
    # old implementation
    # defines list of surveys to consider, together with Tpoint
    sdir = os.path.join(resource_filename('zdm','data/Surveys/'),'CHIME/')
    
    if not os.path.exists(opdir):
        os.mkdir(opdir)
    
    Nbin=6
    surveys=[]
    grids=[]
    
    label="Shin"
    pset=st.shin_fit()
    
    #### for normalised plots ####
    plt.figure()
    ax1=plt.gca()
    plt.xlabel('DM')
    plt.ylabel('p(DM)')
    
    #### without normalisation ####
    plt.figure()
    ax2=plt.gca()
    plt.xlabel('DM')
    plt.ylabel('p(DM)')
    
    
    # iterates over assumed threshold
    thresholds = np.linspace(0.5,8,16) #Jy ms
    kstats = np.zeros([thresholds.size])
    norms = np.zeros([thresholds.size])
    
    for i,thresh in enumerate(thresholds):
        
        savefile=opdir+'saved_data_'+str(i)+'.npz'
        if load:
            data=np.load(savefile)
            dm=data['dm']
            s1=data['s1']
            r1=data['r1']
            b1=data['b1']
            tabdm=data['tabdm']
            csingles=data['csingles']
            creps=data['creps']
            norm=data['norm']
            
        else:
            dm,s1,r1,b1,csingles,creps,tabdm,norm=compare_rsets(Rset,pset=pset,DMcut=DMcut,
                chime_response=chime_response,thresh=thresh)
            np.savez(savefile,dm=dm,s1=s1,
                r1=r1,b1=b1,csingles=csingles,creps=creps,tabdm=tabdm,norm=norm)
        
        # makes a joint distribution of all events
        call = np.concatenate((csingles,creps))
        #we want to compare the predictions from the grid for *all* bursts to the singles + reps model
        
        ddm = dm[1]-dm[0]
        
        
        #ax1.plot(dm,s1+r1,linestyle="--",linewidth=2)
        ax1.plot(dm,s1+b1,linestyle="-",linewidth=1,label=str(thresh))
        #ax2.plot(dm,s1+r1,linestyle="-",linewidth=2,label=str(thresh))
        
        # extracts k statistics
        if i==0:
            # Create cumulative distribution of CHIME single events, ready for the ks test
            corder = np.sort(call)
            idms = corder/ddm
            idms = idms.astype('int')
            chime_hist = np.zeros([dm.size])
            for idm in idms:
                chime_hist[idm] += 1
            cchist = np.cumsum(chime_hist)
            cchist /= cchist[-1]
            sqrtn = len(csingles)**0.5
            
        # makes cumulative sum from predictions
        cs1 = np.cumsum(s1+r1)
        cs1 /= cs1[-1]
        
        cs2 = np.cumsum(s1+b1)
        cs2 /= cs2[-1]
        
        # what we should be doing
        kstat=sp.stats.ks_1samp(corder,cdf,args=(dm,cs1),alternative='two-sided',mode='exact')
        
        #what is actually being done
        kstat2=sp.stats.ks_1samp(corder,cdf,args=(dm,cs2),alternative='two-sided',mode='exact')
        
        
        kstats[i] = kstat2[1]
        norms[i] = norm
        
        
        print("The k stats and the norm for threshold ",thresh, " are ",kstat2[1],norm)
        
    plt.sca(ax1)
    bins=np.linspace(0,4000,21)
    plt.hist(call,bins=bins,alpha=0.5,label='CHIME: all progenitors',edgecolor='black')
    plt.legend(fontsize=8)
    plt.xlim(0,4000)
    
    plt.tight_layout()
    plt.savefig(opdir+'best_fit_all_progenitors.pdf')
    plt.close()
    
    # this will be the unnormed values! Need this...
    plt.sca(ax2)
    plt.close()
    
    
    plt.figure()
    plt.xlabel('Threshold')
    plt.ylabel('$\\log_{10} p(KS)$')
    l1=plt.plot(thresholds,np.log10(kstats),color='red',linestyle='-',label='p(KS)')
    
    ax=plt.gca()
    ax2=ax.twinx()
    l2=ax2.plot(thresholds,np.log10(482./norms),color='blue',linestyle='--',label='norm')
    
    ax1.legend(handles=[l1,l2],labels=['p(KS)','norm'])
    
    ax2.set_ylabel('$\\log_{10} N_{\\rm FRB}$')
    
    ax2.plot([0.5,8],[np.log10(482),np.log10(482)],color='black',linestyle=':')
    
    plt.tight_layout()
    plt.savefig(opdir+'ks_norm_fig.pdf')
    plt.close()
        
def cdf(x,dm,cs):
    """
    Function to return a cdf given dm and cs via linear interpolation
    """
    nx = np.array(x)
    #y=np.zeros(nx.size)
    #y[x <= dm[0]]=0.
    #y[x >= dm[-1])=1.
    
    ddm = dm[1]-dm[0]
    ix1 = (x/ddm).astype('int')
    ix2 = ix1+1
    
    kx2 = x/ddm-ix1
    kx1 = 1.-kx2
    c = cs[ix1]*kx1 + cs[ix2]*kx2
    return c

def compare_rsets(Rset1,pset=None,Nbin=6,DMcut=None,chime_response=True,plot=False,thresh=5.):
    """
    Defined to test some corrections to the repeating FRBs method
    
    If pset=None, we use the defaults.
    """
    # old implementation
    # defines list of surveys to consider, together with Tpoint
    sdir = os.path.join(resource_filename('zdm','data/Surveys/'),'CHIME/')
    
    bdir='Nbounds'+str(Nbin)+'/'
    beams.beams_path = os.path.join(resource_filename('zdm','data/BeamData/CHIME/'),bdir)
    bounds = np.load(beams.beams_path+'bounds.npy')
    solids = np.load(beams.beams_path+'solids.npy')
    
    surveys=[]
    grids=[]
    reps1=[]
    reps2=[]
    state=set_state(pset=pset,chime_response = chime_response)
    
    tnsingles = 0
    tnreps = 0
    ttot1 = 0
    ttot2 = 0
    
    t0=time.time()
    for ibin in np.arange(Nbin):
        
        tag = str(bounds[ibin])+'$^{\\circ} < \\delta < $' + str(bounds[ibin+1])+'$^{\\circ}$'
        name = "CHIME_decbin_"+str(ibin)+"_of_"+str(Nbin)
        s,g = ute.survey_and_grid(survey_name=name,NFRB=None,sdir=sdir,init_state=state,thresh=thresh) # should be equal to actual number of FRBs, but for this purpose it doesn't matter
        
        abdm = np.sum(g.rates,axis=0)
        
        t1=time.time()
        # calculates repetition info
        g.state.rep.Rmin = Rset1["Rmin"]
        g.state.rep.Rmax = Rset1["Rmax"]
        g.state.rep.Rgamma = Rset1["Rgamma"]
        rg1 = rep.repeat_Grid(g,Tfield=s.TOBS,Nfields=1,MC=False,opdir=None,bmethod=2)
        
        
        t2=time.time()
        surveys.append(s)
        grids.append(g)
        reps1.append(rg1)
        Times=s.TOBS #here, TOBS is actually soid angle in steradians, since beamfile contains time factor
        
        t0=t2
        plot=True
        
        # collapses CHIME dm distribution for repeaters and once-off burts
        rdm1 = np.sum(rg1.exact_reps,axis=0)
        sdm1 = np.sum(rg1.exact_singles,axis=0)
        rbdm1 = np.sum(rg1.exact_rep_bursts,axis=0)
        
        adm = np.sum(g.rates,axis=0)*s.TOBS*10**(g.state.FRBdemo.lC)
        
        if ibin==0:
            tabdm = abdm
            trdm1 = rdm1
            tsdm1 = sdm1
            tbdm1 = rbdm1
        else:
            tabdm += abdm
            trdm1 += rdm1
            tsdm1 += sdm1
            tbdm1 += rbdm1
            
        #gets histogram of CHIME bursts
        nreps = s.frbs['NREP']
        ireps = np.where(nreps>1)
        isingle = np.where(nreps==1)[0]
        
        if DMcut is not None:
            # narrows this down to the fraction at high Galactic latitudes
            OKdm = np.where(s.DMGs < DMcut)[0]
            ireps = np.intersect1d(ireps,OKdm,assume_unique=True)
            isingle = np.intersect1d(isingle,OKdm,assume_unique=True)
        
        # normalises to singles
        bins=np.linspace(0,4000,21)
        db = bins[1]-bins[0]
        tot1=np.sum(sdm1)
        nsingles = len(isingle)
        
        tnreps += len(ireps)
        tnsingles += nsingles
        ttot1 += tot1
        
        # this norm factor is per declination bin: interesting,
        # but subject to large fluctuations
        #norm1= nsingles/tot1*db/(g.dmvals[1]-g.dmvals[0])
        #print("Norm factor is ",norm1,nsingles,tot1)
        
        if ibin==0:
            alldmr=s.DMEGs[ireps]
            alldms=s.DMEGs[isingle]
        else:
            alldmr = np.concatenate((alldmr,s.DMEGs[ireps]))
            alldms = np.concatenate((alldms,s.DMEGs[isingle]))
        
    # in the simulations, the units are "per bin"
    # however, when comparing to a histogram, we need the "db" factor
    tnorm1= tnsingles/ttot1*db/(g.dmvals[1]-g.dmvals[0])
    
    # normalises to total progenitor rate
    norm1 = (tnsingles + tnreps)/(ttot1 + np.sum(tbdm1))
    print("Meow! ",tnsingles + tnreps,ttot1 + np.sum(tbdm1))
    #print("\n\nTotal norms were ",tnorm1,tnorm2)
    #print("Total predictions were ",tnsingles,ttot1,'\n\n')
    
    # normalise total burst distribution
    total_bursts = np.sum(tabdm)
    modelled = tnsingles + 17 # total singles plus repeaters
    tabdm *= modelled / total_bursts
    
    # returns these key values for future use...
    return g.dmvals,tsdm1*tnorm1,trdm1*tnorm1,tbdm1*tnorm1,alldms,alldmr,tabdm,norm1

def set_state(pset=None,chime_response=True):
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
    state.energy.lEmax = 41.63
    state.energy.gamma = -0.948
    state.energy.alpha = 1.03
    state.FRBdemo.sfr_n = 1.15
    state.host.lmean = 2.22
    state.host.lsigma = 0.57
    
    state.FRBdemo.lC = 1.963
    
    
    if chime_response:
        state.width.Wmethod=0 #only a single width bin
        state.width.Wbias="CHIME"
    
    
    if pset is not None:
        state.energy.lEmax = pset['lEmax']
        state.energy.gamma = pset['gamma']
        state.energy.alpha = pset['alpha']
        state.FRBdemo.sfr_n = pset['sfr_n']
        state.host.lsigma = pset['lsigma']
        state.host.lmean = pset['lmean']
        state.FRBdemo.lC = pset['lC']
    
    return state





main()
