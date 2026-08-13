""" 

THE DESCRIPTION BELOW IS LIKELY INACCURATE

This script creates example plots for a combination
of FRB surveys and repeat bursts. We consider
two repeater descriptions --- distributed, and strong
--- and two sets of parameters: the best-fit
results from Shin et al, and the best-fit and 90\%
error ranges from James et al.

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

import states as st
import scipy as sp

import matplotlib
import time
from zdm import beams
beams.beams_path = '/Users/cjames/CRAFT/FRB_library/Git/H0paper/papers/Repeaters/BeamData/'
    

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
    Rset1={"Rmin":1e-4,"Rmax":10,"Rgamma":-2.2}
    Rset2={"Rmin":3.9,"Rmax":4,"Rgamma":-1.1}
    #sets=[Rset1,Rset2]
    
    
    ######## We iterate through the different parameter sets, using the CHIME response function #######
    # standard analysis
    extreme_rsets(Rset1,Rset2,opdir='Extremes/chime_',DMcut=None,chime_response=True,allplot=True,load=True,addshin=True)
    
    extreme_rsets(Rset1,Rset2,opdir='Extremes/chime_DM50_',DMcut=50.,chime_response=True,allplot=True,load=True,addshin=True)
    
def MyExp(x, m, t):
    return m * np.exp(-x/t)

        
def extreme_rsets(Rset1,Rset2,opdir='Extremes/',DMcut=None,chime_response=True,allplot=True,load=False,addshin=False):
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
    sdir = os.path.join(resource_filename('zdm','../'),'papers/Repeaters/Surveys')
    
    if not os.path.exists(opdir):
        os.mkdir(opdir)
    
    Nbin=6
    surveys=[]
    grids=[]
    reps1=[]
    reps2=[]
    psets=st.read_extremes()
    pset = st.shin_fit()
    
    F0s = np.linspace(0.5,5,10)
    
    ####### singles plots
    fig=plt.figure()
    ax1 = plt.gca()
    plt.xlim(0,3000)
    plt.xlabel('${\\rm DM}_{\\rm EG}$')
    plt.ylabel('$N({\\rm DM}_{\\rm EG}) \\, [200\\,{\\rm pc}\\,{\\rm cm}^{-3}]^{-1}$')
        
    # was all. Now will be strong distribution
    fig=plt.figure()
    ax2 = plt.gca()
    plt.xlim(0,3000)
    plt.xlabel('${\\rm DM}_{\\rm EG}$')
    plt.ylabel('$N({\\rm DM}_{\\rm EG}) \\, [200\\,{\\rm pc}\\,{\\rm cm}^{-3}]^{-1}$')
    
    
    ####### progenitor plots
    fig=plt.figure()
    ax3 = plt.gca()
    plt.xlim(0,3000)
    plt.xlabel('${\\rm DM}_{\\rm EG}$')
    plt.ylabel('$N({\\rm DM}_{\\rm EG}) \\, [200\\,{\\rm pc}\\,{\\rm cm}^{-3}]^{-1}$')
        
    # was all. Now will be strong distribution
    fig=plt.figure()
    ax4 = plt.gca()
    plt.xlim(0,3000)
    plt.xlabel('${\\rm DM}_{\\rm EG}$')
    plt.ylabel('$N({\\rm DM}_{\\rm EG}) \\, [200\\,{\\rm pc}\\,{\\rm cm}^{-3}]^{-1}$')
        
    
    for i,F0 in enumerate(F0s):
        #state=set_state()
        # sets parameters in state
        prefix=opdir+str(i)
        savefile=opdir+'F0_'+str(i)+'.npz'
        if load and os.path.exists(savefile):
            data=np.load(savefile)
            dm=data['dm']
            s1=data['s1']
            r1=data['r1']
            s2=data['s2']
            r2=data['r2']
            n1=data['n1']
            n2=data['n2']
            b1=data['b1']
            b2=data['b2']
            tabdm=data['tabdm']
            csingles=data['csingles']
            creps=data['creps']
            
        else:
            dm,s1,r1,b1,s2,r2,b2,csingles,creps,tabdm,n1,n2=compare_rsets(Rset1,Rset2,pset=pset,prefix=prefix,DMcut=DMcut,
                chime_response=True,F0=F0)
            np.savez(savefile,dm=dm,s1=s1,r1=r1,b1=b1,s2=s2,r2=r2,b2=b2,\
                csingles=csingles,creps=creps,tabdm=tabdm,n1=n1,n2=n2)
        
        ddm = dm[1]-dm[0]
        scale = 200./(dm[1]-dm[0])
        
        # extracts k statistics
        if i==0:
            # Create cumulative distribution of CHIME single events, ready for the ks test
            corder = np.sort(csingles)
            idms = corder/ddm
            idms = idms.astype('int')
            chime_hist = np.zeros([dm.size])
            for idm in idms:
                chime_hist[idm] += 1
            cchist = np.cumsum(chime_hist)
            cchist /= cchist[-1]
            sqrtn = len(csingles)**0.5
            
            cprog = np.concatenate((csingles,creps))
            cprog = np.sort(cprog)
            idms = cprog/ddm
            idms = idms.astype('int')
            cprog_chime_hist = np.zeros([dm.size])
            for idm in idms:
                cprog_chime_hist[idm] += 1
            cphist = np.cumsum(cprog_chime_hist)
            cphist /= cphist[-1]
            
            
        
        # makes cumulative sum from predictions
        cs1 = np.cumsum(s1)
        cs1 /= cs1[-1]
        
        cs2 = np.cumsum(s2)
        cs2 /= cs2[-2]
        
        kstat1=sp.stats.ks_1samp(corder,cdf,args=(dm,cs1),alternative='two-sided',mode='exact')
        kstat2=sp.stats.ks_1samp(corder,cdf,args=(dm,cs2),alternative='two-sided',mode='exact')
        
        
        ax1.plot(dm,s1/n1,linestyle="--",linewidth=1,label=str(F0)+', k='+str(kstat1)[0:5])
        ax2.plot(dm,s2/n2,linestyle='--',linewidth=1,label=str(F0)+', k='+str(kstat2)[0:5])
        
        # makes cumulative sum from predictions
        p1 = r1+s1
        p2 = r2+s2
        a1 = s1+b1
        a2 = s2+b2
        
        cp1 = np.cumsum(p1)
        cp1 /= cp1[-1]
        
        cp2 = np.cumsum(p2)
        cp2 /= cp2[-2]
        
        ca1 = np.cumsum(a1)
        ca1 /= ca1[-1]
        
        ca2 = np.cumsum(a2)
        ca2 /= ca2[-1]
        
        # this k-stat tests for predicted progenitors vs observed progenitors
        # Relly should test predicted singles vs observed progenitors!!!
        kstat3=sp.stats.ks_1samp(cprog,cdf,args=(dm,ca1),alternative='two-sided',mode='exact')
        kstat4=sp.stats.ks_1samp(cprog,cdf,args=(dm,ca2),alternative='two-sided',mode='exact')
        ax3.plot(dm,a1/n1,linestyle="--",linewidth=1,label=str(F0)+', k='+str(kstat1)[0:5])
        ax4.plot(dm,a2/n2,linestyle='--',linewidth=1,label=str(F0)+', k='+str(kstat2)[0:5])
        print("F0: ",F0," kstats singles ",kstat1.pvalue,kstat2.pvalue," progenitors ",kstat3.pvalue,kstat4.pvalue,"norms ",n1,n2)
        
        
        #ddm = dm[1]-dm[0]
    plt.sca(ax1)
    plt.text(0.54,0.91,'Distributed repeaters',transform=ax1.transAxes)
    bins=np.linspace(0,4000,21)
    plt.hist(csingles,bins=bins,alpha=0.5,label='CHIME: single bursts',edgecolor='black')
    plt.hist(creps,bins=bins,alpha=0.5,label='repeating sources',edgecolor='black')
    plt.legend()
    plt.tight_layout()
    plt.savefig(opdir+'temp1.pdf')
    plt.close()
    
    plt.sca(ax2)
    plt.text(0.54,0.91,'Strong repeaters',transform=ax2.transAxes)
    plt.hist(csingles,bins=bins,alpha=0.5,label='CHIME: single bursts',edgecolor='black')
    plt.hist(creps,bins=bins,alpha=0.5,label='repeating sources',edgecolor='black')
    plt.tight_layout()
    plt.savefig(opdir+'temp2.pdf')
    plt.close()
    
    
    plt.sca(ax3)
    plt.ylim(0,200)
    plt.text(0.54,0.91,'Comparison with all bursts',transform=ax1.transAxes)
    bins=np.linspace(0,4000,21)
    plt.hist(cprog,bins=bins,alpha=0.5,label='CHIME: all progenitors',edgecolor='black')
    plt.legend()
    plt.tight_layout()
    plt.savefig(opdir+'temp3.pdf')
    plt.close()
    
    #plt.sca(ax4)
    #plt.ylim(0,200)
    #plt.text(0.54,0.91,'Strong repeaters',transform=ax2.transAxes)
    #plt.hist(cprog,bins=bins,alpha=0.5,label='CHIME: all progenitors',edgecolor='black')
    #plt.tight_layout()
    #plt.savefig('temp4.pdf')
    #plt.close()
    
    print("Done!")

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
      
def fit_chime_data(Rset,pset=None):
    """
    Defined to test some corrections to the repeating FRBs method
    """
    # old implementation
    from zdm import old_working_repeat_grid as orep
    # defines list of surveys to consider, together with Tpoint
    sdir = os.path.join(resource_filename('zdm','../'),'papers/Repeaters/Surveys')
    
    Nbin=6
    ss=[]
    gs=[]
    rgs=[]
    state=set_state(pset=pset)
    t0=time.time()
    for ibin in np.arange(Nbin):
        name = "CHIME_decbin"+str(ibin)
        s,g = survey_and_grid(survey_name=name,NFRB=None,sdir=sdir,init_state=state) # should be equal to actual number of FRBs, but for this purpose it doesn't matter
        t1=time.time()
        # calculates repetition info
        g.state.rep.Rmin = Rset["Rmin"]
        g.state.rep.Rmax = Rset["Rmax"]
        g.state.rep.Rgamma = Rset["Rgamma"]
        
        rg = rep.repeat_Grid(g,Tfield=s.TOBS,Nfields=1,MC=False,opdir=None,bmethod=2)
        org = orep.repeat_Grid(g,Tfield=s.TOBS,Nfields=1,MC=False,opdir=None,bmethod=2)
        
        t2=time.time()
        ss.append(s)
        gs.append(g)
        rgs.append(rg)
        Times=s.TOBS #here, TOBS is actually soid angle in steradians, since beamfile contains time factor
        print("Took total of ",t1-t0,t2-t1," seconds for init of survey ",ibin)
        t0=t2
    
    #calculates normalisation factor
    NcTOT=0.
    NrgTOT=0.
    for ibin in np.arange(Nbin):
        rg=rgs[ibin]
        g=gs[ibin]
        
        Nrg = np.sum(rg.exact_singles)
        
        #number of CHIME bursts
        nreps=s.frbs['NREP']
        ireps=np.where(nreps>1)
        isingle=np.where(nreps==1)[0]
        Nc = len(isingle)
        
        NrgTOT += Nrg
        NcTOT += Nc
    
    # normalisation constant    
    norm = NcTOT / NrgTOT
    
    print("Global norm is ",norm)

def compare_rsets(Rset1,Rset2,pset=None,prefix="",Nbin=6,DMcut=None,chime_response=True,plot=False,F0=5.):
    """
    Defined to test some corrections to the repeating FRBs method
    
    If pset=None, we use the defaults.
    """
    # old implementation
    # defines list of surveys to consider, together with Tpoint
    sdir = os.path.join(resource_filename('zdm','../'),'papers/Repeaters/Surveys')
    
    bdir='Nbounds'+str(Nbin)+'/'
    beams.beams_path = os.path.join(resource_filename('zdm','../'),'papers/Repeaters/'+bdir)
    bounds = np.load(bdir+'bounds.npy')
    solids = np.load(bdir+'solids.npy')
    
    surveys=[]
    grids=[]
    reps1=[]
    reps2=[]
    state=set_state(pset=pset,chime_response = chime_response)
    
    tnsingles = 0
    ttot1 = 0
    ttot2 = 0
    
    t0=time.time()
    for ibin in np.arange(Nbin):
        
        tag = str(bounds[ibin])+'$^{\\circ} < \\delta < $' + str(bounds[ibin+1])+'$^{\\circ}$'
        name = "CHIME_decbin_"+str(ibin)+"_of_"+str(Nbin)
        s,g = survey_and_grid(survey_name=name,NFRB=None,sdir=sdir,init_state=state,F0=F0) # should be equal to actual number of FRBs, but for this purpose it doesn't matter
        
        abdm = np.sum(g.rates,axis=0)
        
        t1=time.time()
        # calculates repetition info
        g.state.rep.Rmin = Rset1["Rmin"]
        g.state.rep.Rmax = Rset1["Rmax"]
        g.state.rep.Rgamma = Rset1["Rgamma"]
        rg1 = rep.repeat_Grid(g,Tfield=s.TOBS,Nfields=1,MC=False,opdir=None,bmethod=2)
        
        g.state.rep.Rmin = Rset2["Rmin"]
        g.state.rep.Rmax = Rset2["Rmax"]
        g.state.rep.Rgamma = Rset2["Rgamma"]
        rg2 = rep.repeat_Grid(g,Tfield=s.TOBS,Nfields=1,MC=False,opdir=None,bmethod=2)
        
        t2=time.time()
        surveys.append(s)
        grids.append(g)
        reps1.append(rg1)
        reps2.append(rg2)
        Times=s.TOBS #here, TOBS is actually soid angle in steradians, since beamfile contains time factor
        
        t0=t2
        plot=True
        
        # collapses CHIME dm distribution for repeaters and once-off burts
        rdm1 = np.sum(rg1.exact_reps,axis=0)
        sdm1 = np.sum(rg1.exact_singles,axis=0)
        rbdm1 = np.sum(rg1.exact_rep_bursts,axis=0)
        
        adm = np.sum(g.rates,axis=0)*s.TOBS*10**(g.state.FRBdemo.lC)
        
        rdm2 = np.sum(rg2.exact_reps,axis=0)
        sdm2 = np.sum(rg2.exact_singles,axis=0)
        rbdm2 = np.sum(rg2.exact_rep_bursts,axis=0)
        
        if ibin==0:
            tabdm = abdm
            trdm1 = rdm1
            tsdm1 = sdm1
            trbdm1 = rbdm1
            trdm2 = rdm2
            tsdm2 = sdm2
            trbdm2 = rbdm2
        else:
            tabdm += abdm
            trdm1 += rdm1
            tsdm1 += sdm1
            trbdm1 += rbdm1
            trdm2 += rdm2
            tsdm2 += sdm2
            trbdm2 += rbdm2
            
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
        tot2=np.sum(sdm2)
        nsingles = len(isingle)
        
        tnsingles += nsingles
        ttot1 += tot1
        ttot2 += tot2
        
        norm1= nsingles/tot1*db/(g.dmvals[1]-g.dmvals[0])
        norm2= nsingles/tot2*db/(g.dmvals[1]-g.dmvals[0])
        print("Norm factor is ",norm1,norm2,nsingles,tot1,tot2)
        
        if plot:
            fig=plt.figure()
            themax=np.max(sdm1*norm1)
            plt.text(1800,themax*0.5,tag)
                
            plt.plot(g.dmvals,sdm1*norm1,label='Distributed: single bursts',linestyle="-",linewidth=2)
            plt.plot(g.dmvals,rdm1*norm1,label='repeating sources',linestyle="-.",linewidth=2)
            #plt.plot(g.dmvals,(rbdm1+sdm1)*norm1,label='all',linestyle='-')
            
            plt.plot(g.dmvals,sdm2*norm2,label='Strong: single bursts',linestyle='--',linewidth=2)
            plt.plot(g.dmvals,rdm2*norm2,label='repeating sources',linestyle=':',linewidth=2)
            #plt.plot(g.dmvals,(rbdm2+sdm2)*norm2,label='ll',linestyle='--')
            
            h,b=np.histogram(s.DMEGs[isingle],bins=bins)
            if ibin==0:
                alldmr=s.DMEGs[ireps]
                alldms=s.DMEGs[isingle]
                print(type(alldms))
            else:
                alldmr = np.concatenate((alldmr,s.DMEGs[ireps]))
                alldms = np.concatenate((alldms,s.DMEGs[isingle]))
        
            #plt.plot(g.dmvals,adm*norm1,label='All bursts (n1)',linestyle=':')
            plt.hist(s.DMEGs[isingle],bins=bins,alpha=0.5,label='CHIME: single bursts',edgecolor='black')
            if len(ireps)>0:
                plt.hist(s.DMEGs[ireps],bins=bins,alpha=0.5,label='repeating sources',edgecolor='black')
            plt.xlim(0,3000)
            plt.xlabel('${\\rm DM}_{\\rm EG}$')
            plt.ylabel('$N({\\rm DM}_{\\rm EG}) [200\\,{\\rm pc}\\,{\\rm cm}^{-3}]$')
            legend=plt.legend()
            renderer = fig.canvas.get_renderer()
            max_shift = max([t.get_window_extent(renderer).width for t in legend.get_texts()])
            for t in legend.get_texts():
                #    t.set_ha('right') # ha is alias for horizontalalignment
                temp_shift = max_shift - t.get_window_extent().width
                t.set_position((temp_shift*0.7,0))
            plt.tight_layout()
            plt.savefig(prefix+'_'+str(ibin)+'.pdf')
            plt.close()
    # This norm is relevant for comparing histograms and plotting
    # It is *NOT* relevant for comparing predictions!
    tnorm1= tnsingles/ttot1*db/(g.dmvals[1]-g.dmvals[0])
    tnorm2= tnsingles/ttot2*db/(g.dmvals[1]-g.dmvals[0])
    #print("\n\nTotal norms were ",tnorm1,tnorm2)
    print("Total predictions were ",tnsingles,ttot1,ttot2,'\n\n')
    # The following is the relevant physical norm
    rnorm1 = tnsingles/ttot1
    rnorm2 = tnsingles/ttot2
    
    if plot:
        fig=plt.figure()
        plt.text(1800,50,'All declinations')
        plt.xlim(0,3000)
        plt.plot(g.dmvals,tsdm1*tnorm1,label='Distributed: single bursts',linestyle="-",linewidth=2)
        plt.plot(g.dmvals,trdm1*tnorm1,label='repeating sources',linestyle="-.",linewidth=2)
        plt.plot(g.dmvals,tsdm2*tnorm2,label='Strong: single bursts',linestyle='--',linewidth=2)
        plt.plot(g.dmvals,trdm2*tnorm2,label='repeating sources',linestyle=':',linewidth=2)
        
        plt.hist(alldms,bins=bins,alpha=0.5,label='CHIME: single bursts',edgecolor='black')
        plt.hist(alldmr,bins=bins,alpha=0.5,label='repeating sources',edgecolor='black')
        legend=plt.legend()
        renderer = fig.canvas.get_renderer()
        max_shift = max([t.get_window_extent(renderer).width for t in legend.get_texts()])
        for t in legend.get_texts():
            temp_shift = max_shift - t.get_window_extent().width
            t.set_position((temp_shift*0.7,0))
        plt.tight_layout()
        plt.savefig(prefix+'_alldec.pdf')
        plt.close()
    
    # normalise total burst distribution
    total_bursts = np.sum(tabdm)
    modelled = tnsingles + 17 # total singles plus repeaters
    tabdm *= modelled / total_bursts
    
    # returns these key values for future use...
    # wait - do NOT use tnorm, we can do that later!
    return g.dmvals,tsdm1*tnorm1,trdm1*tnorm1,trbdm1*tnorm1, tsdm2*tnorm2,trdm2*tnorm2,\
        trbdm2*tnorm2,alldms,alldmr,tabdm,rnorm1,rnorm2

def compare_old_new(Rset,plot=True):
    """
    Defined to test some corrections to the repeating FRBs method
    """
    # old implementation
    from zdm import old_working_repeat_grid as orep
    # defines list of surveys to consider, together with Tpoint
    sdir = os.path.join(resource_filename('zdm','../'),'papers/Repeaters/Surveys')
    
    Nbin=6
    surveys=[]
    grids=[]
    reps=[]
    state=set_state()
    t0=time.time()
    for ibin in np.arange(Nbin):
        name = "CHIME_decbin"+str(ibin)
        s,g = survey_and_grid(survey_name=name,NFRB=None,sdir=sdir,init_state=state) # should be equal to actual number of FRBs, but for this purpose it doesn't matter
        t1=time.time()
        # calculates repetition info
        g.state.rep.Rmin = Rset["Rmin"]
        g.state.rep.Rmax = Rset["Rmax"]
        g.state.rep.Rgamma = Rset["Rgamma"]
        
        rg = rep.repeat_Grid(g,Tfield=s.TOBS,Nfields=1,MC=False,opdir=None,bmethod=2)
        org = orep.repeat_Grid(g,Tfield=s.TOBS,Nfields=1,MC=False,opdir=None,bmethod=2)
        
        t2=time.time()
        surveys.append(s)
        grids.append(g)
        reps.append(rg)
        Times=s.TOBS #here, TOBS is actually soid angle in steradians, since beamfile contains time factor
        print("Took total of ",t1-t0,t2-t1," seconds for init of survey ",ibin)
        t0=t2
        plot=True
        if plot:
            # collapses CHIME dm distribution for repeaters and once-off burts
            rdm = np.sum(rg.exact_reps,axis=0)
            sdm = np.sum(rg.exact_singles,axis=0)
            rbdm = np.sum(rg.exact_rep_bursts,axis=0)
            adm = np.sum(g.rates,axis=0)*s.TOBS*10**(g.state.FRBdemo.lC)
            
            ordm = np.sum(org.exact_reps,axis=0)
            osdm = np.sum(org.exact_singles,axis=0)
            orbdm = np.sum(org.exact_rep_bursts,axis=0)
            
            #gets histogram of CHIME bursts
            nreps=s.frbs['NREP']
            ireps=np.where(nreps>1)
            isingle=np.where(nreps==1)[0]
            
            # normalises to singles
            bins=np.linspace(0,4000,21)
            db = bins[1]-bins[0]
            tot=np.sum(osdm)#*db/(g.dmvals[1]-g.dmvals[0])
            nsingles = len(isingle)
            norm = nsingles/tot*db/(g.dmvals[1]-g.dmvals[0])
            print("Norm factor is ",norm,nsingles,tot)
            
            plt.figure()
            plt.plot(g.dmvals,sdm*norm,label='singles')
            plt.plot(g.dmvals,rdm*norm,label='repeaters')
            plt.plot(g.dmvals,(rbdm+sdm)*norm,label='all from rg')
            
            plt.plot(g.dmvals,osdm*norm,label='orig singles',linestyle='--')
            plt.plot(g.dmvals,ordm*norm,label='orig repeaters',linestyle='--')
            plt.plot(g.dmvals,(orbdm+osdm)*norm,label='orig all from rg',linestyle='--')
            
            plt.plot(g.dmvals,adm*norm,label='all from grid',linestyle=':')
            
            plt.hist(s.DMEGs[isingle],bins=bins,alpha=0.5,label='CHIME once-off bursts')
            if len(ireps)>0:
                plt.hist(s.DMEGs[ireps],bins=bins,alpha=0.5,label='CHIME repeaters')
            plt.xlim(0,3000)
            plt.xlabel('${\\rm DM}_{\\rm EG}$')
            plt.ylabel('p(DM)')
            plt.legend()
            plt.tight_layout()
            plt.savefig('DecbinFigs/bin_'+str(ibin)+'.pdf')
            plt.close()
            
            
            plt.figure()
            
            
            plt.ylim(0,5)
            plt.plot(g.dmvals,(rbdm+sdm)/adm,label='Ratio: all')
            plt.plot(g.dmvals,sdm/osdm,label='singles')
            plt.plot(g.dmvals,rdm/ordm,label='repeaters')
            plt.xlim(0,3000)
            plt.xlabel('dm')
            plt.ylabel('ratio new/old')
            plt.legend()
            plt.tight_layout()
            plt.savefig('ratio_repeat_grids.pdf')
            exit()


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


def survey_and_grid(survey_name:str='CRAFT/CRACO_1_5000',
            init_state=None,state_dict=None, iFRB:int=0,
               alpha_method=1, NFRB:int=100, 
               lum_func:int=2,sdir=None,F0=5.):
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
    if sdir is None:
        sdir = os.path.join(resource_filename('zdm', 'craco'), 'MC_Surveys')
    
    
    isurvey = survey.load_survey(survey_name, state, dmvals,
                                 NFRB=NFRB, sdir=sdir, Nbeams=5,
                                 iFRB=iFRB)
    
    # modifes survey to have an artificial threshold
    isurvey.meta["THRESH"]=F0
    isurvey.THRESHs[:]=F0
    
    # generates zdm grid
    grids = misc_functions.initialise_grids(
        [isurvey], zDMgrid, zvals, dmvals, state, wdist=True)
    print("Initialised grid")

    # Return Survey and Grid
    return isurvey, grids[0]


main()
