"""

This script produces all plots from cases a,b,c,d

It first loops through these four cases, generating information
about the DM, dec, Nrep, and N(t) distributions of repeaters.
It uses Ndec = 30 declination bins for maximum accuracy
(this mostly only matters for the declination distribution,
and mostly only cosmetically even then).

It then generates the plots featured in the paper, compared
to CHIME distributions.

Currently however, there is a memory leak somewhere in the code,
and this means that each case tends to consume an additional
~6 GB of RAM. Hence, in the code is automatically configured 
to run for only one of each of the four cases, save the output
to a temp file, then exit. Plots are thus only generated when
all four cases are complete, at which point the temporary info
is saved to a common file.

Plots produced are:
- MChistogram
- cumMChistogram (both baed on number of repetitions)



"""
from pkg_resources import resource_filename

import numpy as np
from matplotlib import pyplot as plt
import os
import pickle

import utilities as ute

from zdm.craco import loading
from zdm import cosmology as cos
from zdm import repeat_grid as rep
from zdm import misc_functions
from zdm import survey
from zdm import beams
beams.beams_path = '/Users/cjames/CRAFT/FRB_library/Git/H0paper/papers/Repeaters/BeamData/'

import matplotlib
matplotlib.rcParams['image.interpolation'] = None

defaultsize=14
ds=4
font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : defaultsize}
matplotlib.rc('font', **font)

global opdir
opdir = 'BestfitCalculations/'
if not os.path.exists(opdir):
    os.mkdir(opdir)

def main(Nbin=30):
    Nsets=5
    global opdir
    dims=['$R_{\\rm min}$','$R_{\\rm max}$','$\\gamma_{r}$']
    pdims=['$p(R_{\\rm min})$','$p(R_{\\rm max})$','$p(\\gamma_{r})$']
    names=['Rmin','Rmax','Rgamma']
    string=['Ptot','Pn','Pdm']
    pstrings=['$P_{\\rm tot}$','$P_{N}$','$P_{\\rm DM}$']
    
    
    tmults = np.logspace(0.5,4,8)
    Nmults = tmults.size+1
    
    setnames = ['$\\epsilon_C: {\\rm min}\\, \\alpha$','$\\epsilon_z:$ Best fit','${\\rm min}\\, E_{\\rm max}$',
        '${\\rm max} \\,\\gamma$','${\\rm min} \\, \\sigma_{\\rm host}$']
    
    # gets the possible states for evaluation
    states,names=get_states()
    rlist=[]
    ndm = 1400
    rlist= np.zeros([Nsets,ndm])
    
    tms = np.zeros([Nsets,Nmults])
    
    NMChist = 100
    mcrs=np.linspace(1.5,0.5+NMChist,NMChist) # from 1.5 to 100.5 inclusive
    
    tmults=np.logspace(-1,3,9)
    
    sm=['a','b','c','d']
    for i in np.arange(Nsets):
        
        savefile = opdir+'set_'+str(i)+'_saved_values.npz'
        #if os.path.exists(savefile):
        #    continue
        
        
        infile='Rfitting/converge_set_'+str(i)+'__output.npz'
        
        data=np.load(infile)
        
        lps=data['arr_0']
        oldlns=data['arr_1']
        ldms=data['arr_2']
        
        lns=data['arr_3']
        NRs=data['arr_4']
        
        Rmins=data['arr_5']
        Rmaxes=data['arr_6']
        Rgammas=data['arr_7']
        
        xRmax=np.array([-0.25,0.25,3,3])
        sy=np.array([-1.3,-3.0,-3.0,-2.0])
        xRgamma = sy + 0.001 # adjusts for offset
        
        sm=['a','b','c','d']
        
        
        #fourRmins=[]
        #fourRmaxes=[]
        #fourRgammas=[]
        #Fourdms=[]
        Fourss=[]
        Fourrs=[]
        Fourbs=[]
        Fnss=[]
        Fnrs=[]
        Fnbs=[]
        urmax=[]
        urmin=[]
        urgamma=[]
        MCs=[]
        fMCs=[]
        cMCs=[]
        Tms=[]
        Tmr=[]
        Tmb=[]
        
        
        for j,Rgamma in enumerate(xRgamma):
            
            lRmax = xRmax[j]
            Rmax = 10**lRmax
            irg = np.where(Rgammas == Rgamma)[0][0]
            irm = np.where(Rmaxes == Rmax)[0][0]
            Rmin = Rmins[irg,irm]
            
            states[i].rep.Rmin = Rmin
            states[i].rep.Rmax = Rmax
            states[i].rep.Rgamma = Rgamma
            print("Parameters: ",Rmin,Rmax,Rgamma)
            
            tempsave = opdir+"temp"+str(j)+".npz"
            
            if os.path.exists(tempsave):
                data = np.load(tempsave)
                
                
                dms=data['arr_0']
                ss=data['arr_1']
                rs=data['arr_2']
                bs=data['arr_3']
                nss=data['arr_4']
                nrs=data['arr_5']
                nbs=data['arr_6']
                Mh=data['arr_7']
                fitv=data['arr_8']
                copyMh=data['arr_9']
                Cnreps=data['arr_10']
                Cnss=data['arr_11']
                Cnrs=data['arr_12']
                tms=data['arr_13']
                tmr=data['arr_14']
                tmb=data['arr_15']
            else:
                print("Doing ",j)
                # returns dms, singles,repeaters, bursts, and same for tmult data, all as f(dm)
                dms,ss,rs,bs,nss,nrs,nbs,Mh,fitv,copyMh,Cnreps,Cnss,Cnrs,tms,tmr,tmb = \
                    generate_state(states[i],mcrs=mcrs,tag=str(i)+"_"+sm[j],Rmult=1000.,tmults=tmults,Nbin=Nbin)
            
                np.savez(tempsave,dms,ss,rs,bs,nss,nrs,nbs,Mh,fitv,copyMh,Cnreps,Cnss,Cnrs,tms,tmr,tmb)
                exit()
            
            Fourss.append(ss)
            Fourrs.append(rs)
            Fourbs.append(bs)
            Fnss.append(nss)
            Fnrs.append(nrs)
            Fnbs.append(nbs)
            urmax.append(Rmax)
            urmin.append(Rmin)
            urgamma.append(Rgamma)
            Tms.append(tms)
            Tmr.append(tmr)
            Tmb.append(tmb)
            
            MCs.append(Mh)
            fMCs.append(fitv)
            cMCs.append(copyMh)
            
        np.savez(savefile,dms,Fourss,Fourrs,Fourbs,urmax,urmin,urgamma,Fnss,Fnrs,Fnbs,MCs,fMCs,cMCs,\
            Cnreps,Cnss,Cnrs,Tms,Tmr,Tmb)
        break
    
    for i in np.arange(Nsets):
        savefile = opdir+'set_'+str(i)+'_saved_values.npz'
        data = np.load(savefile)
        dms = data['arr_0']
        Fourss=data['arr_1']
        Fourrs=data['arr_2']
        Fourbs=data['arr_3']
        
        urmax=data['arr_4']
        urmin=data['arr_5']
        urgamma=data['arr_6']
        
        
        Fnss=data['arr_7']
        Fnrs=data['arr_8']
        Fnbs=data['arr_9']
        
        MCs=data['arr_10']
        fMCs=data['arr_11']
        cMCs=data['arr_12']
        Cnreps=data['arr_13']
        Cnss=data['arr_14']
        Cnrs=data['arr_15']
        
        Tms=data['arr_16']
        Tmr=data['arr_17']
        Tmb=data['arr_18']
        break
    
    
    ##### sets some labels and styles for all pots #####
    linestyles=['-','--','-.',':']
    labels=['case a','case b','case c','case d']
    #### time evolution plot #####
    
    # re-organises tmult order
    for i in np.arange(4):
        break
        
        # correct for the fact that we incorrectly add Tobs=1 here
        #Tms[i] = np.array(Tms[i])
        #Tmr[i] = np.array(Tmr[i])
        #Tmb[i] = np.array(Tmb[i])
        #Tms[i] = Tms[i][1:]
        #Tmr[i] = Tmr[i][1:]
        #Tmb[i] = Tmb[i][1:]
        print(tmults.shape)
        print(Tms[i].shape)
        
        #ts=Tms[i][0]
        #tr=Tms[i][0]
        #tb=Tms[i][0]
        #Tms[i][0:2] = Tms[i][1:3]
        #Tmr[i][0:2] = Tmr[i][1:3]
        #Tmb[i][0:2] = Tmb[i][1:3]
        #Tms[i][2] = ts
        #Tmr[i][2] = tr
        #Tmb[i][2] = tb
    
    
    plt.figure()
    plt.xlabel('Time $T \\, [{\\rm per}~T_{\\rm Cat 1}]$')
    plt.ylabel('Repeaters per time: $N_{\\rm rep} T^{-1}$')
    
    for i,rs in enumerate(Fourrs):
        label=labels[i]
        plt.plot(tmults,Tmr[i]/tmults,label=label,linestyle=linestyles[i],linewidth=3)
    plt.legend()
    plt.xscale('log')
    plt.tight_layout()
    plt.savefig(opdir+'set_0_time_effect.pdf')
    plt.close()
    exit()
    ########## DM distribution plot ########3
    
    bins = np.linspace(0,2000,21)
    ddm = dms[1]-dms[0]
    dbin = bins[1]-bins[0]
    scale = dbin/ddm
    
    nCsdms,nCrdms,nCsdecs,nCrdecs = ute.get_chime_dec_dm_data(DMhalo=50,newdata='only')
    
    Csdms,Crdms,Csdecs,Crdecs = ute.get_chime_dec_dm_data(DMhalo=50,newdata=False)
    
    # plots hist of DM for repeaters over all best-fit options
    plt.figure()
    plt.xlim(0,2000)
    plt.xlabel('${\\rm DM}_{\\rm EG}$')
    plt.ylabel('$N_{\\rm rep}({\\rm DM}_{\\rm EG}) \\, [200\\,{\\rm pc}\\,{\\rm cm}^{-3}]^{-1}$')
    
    plt.hist(Crdms,bins=bins,alpha=1.0,label='CHIME catalog 1 (17)',edgecolor='black')
    print("Mean DM of CHIME repeaters is ",np.sum(Crdms)/Crdms.size)
    
    plt.hist(nCrdms,bins=bins,alpha=0.3,label='Golden sample (25)',edgecolor='black',\
        weights=np.full([25],17./25.))
    
    labels=['$a: [~\\,-10,-0.25,-1.3]$','$b: [-1.32,~~~0.25,~~~-3]$',\
        '$c: [-1.38,~~~~~~~~3,~~~-3]$','$d: [\\,-4.8,~~~~~~~~3,~~~-2]$']
    
    plt.xlim(0,1750)
    for i,rs in enumerate(Fourrs):
        label=labels[i]
        print("Total predicted repeaters are ",np.sum(rs))
        rs *= np.sum(Fourrs[3])/np.sum(rs)
        plt.plot(dms,rs*scale,label=label,linestyle=linestyles[i],linewidth=3)
        print("Mean predicted DM in model ",label," is ",np.sum(rs*dms)/np.sum(rs))
    
        
    plt.legend(title='$~~~~~~~~~~~~[\\log_{10} R_{\\rm min},\\log_{10} R_{\\rm max},R_{\\gamma}]$',title_fontsize=12)
    
    plt.tight_layout()
    plt.savefig(opdir+'set_0_four_cases_dm.pdf')
    plt.close()
    
    
    ############# produce an MC rate plot ###########
    
    plt.figure()
    
    
    #mids=(mcrs[1:] + mcrs[:-1])/2.
    
    vals = np.arange(2,101)
    plt.hist(vals,bins=mcrs,weights=Cnreps,label='CHIME Cat 1',alpha=0.5)
    
    rates = np.linspace(2,100,99)
    markers=['+','s','x','o']
    
    styles=['-','--','-.',':']
    for i,label in enumerate(sm):
        #scale = np.sum(Cnreps)/np.sum(MCs[i])
        scale=1.
        plt.plot(rates,MCs[i]*scale,linestyle="",marker=markers[i],label="case "+label)
        plt.plot(rates,fMCs[i]*scale,linestyle=styles[i],color=plt.gca().lines[-1].get_color(),linewidth=3)
    
    #plt.scatter(bcs[tooLow],Mh[tooLow],marker='x',s=10.)
    plt.legend()
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('$N_{\\rm reps}$')
    plt.ylabel('$N_{\\rm prog}(N_{\\rm reps})$')
    plt.tight_layout()
    plt.savefig(opdir+'MChistogram.pdf')
    plt.close()
    
    #### now does cumulative plot ###
    
    plt.figure()
    
    
    #mids=(mcrs[1:] + mcrs[:-1])/2.
    
    vals = np.arange(2,101)
    Ccum = np.cumsum(Cnreps)
    #Ccum = Ccum / Ccum[-1]
    
    plt.plot(rates,Ccum,label='CHIME',linewidth=3)
    
    rates = np.linspace(2,100,99)
    markers=['+','s','x','o']
    
    styles=['-','--','-.',':']
    for i,label in enumerate(sm):
        cumMC = np.cumsum(MCs[i])
        #cumMC /= cumMC[-1]
        plt.plot(rates,cumMC,linestyle=styles[i],label="case "+label,linewidth=3)
        #plt.plot(rates,fMCs[i]*scale,linestyle=styles[i],color=plt.gca().lines[-1].get_color())
    
    #plt.scatter(bcs[tooLow],Mh[tooLow],marker='x',s=10.)
    plt.legend()
    plt.xscale('log')
    #plt.yscale('log')
    plt.xlabel('$N_{\\rm bursts}$')
    plt.ylabel('$N_{\\rm rep}(N_{\\rm bursts})$')
    
    #ax=plt.gca()
    #ax.set_xticklabels([2,3,5,10,20,30,50,100])
    
    plt.tight_layout()
    plt.savefig(opdir+'cumMChistogram.pdf')
    plt.close()
    
    
    ######### produce decbin division plot ########
    
    bdir='Nbounds'+str(Nbin)+'/'
    bounds = np.load(bdir+'bounds.npy')
    
    plt.figure()
    
    plt.ylim(0,1)
    plt.xlabel('${\\delta}$ [deg]')
    plt.ylabel('$N_{\\rm rep}(\\delta)$')
    
    # adds on initial and final points
    #temp=np.array([-11,90.])
    #Crdecs = np.concatenate((Crdecs,temp))
    #Crdecs = np.sort(Crdecs)
    #yCrdecs = np.linspace(0,1.,Crdecs.size)
    #
    #nCrdecs = np.concatenate((nCrdecs,temp))
    #nCrdecs = np.sort(nCrdecs)
    #ynCrdecs = np.linspace(0,1.,nCrdecs.size)
    
    sx,sy,rx,ry = ute.get_chime_rs_dec_histograms(DMhalo=50)
    nsx,nsy,nrx,nry = ute.get_chime_rs_dec_histograms(DMhalo=50,newdata='only')
    
    #plt.plot(Crdecs,yCrdecs,label='CHIME repeaters (cat1)')
    #plt.plot(nCrdecs,ynCrdecs,label='CHIME repeaters (3 yr)')
    
    plt.plot(rx,ry,label='CHIME repeaters (cat1)')
    plt.plot(nrx,nry,label='CHIME repeaters (3 yr)')
    
    for i,label in enumerate(sm):
        
        cumdec = np.cumsum(Fnrs[i])
        cumdec = cumdec/cumdec[-1]
        cumdec = np.concatenate(([0],cumdec))
        plt.plot(bounds,cumdec,linestyle=styles[i],label="case "+label)
    
    plt.legend()
    plt.tight_layout()
    plt.savefig(opdir+'cumulative_dec_set0.pdf')
    plt.close()
    
    
    
    
    exit()
    
    
    
    
    
    
    ##### we now produce a tmult plot ######
    
    plt.figure()
    plt.xlabel('time [Catalog 1 = 1]')
    plt.ylabel('Rate of repeater discovery')
    tmults = np.logspace(0,4,9)
    for i in np.arange(Nsets):
        plt.plot(tmults,alltrs[i,:]/tmults,label = setnames[i])
    plt.xscale('log')
    plt.legend()
    plt.ylim(0,60)
    plt.xlim(1,1e4)
    plt.tight_layout()
    plt.savefig(opdir+'all_tmult_repeaters.pdf')
    plt.close()
    
    plt.figure()
    plt.xlabel('time [Catalog 1 = 1]')
    plt.ylabel('Rate of new single bursts')
    tmults = np.logspace(0,4,9)
    for i in np.arange(Nsets):
        plt.plot(tmults,alltss[i,:]/tmults,label = setnames[i])
    plt.xscale('log')
    plt.legend()
    #plt.ylim(0,60)
    plt.xlim(1,1e4)
    plt.tight_layout()
    plt.savefig(opdir+'all_tmult_singles.pdf')
    plt.close()
    
    plt.figure()
    plt.xlabel('time [Catalog 1 = 1]')
    plt.ylabel('Rate of busts from repeaters')
    tmults = np.logspace(0,4,9)
    for i in np.arange(Nsets):
        plt.plot(tmults,alltbs[i,:]/tmults,label = setnames[i])
    plt.xscale('log')
    plt.legend()
    #plt.ylim(0,60)
    plt.xlim(1,1e4)
    plt.tight_layout()
    plt.savefig(opdir+'all_tmult_rep_bursts.pdf')
    plt.close()
 
def generate_state(state,Nbin=6,Rmult=1.,mcrs=None,tag=None,tmults=None):
    """
    Defined to test some corrections to the repeating FRBs method
    """
    # old implementation
    # defines list of surveys to consider, together with Tpoint
    sdir = os.path.join(resource_filename('zdm','../'),'papers/Repeaters/Surveys')
    
    ndm=1400
    nz=500
    
    # holds grids and surveys
    ss = []
    gs = []
    rgs = []
    Cnrs = []
    Cnss =  []
    irs = []
    
    # holds lists of dm and z distributions for predictions
    zrs=[]
    zss=[]
    dmrs=[]
    dmss=[]
    nrs=[]
    nss=[]
    nbs=[]
    
    # holds sums over dm and z
    tdmr = np.zeros([ndm])
    tdmb = np.zeros([ndm])
    tdms = np.zeros([ndm])
    tzr = np.zeros([nz])
    tzs = np.zeros([nz])
    
    # saves number of repeaters
    if tmults is not None:
        tmultlists = np.zeros([tmults.size])
        tmultlistr = np.zeros([tmults.size])
        tmultlistb = np.zeros([tmults.size])
    else:
        tmultlists = None
        tmultlistr = None
        tmultlistb = None
    
    # total number of repeaters and singles
    CNR=0
    CNS=0
    tnr = 0
    tns = 0
    
    bdir='Nbounds'+str(Nbin)+'/'
    beams.beams_path = os.path.join(resource_filename('zdm','../'),'papers/Repeaters/'+bdir)
    bounds = np.load(bdir+'bounds.npy')
    solids = np.load(bdir+'solids.npy')
    
    # for MC distribution
    numbers = np.array([])
    
    Cnreps=np.array([])
    
    # we initialise surveys and grids
    for ibin in np.arange(Nbin):
        # generate basic grids
        name = "CHIME_decbin_"+str(ibin)+"_of_"+str(Nbin)
        s,g = survey_and_grid(survey_name=name,NFRB=None,sdir=sdir,init_state=state)
        
        #ss.append(s)
        #gs.append(g)
        
        ir = np.where(s.frbs['NREP']>1)[0]
        cnr=len(ir)
        irs.append(ir)
        
        i_s = np.where(s.frbs['NREP']==1)[0]
        cns=len(i_s)
        
        CNR += cnr
        CNS += cns
        
        Cnrs.append(cnr)
        Cnss.append(cns)
        
        ###### repeater part ####
        rg = rep.repeat_Grid(g,Tfield=s.TOBS,Nfields=1,MC=False,opdir=None,bmethod=2)
        #rgs.append(rg)
        
        if ibin == 0:
            tot_exact_reps = rg.exact_reps
            tot_exact_singles = rg.exact_singles
            tot_exact_rbursts = rg.exact_rep_bursts
        else:
            tot_exact_reps += rg.exact_reps
            tot_exact_singles += rg.exact_singles
            tot_exact_rbursts += rg.exact_rep_bursts
        
        
        # collapses CHIME dm distribution for repeaters and once-off burts
        zr = np.sum(rg.exact_reps,axis=1)
        zs = np.sum(rg.exact_singles,axis=1)
        dmrb = np.sum(rg.exact_rep_bursts,axis=0)
        dmr = np.sum(rg.exact_reps,axis=0)
        dms = np.sum(rg.exact_singles,axis=0)
        nb = np.sum(dmrb)
        nr = np.sum(dmr)
        ns = np.sum(dms)
        
        
        
        if tmults is not None:
            #tmultlists[0] += ns
            #tmultlistr[0] += nr
            #tmultlistb[0] += nb
            for it,tmult in enumerate(tmults):
                if tmult == 1:
                    tmultlists[it] += ns
                    tmultlistr[it] += nr
                    tmultlistb[it] += nb
                else:
                    
                    thisrg = rep.repeat_Grid(g,Tfield=s.TOBS,Nfields=tmult,MC=False,opdir=None,bmethod=2)
                    thisns = np.sum(thisrg.exact_singles)
                    thisnr = np.sum(thisrg.exact_reps)
                    thisnb = np.sum(thisrg.exact_rep_bursts)
                    tmultlists[it] += thisns
                    tmultlistr[it] += thisnr
                    tmultlistb[it] += thisnb
        
        # adds to running totals
        tzr += zr
        tzs += zs
        tdmb += dmrb
        tdmr += dmr
        tdms += dms
        tnr += nr
        tns += ns
        
        zrs.append(zr)
        zss.append(zs)
        dmrs.append(dmr)
        dmss.append(dms)
        nbs.append(nb)
        nss.append(ns)
        nrs.append(nr)
        rgs.append(rg)
        
        # extract single and repeat DMs, create total list
        nreps=s.frbs['NREP']
        ireps=np.where(nreps>1)
        isingle=np.where(nreps==1)[0]
        if ibin==0:
            alldmr=s.DMEGs[ireps]
            alldms=s.DMEGs[isingle]
            print(type(alldms))
        else:
            alldmr = np.concatenate((alldmr,s.DMEGs[ireps]))
            alldms = np.concatenate((alldms,s.DMEGs[isingle]))
        
        dmvals=g.dmvals
        zvals=g.zvals
        ddm = g.dmvals[1]-g.dmvals[0]
        
        del rg
        if Rmult is None:
            del s
            del g
            continue
        
        ################  MC ##########
        rg = rep.repeat_Grid(g,Tfield=s.TOBS,Nfields=1,opdir=None,bmethod=2,\
                Exact=False,MC=Rmult)
        numbers = np.append(numbers,rg.MC_numbers)
        
        ir = np.where(s.frbs['NREP']>1)[0]
        nr=len(ir)
        irs.append(ir)
        nreps = s.frbs['NREP'][ir]
        Cnreps = np.concatenate((Cnreps,nreps))
        
        del s
        del g
        del rg
        
    
    Chist,bins = np.histogram(Cnreps,bins=mcrs)
    
    
    ############ standard analysis ############3
     
    print("Calculated ",CNS,CNR)
    rnorm = CNS/tns # this is NOT the plotting norm, it is the physical norm
    #ddm = g.dmvals[1]-g.dmvals[0]
    norm = rnorm*200 / ddm
    
    
    nbs = np.array(nbs)
    nrs = np.array(nrs)
    nss = np.array(nss)
    nbs *= rnorm
    nrs *= rnorm
    nss *= rnorm
    
    which = ["Repeaters","Singles","Bursts"]
    tot_exact_reps *= rnorm
    tot_exact_singles *= rnorm
    tot_exact_rbursts *= rnorm
    
    # performs exact plotting
    for ia,array in enumerate([tot_exact_reps,tot_exact_singles,tot_exact_rbursts]):
        name = opdir + 'set_'+tag+'_'+which[ia]+".pdf"
        misc_functions.plot_grid_2(array,zvals,dmvals,
            name=name,norm=3,log=True,label='$\\log_{10} p({\\rm DM}_{\\rm EG},z)$  [a.u.]',
            project=False,Aconts=[0.01,0.1,0.5],zmax=1.5,
            DMmax=1500)
    
    
    bins=np.linspace(0,3000,16)
    
    tdms *= rnorm
    tdmr *= rnorm
    tdmb *= rnorm
    
    tmultlists *= rnorm
    tmultlistr *= rnorm
    tmultlistb *= rnorm
    
    ########## MC analysis #############
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
    
    # values to return
    # Mh: MC histogram
    # fitv: fitted values
    # copyMh: Mh with over-written too-low values
    
    
    return dmvals,tdms,tdmr,tdmb,nss,nrs,nbs,Mh,fitv,copyMh,Chist,Cnss,\
        Cnrs,tmultlists,tmultlistr,tmultlistb # returns singles and total bursts
    
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



def read_extremes(infile='planck_extremes.dat',H0=67.4):
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


main()
