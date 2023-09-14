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
import states as st

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

def main(Nbin=30,FC=1.0):
    Nsets=5
    global opdir
    opdir =  'Rfitting39_'+str(FC)+'/BestFitCalcs/'
    if not os.path.exists(opdir):
        os.mkdir(opdir)
    
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
    states,names=st.get_states()
    rlist=[]
    ndm = 1400
    rlist= np.zeros([Nsets,ndm])
    
    tms = np.zeros([Nsets,Nmults])
    
    NMChist = 100
    mcrs=np.linspace(1.5,0.5+NMChist,NMChist) # from 1.5 to 100.5 inclusive
    
    tmults=np.logspace(-1,3,9)
    #tmults=None
    
    sm=['a','b','c','d']
    for i in np.arange(Nsets):
        
        savefile = opdir+'set_'+str(i)+'_saved_values.npz'
        #if os.path.exists(savefile):
        #    continue
        
        
        infile='Rfitting39_'+str(FC)+'/mc_FC39'+str(FC)+'converge_set_'+str(i)+'_output.npz'
        
        data=np.load(infile)
        
        lps=data['arr_0']
        oldlns=data['arr_1']
        ldms=data['arr_2']
        
        lns=data['arr_3']
        NRs=data['arr_4']
        
        Rmins=data['arr_5']
        Rmaxes=data['arr_6']
        Rgammas=data['arr_7']
        
        xRmax=np.array([-0.25,-0.25,3,3])
        sy=np.array([-1.9,-3.0,-3.0,-2.1])
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
            irg = np.argmin((Rgammas - Rgamma)**2)
            irm = np.argmin((Rmaxes - Rmax)**2)
            Rmin = Rmins[irg,irm]
            
            states[i].rep.Rmin = Rmin
            states[i].rep.Rmax = Rmax
            states[i].rep.Rgamma = Rgamma
            print("Parameters for ",j," are ",Rmin,Rmax,Rgamma)
            
            tempsave = opdir+"temp"+str(j)+".npz"
            print("searching for ",tempsave)
            if j != 0 and os.path.exists(tempsave):
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
                if tmults is not None:
                    tms=data['arr_13']
                    tmr=data['arr_14']
                    tmb=data['arr_15']
            else:
                print("Doing ",j)
                # returns dms, singles,repeaters, bursts, and same for tmult data, all as f(dm)
                dms,ss,rs,bs,nss,nrs,nbs,Mh,fitv,copyMh,Cnreps,Cnss,Cnrs,tms,tmr,tmb = \
                    generate_state(states[i],mcrs=mcrs,tag=str(i)+"_"+sm[j],Rmult=1000.,tmults=tmults,Nbin=Nbin)
            
                np.savez(tempsave,dms,ss,rs,bs,nss,nrs,nbs,Mh,fitv,copyMh,Cnreps,Cnss,Cnrs,tms,tmr,tmb)
                print("Artificially stopped to avoid memory leaks.")
                print("Please keep re-running until this message disappears")
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
            if tmults is not None:
                Tms.append(tms)
                Tmr.append(tmr)
                Tmb.append(tmb)
            
            MCs.append(Mh)
            fMCs.append(fitv)
            cMCs.append(copyMh)
            
        if tmults is not None:
            np.savez(savefile,dms,Fourss,Fourrs,Fourbs,urmax,urmin,urgamma,Fnss,Fnrs,Fnbs,MCs,fMCs,cMCs,\
                Cnreps,Cnss,Cnrs,Tms,Tmr,Tmb)
        else:
            np.savez(savefile,dms,Fourss,Fourrs,Fourbs,urmax,urmin,urgamma,Fnss,Fnrs,Fnbs,MCs,fMCs,cMCs,\
                Cnreps,Cnss,Cnrs)
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
        if tmults is not None:
            Tms=data['arr_16']
            Tmr=data['arr_17']
            Tmb=data['arr_18']
        break
    
    
    ##### sets some labels and styles for all pots #####
    linestyles=['-','--','-.',':']
    labels=['case a','case b','case c','case d']
    #### time evolution plot #####
    
    if tmults is not None:
        plt.figure()
        plt.xlabel('Time $T / T_{\\rm Cat1}$')
        plt.ylabel('Repeaters per time: $N_{\\rm rep} T_{\\rm Cat1} / T$')
        
        plt.plot(tmults,Tmr[0],linestyle=linestyles[0],linewidth=0)
        for i,rs in enumerate(Fourrs):
            # normalisation - calculations only approximately normalsied
            norm = np.sum(Fnrs[i])/16.
            label=labels[i]
            # generates a dummy plot if case a or c
            if i==0 or i==2:
                plt.plot(tmults,Tmr[i]/tmults/norm,linestyle=linestyles[i],linewidth=0)
            else:
                plt.plot(tmults,Tmr[i]/tmults/norm,label=label,linestyle=linestyles[i],linewidth=3)
        plt.legend()
        plt.ylim(0,70)
        plt.xscale('log')
        plt.tight_layout()
        plt.savefig(opdir+'set_0_time_effect.pdf')
        plt.close()
    
    ################### DM distribution plot #################
    
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
    
    plt.hist(Crdms,bins=bins,alpha=1.0,label='CHIME Catalog 1 (16)',edgecolor='black')
    print("Mean DM of CHIME repeaters is ",np.sum(Crdms)/Crdms.size)
    
    labels=['$a: [-1.73,-0.25,-1.9]$','$b: [-1.23,-0.25,~~~-3]$',\
        '$c: [-1.38,\\,~~~~~~~~3,~~~-3]$','$d: [-4.54,\\,~~~~~~~~3,-2.1]$']
    
    plt.xlim(0,1750)
    for i,rs in enumerate(Fourrs):
        label=labels[i]
        print("Total predicted repeaters are ",np.sum(rs))
        print("Model ",i," mean DM of reps",np.sum(rs*dms)/np.sum(rs)," total ", np.sum(rs))
        print("Model ",i," mean DM of singles",np.sum(Fourss[i]*dms)/np.sum(Fourss[i])," total ", np.sum(rs))
        #rs *= np.sum(Fourrs[3])/np.sum(rs)
        # the above line normalises them all to a particular curve
        rs *= Crdms.size/ np.sum(rs)
        plt.plot(dms,rs*scale,label=label,linestyle=linestyles[i],linewidth=3)
        print("Mean predicted DM in model ",label," is ",np.sum(rs*dms)/np.sum(rs))
    
    
    plt.hist(nCrdms,bins=bins,alpha=0.3,label='CHIME Gold sample (25)',edgecolor='black',\
        weights=np.full([25],16./25.))
    
    
    #handles, labels = plt.gca().get_legend_handles_labels()
    #order=[0,5,1,2,3,4]
    #plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order])
    
    plt.legend(title='$~~~~~~~~~~~~[\\log_{10} R_{\\rm min},\\log_{10} R_{\\rm max},R_{\\gamma}]$',title_fontsize=12)
    
    plt.tight_layout()
    plt.savefig(opdir+'set_0_four_cases_dm.pdf')
    plt.close()
    
    ### cumulative version of DM plot
    plt.figure()
    plt.xlim(0,2000)
    plt.xlabel('${\\rm DM}_{\\rm EG}$')
    plt.ylabel('Cumulative $N_{\\rm rep}({\\rm DM}_{\\rm EG}) \\, [200\\,{\\rm pc}\\,{\\rm cm}^{-3}]^{-1}$')
    
    
    sxvals,syvals,rxvals,ryvals=ute.get_chime_rs_dm_histograms(DMhalo=50,newdata=False)
    plt.plot(rxvals,ryvals,label="CHIME Catalog 1 (16)")
    
    for i,rs in enumerate(Fourrs):
        label=labels[i]
        rs = np.cumsum(rs)
        rs /= rs[-1]
        plt.plot(dms,rs,label=label,linestyle=linestyles[i],linewidth=3)
    
    sxvals,syvals,rxvals,ryvals=ute.get_chime_rs_dm_histograms(DMhalo=50,newdata='only')
    plt.plot(rxvals,ryvals,label="CHIME Gold sample (25)")
    
    handles, labels = plt.gca().get_legend_handles_labels()
    order=[0,5,1,2,3,4]
    plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order])
    
    #plt.legend()########
    
    plt.tight_layout()
    plt.savefig(opdir+'cumulative_set_0_four_cases_dm.pdf')
    plt.close()
    
    ############# produce an MC rate plot ###########
    
    #mids=(mcrs[1:] + mcrs[:-1])/2.
    
    vals = np.arange(2,101)
    plt.hist(vals,bins=mcrs,weights=Cnreps,label='CHIME Catalog 1 (16)',alpha=0.5)
    
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
    
    plt.plot(rates,Ccum,label='CHIME Catalog 1 (16)',linewidth=3)
    
    rates = np.linspace(2,100,99)
    markers=['+','s','x','o']
    
    styles=['-','--','-.',':']
    for i,label in enumerate(sm):
        cumMC = np.cumsum(MCs[i])
        plt.plot(rates,cumMC,linestyle=styles[i],label="case "+label,linewidth=3)
        
    plt.legend()
    plt.xscale('log')
    plt.xlabel('$N_{\\rm bursts}$')
    plt.ylabel('$N_{\\rm rep}(N_{\\rm bursts})$')
    
    plt.tight_layout()
    plt.savefig(opdir+'cumMChistogram.pdf')
    plt.close()
    
    
    ######### produce decbin division plot ########
    
    bdir='Nbounds'+str(Nbin)+'/'
    beams.beams_path = os.path.join(resource_filename('zdm','data/BeamData/CHIME/'),bdir)
    bounds = np.load(beams.beams_path+'bounds.npy')
    solids = np.load(beams.beams_path+'solids.npy')
    
    
    plt.figure()
    
    plt.ylim(0,1)
    plt.xlabel('${\\delta}$ [deg]')
    plt.ylabel('$N_{\\rm rep}(\\delta)$')
    
    sx,sy,rx,ry = ute.get_chime_rs_dec_histograms(DMhalo=50)
    nsx,nsy,nrx,nry = ute.get_chime_rs_dec_histograms(DMhalo=50,newdata='only')
    
    plt.plot(rx,ry,label='CHIME Catalog 1 (16)')
    
    for i,label in enumerate(sm):
        
        cumdec = np.cumsum(Fnrs[i])
        cumdec = cumdec/cumdec[-1]
        cumdec = np.concatenate(([0],cumdec))
        plt.plot(bounds,cumdec,linestyle=styles[i],label="case "+label)
    
    
    plt.plot(nrx,nry,label='CHIME Gold sample (25)')
    handles, labels = plt.gca().get_legend_handles_labels()
    order=[0,5,1,2,3,4]
    plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order])
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
    sdir = os.path.join(resource_filename('zdm','data/Surveys/'),'CHIME/')
    
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
    beams.beams_path = os.path.join(resource_filename('zdm','data/BeamData/CHIME/'),bdir)
    bounds = np.load(beams.beams_path+'bounds.npy')
    solids = np.load(beams.beams_path+'solids.npy')
    
    # for MC distribution
    numbers = np.array([])
    
    Cnreps=np.array([])
    
    # we initialise surveys and grids
    for ibin in np.arange(Nbin):
        # generate basic grids
        name = "CHIME_decbin_"+str(ibin)+"_of_"+str(Nbin)
        s,g = ute.survey_and_grid(survey_name=name,NFRB=None,sdir=sdir,init_state=state)
        
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
    
    # All repeater DMs from Cat1
    375.9,102.0818182,158.05,150.4368421,1239.25,62.4,334.4166667,424.65,344.75,508.6,379.75,248.7,608.65,372.7,192.8,520.95
    
    
    # FRB 20180916 https://ui.adsabs.harvard.edu/abs/2020Natur.577..190M/abstract
    #z 0.0337, DMne2001 150.4
    
    
    #FRB 20200120E (NOT in cat 1!) https://arxiv.org/abs/2103.01295
    # z ~ 0.00084 # approx redshift for M81 at 3.6 Mpc
    # DMne2001 = 87.8-40 = 47.8
    
    # FRB 20181030A (in cat 1, 62.4) https://arxiv.org/abs/2108.12122
    # DMne2001: 62.4, z= 0.00385 (20 Mpc)
    
    # Michelli: https://ui.adsabs.harvard.edu/abs/2023ApJ...950..134M/abstract
    # FRB 20180814A z~0.068 DMne2001 = 102
    # FRB 20190303A  z=0.064, DMne2001 = 192.8
    
    # Ibik et al: https://arxiv.org/abs/2304.02638
    #20200223B z=0.06024, DM=202.068-46  {NOT in cat1]
    #20190110C z=0.122244, DM=186.3  [In catalogue 1, NOT a repeater! (just one burst)]
    # 20191106C z=0.10775, DM=333.4-25. [NOT in cat 1, association only approx]
    
    DMhalo=50.
    
    special=[[202.068-46-DMhalo,186.3-DMhalo,333.4-25.-DMhalo],[0.06024,0.122244,0.10775]]
    
    # DMEG for unlocalised repeaters
    repDMonly = np.array([375.9,158.05,1239.25,334.4166667,424.65,344.75,508.6,379.75,248.7,608.65,372.7,520.95])-DMhalo
    # z and DM for localised repeaters
    repzvals = np.array([0.0337,0.00385,0.068,0.064])
    repdmeg = np.array([150.4, 62.4,102,192.8]) - DMhalo
    
    #misc_functions.plot_grid_2(rtot,g.zvals,g.dmvals,
    #        name=opdir+'combined_localised_FRBs_lines.pdf',norm=3,log=True,
    #        label='$\\log_{10} p({\\rm DM}_{\\rm EG},z)$ [a.u.]',
    #        project=False,FRBDM=frbdmvals,FRBZ=frbzvals,Aconts=[0.01,0.1,0.5],
    #        zmax=2.0,DMmax=2000,DMlines=nozlist)
    
    # performs exact plotting
    for ia,array in enumerate([tot_exact_reps,tot_exact_singles,tot_exact_rbursts]):
        name = opdir + 'set_'+tag+'_'+which[ia]+".pdf"
        misc_functions.plot_grid_2(array,zvals,dmvals,
            name=name,norm=3,log=True,label='$\\log_{10} p({\\rm DM}_{\\rm EG},z)$  [a.u.]',
            project=False,Aconts=[0.01,0.1,0.5],zmax=1.5,
            DMmax=1500)
        name = opdir + 'set_'+tag+'_'+which[ia]+"_wFRBs.pdf"
        misc_functions.plot_grid_2(array,zvals,dmvals,
            name=name,norm=3,log=True,label='$\\log_{10} p({\\rm DM}_{\\rm EG},z)$  [a.u.]',
            project=False,Aconts=[0.01,0.1,0.5],zmax=1.5,
            DMmax=1500,FRBDM=repdmeg,FRBZ=repzvals,
            special=special,DMlines=repDMonly)
    
    bins=np.linspace(0,3000,16)
    
    tdms *= rnorm
    tdmr *= rnorm
    tdmb *= rnorm
    
    if tmults is not None:
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
    
    if tmults is not None:
        return dmvals,tdms,tdmr,tdmb,nss,nrs,nbs,Mh,fitv,copyMh,Chist,Cnss,\
            Cnrs,tmultlists,tmultlistr,tmultlistb # returns singles and total bursts
    else:
        return dmvals,tdms,tdmr,tdmb,nss,nrs,nbs,Mh,fitv,copyMh,Chist,Cnss,\
            Cnrs,None,None,None # returns singles and total bursts

main()
