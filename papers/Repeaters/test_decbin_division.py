"""

Test the influence of dividing into declination bins using one of the better
fit outputs

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

global opdir
opdir = 'BestfitCalculations/unrestrained'
if not os.path.exists(opdir):
    os.mkdir(opdir)

def main():
    Nsets=5
    global opdir
    dims=['$R_{\\rm min}$','$R_{\\rm max}$','$\\gamma_{r}$']
    pdims=['$p(R_{\\rm min})$','$p(R_{\\rm max})$','$p(\\gamma_{r})$']
    names=['Rmin','Rmax','Rgamma']
    string=['Ptot','Pn','Pdm']
    pstrings=['$P_{\\rm tot}$','$P_{N}$','$P_{\\rm DM}$']
    
    tmults = np.logspace(0.5,4,8)
    ntm = tmults.size+1
    
    setnames = ['$\\epsilon_C: {\\rm min}\\, \\alpha$','$\\epsilon_z:$ Best fit','${\\rm min}\\, E_{\\rm max}$',
        '${\\rm max} \\,\\gamma$','${\\rm min} \\, \\sigma_{\\rm host}$']
    
    
    # gets the possible states for evaluation
    states,names=get_states()
    rlist=[]
    ndm = 1400
    rlist= np.zeros([Nsets,ndm])
    
    tms = np.zeros([Nsets,ntm])
    
    savefile = opdir+'_saved_values.npz'
    if os.path.exists(savefile):
        load = True
    else:
        load=False
    load = False
    
    
    #states[i].rep.Rmin = 1e-4
    #states[i].rep.Rmax = 1.
    #states[i].rep.Rgamma = -1.7
    # two different numbers of bounds to compare results for
    N1=6
    N2=30
    
    #states[0].rep.Rmin = 9.
    #states[0].rep.Rmax = 10.
    #states[0].rep.Rgamma = -1.1
    #make_decbin_division(99,states[0],N1,N2)
    
    for iset in np.arange(Nsets):
        infile='Rfitting/set_'+str(iset)+'_iteration_output.npz'
        data=np.load(infile)
        lns=data['arr_3']
        Rmins=data['arr_5']
        Rmaxes=data['arr_6']
        Rgammas=data['arr_7']
        
        bestfit = np.argmax(lns)
        nlogmax = np.max(lns)
        inds = np.unravel_index(bestfit,lns.shape)
        
        Rmin = Rmins[inds[0]]
        Rmax = Rmaxes[inds[1]]
        Rgamma = Rgammas[inds[2]]
        states[iset].rep.Rmin = Rmin
        states[iset].rep.Rmax = Rmax
        states[iset].rep.Rgamma = Rgamma
        make_decbin_division(iset,states[iset],N1,N2)
        
        states[iset].rep.Rmin = 9.
        states[iset].rep.Rmax = 10.
        states[iset].rep.Rgamma = -1.1
        
        make_decbin_division(iset+99,states[iset],N1,N2)
    
def make_decbin_division(iset,state,N1,N2,opdir='DecbinTests/'):
    """
    Makes a plot showing dm, declination, and redshift distributions of repeaters
    """ 
    
    
    # return values:
    # dms: vector of dms
    # rs: repeaters per dm
    # crs: CHIME repeaters per dm
    # singles: single bursts per dm
    # css: CHIME singles per dm
    # nrs: number of repeaters per decbin
    # nss: number of singles in decbins
    # bounds: dec bounds of bins
    # solids: solid angles of each bin
    sf1 = 'DecbinTests/Nbins1_outputs_state'+str(iset)+'.npz'
    if os.path.exists(sf1):
        dic=np.load(sf1)
        dms=dic['dms']
        rs=dic['rs']
        crs=dic['crs']
        singles=dic['singles']
        css=dic['css']
        nrs=dic['nrs']
        nss=dic['nss']
        bounds=dic['bounds']
        solids=dic['solids']
        Cnrs=dic['Cnrs']
        Cnss=dic['Cnss']
        tzs=dic['tzs']
        tzr=dic['tzr']
    else:
        
        dms,rs,crs,singles,css,nrs,nss,bounds,solids,Cnrs,Cnss,tzs,tzr = generate_state(state,N1)
        np.savez(sf1,dms=dms,rs=rs,crs=crs,
            singles=singles,css=css,nrs=nrs,nss=nss,bounds=bounds,
            solids=solids,Cnrs=Cnrs,Cnss=Cnss,tzs=tzs,tzr=tzr)
    
    sf2 = 'DecbinTests/Nbins2_outputs_state'+str(iset)+'.npz'
    if os.path.exists(sf2):
        
        dic2=np.load(sf2)
        dms2=dic2['dms2']
        rs2=dic2['rs2']
        crs2=dic2['crs2']
        singles2=dic2['singles2']
        css2=dic2['css2']
        nrs2=dic2['nrs2']
        nss2=dic2['nss2']
        bounds2=dic2['bounds2']
        solids2=dic2['solids2']
        Cnrs2=dic2['Cnrs2']
        Cnss2=dic2['Cnss2']
        tzs2=dic2['tzs2']
        tzr2=dic2['tzr2']
    else:
        print(sf2,"MEOWWWW")
        dms2,rs2,crs2,singles2,css2,nrs2,nss2,bounds2,solids2,Cnrs2,Cnss2,tzs2,tzr2 = generate_state(state,N2)
        
        np.savez(sf2,dms2=dms2,rs2=rs2,crs2=crs2,
            singles2=singles2,css2=css2,nrs2=nrs2,nss2=nss2,bounds2=bounds2,
            solids2=solids2,Cnrs2=Cnrs2,Cnss2=Cnss2,tzs2=tzs2,tzr2=tzr2)
    
    
    
    cum_nrs=np.zeros([bounds.size])
    cum_nss=np.zeros([bounds.size])
    cum_nrs2=np.zeros([bounds2.size])
    cum_nss2=np.zeros([bounds2.size])
    cum_Cnrs=np.zeros([bounds.size])
    cum_Cnss=np.zeros([bounds.size])
    
    cum_nrs[1:] = np.cumsum(nrs)
    cum_nss[1:] = np.cumsum(nss)
    cum_nrs2[1:] = np.cumsum(nrs2)
    cum_nss2[1:] = np.cumsum(nss2)
    cum_Cnrs[1:] = np.cumsum(Cnrs)
    cum_Cnss[1:] = np.cumsum(Cnss)
    
    cum_nrs /= cum_nrs[-1]
    cum_nss /= cum_nss[-1]
    cum_nrs2 /= cum_nrs2[-1]
    cum_nss2 /= cum_nss2[-1]
    cum_Cnrs /= cum_Cnrs[-1]
    cum_Cnss /= cum_Cnss[-1]
    
    
    sxvals,syvals,rxvals,ryvals = ute.get_chime_rs_dec_histograms()
    sxvals,syvals,rxvals2,ryvals2 = ute.get_chime_rs_dec_histograms(newdata=True)
    
    ### plots cumulative declination info #####
    
    plt.figure()
    plt.xlim(-11,90)
    plt.ylim(0,1)
    plt.plot(bounds,cum_nss,label='All singles: $N_\\delta = 6$')
    plt.plot(bounds,cum_nrs,label='All repeaters: $N_\\delta = 6$',color=plt.gca().lines[-1].get_color(),linestyle='--')
    
    plt.plot(bounds2,cum_nss2,label='All singles: $N_\\delta = 30$')
    plt.plot(bounds2,cum_nrs2,label='All repeaters: $N_\\delta = 30$',color=plt.gca().lines[-1].get_color(),linestyle='--')
    
    #plt.plot(bounds,cum_Cnrs,label='CHIME repeaters')
    #plt.plot(bounds,cum_Cnss,label='CHIME singles',color=plt.gca().lines[-1].get_color(),linestyle='--')
    
    plt.plot(sxvals,syvals,label='CHIME singles (cat 1)',color='black',linewidth=2)
    plt.plot(rxvals,ryvals,label='CHIME repeaters (cat 1)',color=plt.gca().lines[-1].get_color(),linestyle='--',linewidth=2)
    plt.plot(rxvals2,ryvals2,label='CHIME repeaters (3 yr)',color='purple',linestyle='--',linewidth=2)
    
    plt.xlabel('Declination (degrees)')
    plt.ylabel('Number of events')
    plt.legend()
    plt.tight_layout()
    plt.savefig(opdir+'cum_decbin_division_test_state'+str(iset)+'.pdf')
    plt.close()
    
    ### plots cumulative and differential DM info ###
    
    Csdms,Crdms,Csdecs,Crdecs = ute.get_chime_dec_dm_data(DMhalo=50,newdata=False)
    Csdms2,Crdms2,Csdecs2,Crdecs2 = ute.get_chime_dec_dm_data(DMhalo=50,newdata=True)
    
    sxvals,syvals,rxvals,ryvals = ute.get_chime_rs_dm_histograms(newdata=False)
    sxvals,syvals,rxvals2,ryvals2 = ute.get_chime_rs_dm_histograms(newdata=True)
    
    cum_singles = np.cumsum(singles)
    cum_singles /= cum_singles[-1]
    cum_singles2 = np.cumsum(singles2)
    cum_singles2 /= cum_singles2[-1]
    cum_rs = np.cumsum(rs)
    cum_rs /= cum_rs[-1]
    cum_rs2 = np.cumsum(rs2)
    cum_rs2 /= cum_rs2[-1]
    
    bins=np.linspace(0,2000,21)
    norm = 100 /(dms[1]-dms[0])
    
    plt.figure()
    ax1=plt.gca()
    rmult=10.
    rm2 = rmult * Crdms.size/Crdms2.size
    
    ax2=plt.gca().twinx()
    
    plt.ylim(0,7.5)
    plt.ylabel('$N_{r}({\\rm DM}_{\\rm EG}) \\, [200\\,{\\rm pc}\\,{\\rm cm}^{-3}]^{-1}$')
    
    plt.sca(ax1)
    plt.xlim(0,2000)
    plt.ylim(0,75)
    plt.xlabel('${\\rm DM}_{\\rm EG}$')
    plt.ylabel('$N_s({\\rm DM}_{\\rm EG}) \\, [200\\,{\\rm pc}\\,{\\rm cm}^{-3}]^{-1}$')
    
    plt.hist(Csdms,bins=bins,alpha=0.5,label='CHIME single FRBs (cat1)',edgecolor='black')
    plt.hist(Crdms,bins=bins,alpha=0.5,label='CHIME repeaters (cat1)',edgecolor='black',weights=np.full([Crdms.size],rmult))
    plt.hist(Crdms2,bins=bins,alpha=0.5,label='CHIME repeaters (3 yr)',edgecolor='black',weights=np.full([Crdms2.size],rm2))
    
    
    plt.plot(dms,singles*norm,label='Predicted singles: $N_\\delta=6$')
    plt.plot(dms,singles2*norm,label='Predicted singles:  $N_\\delta=30$',linestyle=':')
    plt.plot(dms,rs*norm*rmult,label='Predicted repeaters:  $N_\\delta=6$')
    plt.plot(dms,rs2*norm*rmult,label='Predicted repeaters:  $N_\\delta=30$',linestyle=':')
    
    plt.legend()
    plt.tight_layout()
    plt.savefig(opdir+'dm_division_test_state'+str(iset)+'.pdf')
    plt.close()
    
    plt.figure()
    plt.ylim(0,1)
    plt.xlim(0,2000)
    plt.xlabel('${\\rm DM}_{\\rm EG}$')
    plt.ylabel('$N({\\rm DM}_{\\rm EG}) \\, [200\\,{\\rm pc}\\,{\\rm cm}^{-3}]^{-1}$')
    
    plt.plot(sxvals,syvals,label='CHIME singles',color='black',linewidth=2)
    plt.plot(rxvals,ryvals,label='CHIME repeaters',color=plt.gca().lines[-1].get_color(),linestyle='--',linewidth=2)
    
    plt.plot(rxvals2,ryvals2,label='CHIME repeaters (new data)',color='purple',linestyle='--',linewidth=2)
    
    
    plt.plot(dms,cum_singles,label='Predicted singles: 6 decbins')
    plt.plot(dms,cum_singles2,label='Predicted singles: 30 decbins',linestyle=':')
    plt.plot(dms,cum_rs,label='Predicted repeaters: 6 decbins')
    plt.plot(dms,cum_rs2,label='Predicted repeaters: 30 decbins',linestyle=':')
    
    plt.legend()
    plt.tight_layout()
    plt.savefig(opdir+'cum_dm_division_test_state'+str(iset)+'.pdf')
    plt.close()
    
    
    # now does cumulative plot
    
    

 
def generate_state(state,Nbin,plot=False,tag=None,tmults=None):
    """
    Defined to test some corrections to the repeating FRBs method
    """
    # old implementation
    # defines list of surveys to consider, together with Tpoint
    sdir = os.path.join(resource_filename('zdm','../'),'papers/Repeaters/Surveys')
    
    ndm=1400
    nz=500
    
    #Nbin=6
    #bounds=np.array([-11,30,60,70,80,85,90])
    
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
    
    # holds sums over dm and z
    tdmr = np.zeros([ndm])
    tdms = np.zeros([ndm])
    tzr = np.zeros([nz])
    tzs = np.zeros([nz])
    
    # saves number of repeaters
    if tmults is not None:
        tmultlist = np.zeros([tmults.size+1])
    
    # total number of repeaters and singles
    CNR=0
    CNS=0
    tnr = 0
    tns = 0
    
    # loads bin info to create histogram of single and repeater rates
    
    bdir='Nbounds'+str(Nbin)+'/'
    beams.beams_path = os.path.join(resource_filename('zdm','../'),'papers/Repeaters/'+bdir)
    
    # loads declinations
    bounds = np.load(bdir+'bounds.npy')
    solids = np.load(bdir+'solids.npy')
    
    # we initialise surveys and grids
    for ibin in np.arange(Nbin):
        # generate basic grids
        name = "CHIME_decbin_"+str(ibin)+"_of_"+str(Nbin)
        
        s,g = survey_and_grid(survey_name=name,NFRB=None,sdir=sdir,init_state=state)
        
        ss.append(s)
        gs.append(g)
        
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
        
        rgs.append(rg)
        
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
        dmr = np.sum(rg.exact_reps,axis=0)
        dms = np.sum(rg.exact_singles,axis=0)
        nr = np.sum(dmr)
        ns = np.sum(dms)
        
        if tmults is not None:
            tmultlist[0] += nr
            for it,tmult in enumerate(tmults):
                print("For bin ",it," doing tmult ",tmult)
                thisrg = rep.repeat_Grid(g,Tfield=s.TOBS,Nfields=tmult,MC=False,opdir=None,bmethod=2)
                thisnr = np.sum(thisrg.exact_reps)
                tmultlist[it+1] += thisnr
        
        # adds to running totals
        tzr += zr
        tzs += zs
        tdmr += dmr
        tdms += dms
        tnr += nr
        tns += ns
        
        zrs.append(zr)
        zss.append(zs)
        dmrs.append(dmr)
        dmss.append(dms)
        nrs.append(nr)
        nss.append(ns)
        rgs.append(rg)
        
        # extract single and repeat DMs, create total list
        nreps=s.frbs['NREP']
        ireps=np.where(nreps>1)
        isingle=np.where(nreps==1)[0]
        # creates list of CHIME FRBs at each declination
        if ibin==0:
            alldmr=s.DMEGs[ireps]
            alldms=s.DMEGs[isingle]
        else:
            alldmr = np.concatenate((alldmr,s.DMEGs[ireps]))
            alldms = np.concatenate((alldms,s.DMEGs[isingle]))
    
    rnorm = CNS/tns # this is NOT the plotting norm, it is the physical norm
    ddm = g.dmvals[1]-g.dmvals[0]
    norm = rnorm*200 / ddm
    
    nrs = np.array(nrs)
    nss = np.array(nss)
    nrs *= rnorm
    nss *= rnorm
    
    which = ["Repeaters","Singles","Bursts"]
    tot_exact_reps *= rnorm
    tot_exact_singles *= rnorm
    tot_exact_rbursts *= rnorm
    
    Cnrs = np.array(Cnrs)
    Cnss = np.array(Cnss)
    
    #bins=np.linspace(0,3000,16)
    
    if plot:
        # performs exact plotting
        for ia,array in enumerate([tot_exact_reps,tot_exact_singles,tot_exact_rbursts]):
            name = opdir + 'set_'+tag+'_'+which[ia]+".pdf"
            misc_functions.plot_grid_2(array,g.zvals,g.dmvals,
                name=name,norm=3,log=True,label='$\\log_{10} p({\\rm DM}_{\\rm EG},z)$  [a.u.]',
                project=True,Aconts=[0.01,0.1,0.5],zmax=1.5,
                DMmax=1500)
        
        
        fig=plt.figure()
        plt.xlim(0,3000)
        plt.xlabel('${\\rm DM}_{\\rm EG}$')
        plt.ylabel('$N({\\rm DM}_{\\rm EG}) \\, [200\\,{\\rm pc}\\,{\\rm cm}^{-3}]^{-1}$')
        # generate histograms of single and repeat FRBs observed by CHIME
        plt.hist(alldms,bins=bins,alpha=0.5,label='CHIME: single bursts',edgecolor='black')
        plt.hist(alldmr,bins=bins,alpha=0.5,label='repeating sources',edgecolor='black')
        
        plt.plot(g.dmvals,tdms*norm,label='Distributed: single bursts',linestyle="-",linewidth=2)
        plt.plot(g.dmvals,tdmr*norm,label='repeating sources',linestyle="-.",linewidth=2)
        
        legend=plt.legend()
        renderer = fig.canvas.get_renderer()
        max_shift = max([t.get_window_extent(renderer).width for t in legend.get_texts()])
        for t in legend.get_texts():
            temp_shift = max_shift - t.get_window_extent().width
            t.set_position((temp_shift*0.7,0))
        
        plt.tight_layout()
        plt.savefig(opdir+tag+'.pdf')
        plt.close()
        # we now generate some plots!
    
    # First, we plot the total single and repeat distributions against actual data
    
    
    print("For this parameter set, we initially estimated ",tns," single bursts")
    print("    This gives us a normalisation factor of ",rnorm)
    print("    However, the total number of repeat bursts was ",tnr)
    print("    This gets renormalised to ",tnr*rnorm," compared to ",CNR," observed")
    
    return g.dmvals,tdmr*rnorm,alldmr,tdms*rnorm,alldms,nrs,nss,bounds,solids,Cnrs,Cnss,tzs*rnorm,tzr*rnorm
     
    
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
    
    # Return Survey and Grid
    return isurvey, grids[0]

def get_chime_data(DMhalo=50):
    """
    Imports data from CHIME catalog 1
    """
    chimedir = 'CHIME_FRBs/'
    infile = chimedir+'chimefrbcat1.csv'
    
    idec=6
    idm=18
    idmeg=26
    iname=0
    irep=2
    iwidth=42
    isnr=17
    
    NFRB=600
    decs=np.zeros([NFRB])
    dms=np.zeros([NFRB])
    dmegs=np.zeros([NFRB])
    dmgs=np.zeros([NFRB])
    snrs=np.zeros([NFRB])
    widths=np.zeros([NFRB])
    names=[]
    reps=np.zeros([NFRB])
    
    # holds repeater info
    rnames=[]
    ireps=[]
    nreps=[]
    badcount=0
    
    with open(infile) as f:
        lines = f.readlines()
        count=-1
        for i,line in enumerate(lines):
            if count==-1:
                columns=line.split(',')
                #for ic,w in enumerate(columns):
                #    print(ic,w)
                count += 1
                continue
            words=line.split(',')
            # seems to indicate new bursts have been added
            #if words[5][:2]=="RA":
            #    badcount += 1
                #print("BAD : ",badcount)
                #continue
            decs[i-1]=float(words[idec])
            dms[i-1]=float(words[idm])
            dmegs[i-1]=float(words[idmeg])
            names.append(words[iname])
            snrs[i-1]=float(words[isnr])
            # guards against upper limits
            if words[iwidth][0]=='<':
                widths[i-1]=0.
            else:
                widths[i-1]=float(words[iwidth])*1e3 #in ms
            dmgs[i-1] = dms[i-1]-dmegs[i-1]
            rep=words[irep]
            
            
            if rep=='-9999':
                reps[i-1]=0
            else:
                reps[i-1]=1
                if rep in rnames:
                    ir = rnames.index(rep)
                    nreps[ir] += 1
                else:
                    rnames.append(rep)
                    ireps.append(i-1)
                    nreps.append(1)
            count += 1
    print("Total of ",len(rnames)," repeating FRBs found")
    print("Total of ",len(np.where(reps==0)[0])," once-off FRBs")
    
    # sorts by declination
    singles = np.where(reps==0)
    sdecs = decs[singles]
    reps = np.where(reps>0)
    rdecs = decs[reps]
    
    sdecs = np.sort(sdecs)
    rdecs = np.sort(rdecs)
    
    ns = sdecs.size
    nr = rdecs.size
    
    #creates cumulative hist
    sxvals = np.zeros([ns*2+2])
    rxvals = np.zeros([nr*2+2])
    syvals = np.zeros([ns*2+2])
    ryvals = np.zeros([nr*2+2])
    for i,dec in enumerate(sdecs):
        sxvals[i*2+1]=dec
        sxvals[i*2+2]=dec
        syvals[i*2+1]=i/ns
        syvals[i*2+2]=(i+1)/ns
    syvals[-1]=1.
    sxvals[-1]=90
    
    for i,dec in enumerate(rdecs):
        rxvals[i*2+1]=dec
        rxvals[i*2+2]=dec
        ryvals[i*2+1]=i/nr
        ryvals[i*2+2]=(i+1)/nr
    ryvals[-1]=1.
    rxvals[-1]=90
    
    return sxvals,syvals,rxvals,ryvals
    
    
    
main()
