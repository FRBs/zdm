"""

Test the influence of dividing into declination bins using one of the better
fit outputs (or close to it). This is used for the figure in the Appendix.

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
    states,names=st.get_states()
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
    
    # two different numbers of bounds to compare results for
    N1=6
    N2=30
    
    #states[0].rep.Rmin = 9.
    #states[0].rep.Rmax = 10.
    #states[0].rep.Rgamma = -1.1
    #make_decbin_division(99,states[0],N1,N2)
    
    # only does this for a single set
    for iset in [0]:
        # hard-coded
        infile='Rfitting39_1.0/mc_FC391.0converge_set_0_output.npz'
        data=np.load(infile)
        lns=data['arr_3']
        Rmins=data['arr_5']
        Rmaxes=data['arr_6']
        Rgammas=data['arr_7']
        
        bestfit = np.argmax(lns)
        nlogmax = np.max(lns)
        inds = np.unravel_index(bestfit,lns.shape)
        
        Rmax = Rmaxes[inds[1]]
        Rgamma = Rgammas[inds[0]]
        Rmin = Rmins[inds[0],inds[1]]
        print("Using ",Rmin,Rmax,Rgamma)
        
        states[iset].rep.Rmin = Rmin
        states[iset].rep.Rmax = Rmax
        states[iset].rep.Rgamma = Rgamma
        make_decbin_division(iset,states[iset],N1,N2)
        exit()
        # repeats fro strong FRBs
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
    
    
    sxvals,syvals,rxvals,ryvals = ute.get_chime_rs_dec_histograms(newdata=False)
    sxvals2,syvals2,rxvals2,ryvals2 = ute.get_chime_rs_dec_histograms(newdata='only')
    
    ### plots cumulative declination info #####
    
    plt.figure()
    plt.xlim(-11,90)
    plt.ylim(0,1)
    plt.plot(bounds,cum_nss,label='Model singles: $N_\\delta = 6$')
    plt.plot(bounds,cum_nrs,label='Model repeaters: $N_\\delta = 6$',color=plt.gca().lines[-1].get_color(),linestyle='--')
    
    plt.plot(bounds2,cum_nss2,label='Model singles: $N_\\delta = 30$',linestyle='-.')
    plt.plot(bounds2,cum_nrs2,label='Model repeaters: $N_\\delta = 30$',color=plt.gca().lines[-1].get_color(),linestyle=':')
    
    #plt.plot(bounds,cum_Cnrs,label='CHIME repeaters')
    #plt.plot(bounds,cum_Cnss,label='CHIME singles',color=plt.gca().lines[-1].get_color(),linestyle='--')
    
    plt.plot(sxvals,syvals,label='CHIME singles (Cat 1)',color='black',linewidth=2)
    plt.plot(rxvals,ryvals,label='CHIME repeaters (Cat 1)',color=plt.gca().lines[-1].get_color(),linestyle='--',linewidth=2)
    plt.plot(rxvals2,ryvals2,label='CHIME repeaters (Gold25)',color='purple',linestyle='--',linewidth=2)
    
    plt.xlabel('Declination [deg.]')
    plt.ylabel('Cumulative fraction of events')
    plt.legend()
    plt.tight_layout()
    plt.savefig(opdir+'cum_decbin_division_test_state'+str(iset)+'.pdf')
    plt.close()
    
    ### plots cumulative and differential DM info ###
    
    Csdms,Crdms,Csdecs,Crdecs = ute.get_chime_dec_dm_data(DMhalo=50,newdata=False)
    Csdms2,Crdms2,Csdecs2,Crdecs2 = ute.get_chime_dec_dm_data(DMhalo=50,newdata=True)
    
    sxvals,syvals,rxvals,ryvals = ute.get_chime_rs_dm_histograms(newdata=False)
    sxvals,syvals,rxvals2,ryvals2 = ute.get_chime_rs_dm_histograms(newdata='only')
    
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
    
    plt.hist(Csdms,bins=bins,alpha=0.5,label='CHIME single FRBs (Cat1)',edgecolor='black')
    plt.hist(Crdms,bins=bins,alpha=0.5,label='CHIME repeaters (Cat1)',edgecolor='black',weights=np.full([Crdms.size],rmult))
    plt.hist(Crdms2,bins=bins,alpha=0.5,label='CHIME repeaters (Gold25)',edgecolor='black',weights=np.full([Crdms2.size],rm2))
    
    
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
    sdir = os.path.join(resource_filename('zdm','data/Surveys/'),'CHIME/')
    
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
    beams.beams_path = os.path.join(resource_filename('zdm','data/BeamData/CHIME/'),bdir)
    bounds = np.load(beams.beams_path+'bounds.npy')
    solids = np.load(beams.beams_path+'solids.npy')
    
    # we initialise surveys and grids
    for ibin in np.arange(Nbin):
        # generate basic grids
        name = "CHIME_decbin_"+str(ibin)+"_of_"+str(Nbin)
        
        s,g = ute.survey_and_grid(survey_name=name,NFRB=None,sdir=sdir,init_state=state)
        
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
  
main()
