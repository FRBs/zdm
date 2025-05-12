""" 
This script creates example plots for a combination
of FRB surveys and repeat bursts

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
from zdm import beams
import pickle
import numpy as np
from zdm import survey
from matplotlib import pyplot as plt

import scipy as sp

import matplotlib
import time
import utilities as ute

matplotlib.rcParams['image.interpolation'] = None

defaultsize=14
ds=4
font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : defaultsize}
matplotlib.rc('font', **font)

def main():
    #repetition parameters
    #Rset={"Rmin":1e-4,"Rmax":10,"Rgamma":-2.2}
    
    #defines lists of repeater properties, in order Rmin,Rmax,r
    # units of Rmin and Rmax are "per day above 10^39 erg"
    Rset={"Rmin":3.9,"Rmax":4,"Rgamma":-1.1}
    
    # defines list of surveys to consider, together with Tpoint
    sdir = os.path.join(resource_filename('zdm','data/Surveys/'),'CHIME/')
    
    surveys = ["CHIME_beff","CHIME_30_fbar","CHIME_allb10"] #"CHIME_allb","CHIME_150_fbar",
    names = ["$T_{\\rm eff}$","$T({\\overline{B}})$","$\\overline{T}(B)$"]
    
    #special beams path for this unique beam data
    beams.beams_path = os.path.join(resource_filename('zdm','../paper/Repetition/'),'BeamData/')
    
    # in case you wish to switch to another output directory
    opdir='Beamtests/'
    if not os.path.exists(opdir):
        os.mkdir(opdir)
    
    # for overall loops
    rgs=[]
    
    state = set_state()
    
    # intiial test for 10 days on field. This is WRONG however.
    # should be set to time per dt (in days) interval multiplied by number of days
    # the CHIME 'dt' could/should be the primary beamwidth
    Tfield=10
    gs=[]
    rgs=[]
    load=[True,True,True]
    #load=[True,True,True,True,False]
    #load=[False,False,False]#,False,False]
    if any(load):
        with open(opdir+'repgrids.pkl', 'rb') as infile:
            gs=pickle.load(infile)
            rgs=pickle.load(infile)
    else:
        gs=[]
        rgs=[]
    for i,name in enumerate(surveys):
        if load[i]:
            continue
        # initialises survey and grid
        t0=time.time()
        s,g = ute.survey_and_grid(survey_name=name,NFRB=None,sdir=sdir,
            init_state=state) # should be equal to actual number of FRBs, but for this purpose it doesn't matter
        
        # calculates repetition info
        g.state.rep.Rmin = Rset["Rmin"]
        g.state.rep.Rmax = Rset["Rmax"]
        g.state.rep.Rgamma = Rset["Rgamma"]
        
        gs.append(g)
        #gs[i]=g
        # adds repeating grid
        t1=time.time()
        print("Grid ",i," took ",t1-t0," seconds to initialise")
        rg = rep.repeat_Grid(g,Tfield=Tfield,Nfields=1,MC=False,opdir=None,bmethod=1)
        
        t2=time.time()
        print("Repeat Grid ",i," took ",t2-t1," seconds to initialise")
        rgs.append(rg)
        #rgs[i]=rg
        #plot_CHIME_dm(g.rates,g.dmvals)
        
    #if load:
    #    with open(opdir+'repgrids.pkl', 'rb') as infile:
    #        gs=pickle.load(infile)
    #        rgs=pickle.load(infile)
    #else:
    if not all(load):
        with open(opdir+'repgrids.pkl', 'wb') as output:
            pickle.dump(gs, output, pickle.HIGHEST_PROTOCOL)
            pickle.dump(rgs, output, pickle.HIGHEST_PROTOCOL)
    
        #if i==1:
        #    break
    
    time_effect_for_beams(rgs,names,xmax=3,ymax=420,suffix="",Tfield=1)
    
    plot_CHIME_dm(gs)
    
    DMmax=2500
    zmax=2.5
    
    for i,name in enumerate(surveys):
        # Cosmology
        if i==0 and all(load):
            cos.set_cosmology(state)
            cos.init_dist_measures()
        g=gs[i]
        rg=rgs[i]
        name=surveys[i]
        misc_functions.plot_grid_2(g.rates,g.zvals,g.dmvals,
            name=opdir+name+'_grid.pdf',norm=3,log=True,label='$\\log_{10} p({\\rm DM}_{\\rm EG},z)$  [a.u.]',
            project=True,Aconts=[0.01,0.1,0.5],zmax=zmax,
            DMmax=DMmax)#,DMlines=s.DMEGs[s.nozlist])
        misc_functions.plot_grid_2(rg.exact_singles,g.zvals,g.dmvals,
            name=opdir+name+'_single_grid.pdf',norm=3,log=True,label='$\\log_{10} p({\\rm DM}_{\\rm EG},z)$  [a.u.]',
            project=True,Aconts=[0.01,0.1,0.5],zmax=zmax,
            DMmax=DMmax)
        misc_functions.plot_grid_2(rg.exact_reps,g.zvals,g.dmvals,
            name=opdir+name+'_reps_grid.pdf',norm=3,log=True,label='$\\log_{10} p({\\rm DM}_{\\rm EG},z)$  [a.u.]',
            project=True,Aconts=[0.01,0.1,0.5],zmax=zmax,
            DMmax=DMmax)

def get_chime_data(DMhalo=50):
    """
    Imports data from CHIME catalog 1
    """
    import numpy as np
    # read in the CHIME FRB data
    chime_single_dmeg=np.zeros([474])
    names=[]
    with open("CHIME_FRBs/single_dm_eg.dat") as f:
        lines=f.readlines()
        for i,l in enumerate(lines):
            data=l.split()
            names.append(data[0])
            dm_eg=float(data[1])
            dm_eg -= DMhalo # subtracts assumed halo contribution from Macquart et al fit
            chime_single_dmeg[i]=dm_eg
            
    chime_all_dmeg=np.zeros([536])
    chime_first_reps=[]
    replist=[]
    rep_bursts=[]
    Nrepeaters=18
    nreps = np.zeros([Nrepeaters])
    with open("CHIME_FRBs/chime_all_ne.dat") as f:
        lines=f.readlines()
        j=0
        for i,l in enumerate(lines):
            data=l.split()
            # record all bursts regardless
            chime_all_dmeg[i] = float(data[1])
    
    with open("CHIME_FRBs/repeat_ne.dat") as f:
        lines=f.readlines()
        for i,l in enumerate(lines):
            data=l.split()
            if data[1] in replist:
                # we already have this repeater
                irep = replist.index(data[1])
                nreps[irep] += 1
                
                rep_bursts.append(float(data[2]))
            else:
                # it is first burst of repeater
                replist.append(data[1])
                chime_first_reps.append(float(data[2])-50.)
    chime_first_reps = np.array(chime_first_reps)
    rep_bursts = np.array(rep_bursts)
    return names,chime_all_dmeg,chime_single_dmeg,chime_first_reps,rep_bursts,nreps


def plot_CHIME_dm(grids):
    """
    Plots the p(DM) of the CHIME grid
    """
    #names,decs,dms,dmegs,snrs,reps,ireps,widths = ute.get_chime_data(DMhalo=DMhalo)
    #names,chime_all_dmeg,chime_single_dmeg,chime_first_reps,rep_bursts,nreps = ute.get_chime_data()
    
    chime_single_dmeg,chime_first_reps,sdecs,rdecs,nreps = ute.get_chme_dec_dm_data(donreps=True)
    
    bins=np.linspace(0,4000,41)
    hsingle,bins = np.histogram(chime_single_dmeg,bins=bins)
    hreps,bins = np.histogram(chime_first_reps,bins=bins)
    hbursts,bins = np.histogram(chime_first_reps,bins=bins,weights=nreps)
    binwidth = bins[1]-bins[0]
    NFRB = chime_all_dmeg.size # hard-coded
    
    linestyles = ["-","--",":","-.","-"]
    
    plt.figure()
    plt.xlim(0,4000)
    plt.ylim(0,80)
    plt.xlabel('${\\rm DM}_{\\rm EG}$ [pc cm$^{-3}$]')
    plt.ylabel('$p({\\rm DM}_{\\rm EG})$')
    plt.xlim(0,4000)
    # want proj to have units of "FRBs per bin"
    # to get this, first make it a probability distribution, such that \int proj ddm = 1
    for ig,grid in enumerate(grids):
        proj = np.sum(grid.rates,axis=0)
        # normalises to a probability distribution
        proj /= np.sum(proj)
        proj /= (grid.dmvals[1]-grid.dmvals[0])
        # multiplies by a bin width of latter histograms
        proj *= binwidth
        # distribution for NFRBs
        proj *= NFRB
        #proj *=
        #norm = binwidth * NFRB / np.sum(proj)
        plt.plot(grid.dmvals,proj,linestyle = linestyles[ig])
    
    plt.hist(bins[:-1],bins,weights=hsingle,label="Once-off FRBs",alpha=0.5)
    plt.hist(bins[:-1],bins,weights=hreps,bottom=hsingle,label="Repeaters",alpha=0.7)
    plt.hist(bins[:-1],bins,weights=hbursts,bottom=hsingle+hreps,label="Repeat bursts",alpha=0.8)
    plt.legend()
    plt.tight_layout()
    plt.savefig('Beamtests/chime_dm_dist.pdf')
    plt.close()
    

def set_state():
    """
    Sets the state parameters
    """
    
    state = loading.set_state(alpha_method=1)
    state_dict = dict(cosmo=dict(fix_Omega_b_h2=True))
    state.energy.luminosity_function = 2 # this is Schechter
    state.update_param_dict(state_dict)
    # changes the beam method to be the "exact" one, otherwise sets up for FRBs
    state.beam.Bmethod=3
    state.width.Wmethod=0
    state.width.Wbias="CHIME"
    
    # updates to most recent best-fit values
    state.cosmo.H0 = 67.4
    state.energy.lEmax = 41.63
    state.energy.gamma = -0.948
    state.energy.alpha = 1.03
    state.FRBdemo.sfr_n = 1.15
    state.host.lmean = 2.22
    state.host.lsigma = 0.57
    return state

def time_effect_for_beams(rlist,names,xmax=None,ymax=None,suffix="",Tfield=1):
    """
    Plots the different results for different ways of using the CHIME beam
    """
    
    plt.figure()
    plt.xlabel('z')
    plt.ylabel('N(z)dz (year$^{-1}$ sr$^{-1}$)')
    if xmax is not None:
        plt.xlim(0,xmax)
    if ymax is not None:
        plt.ylim(0,ymax)
    styles=['-','--','-.',':','-']
    colours=['red','green','blue','orange','purple']
    lw=1.5
    for i,rg in enumerate(rlist): #Here, T is in days: 10,100,1000 days
        print(i,names)
        print("Doing ",names[i])
        dz=rg.zvals[1]-rg.zvals[0]
        
        zproj=np.sum(rg.exact_singles,axis=1)
        #zproj /= np.sum(zproj)
        zproj /= dz
        zproj /= Tfield
        plt.plot(rg.zvals,zproj,label="Single bursts",color=colours[i],linestyle="-",linewidth=lw)
        
        # prints the peak in this
        imax=np.argmax(zproj)
        zmax=rg.zvals[imax]
        print("Redshift of single burst peak is ",zmax)
        
        zproj=np.sum(rg.exact_reps,axis=1)
        #zproj /= np.sum(zproj)
        zproj /= dz
        zproj /= Tfield
        plt.plot(rg.zvals,zproj,label="Repeating sources",color=colours[i],linestyle=":",linewidth=lw)
        
        # prints the peak in this
        imax=np.argmax(zproj)
        zmax=rg.zvals[imax]
        print("Refshift of repeating sources is ",zmax)
             
        zproj=np.sum(rg.exact_rep_bursts,axis=1)
        #zproj /= np.sum(zproj)
        zproj /= dz
        zproj /= Tfield
        plt.plot(rg.zvals,zproj,label="Bursts from repeaters",color=colours[i],linestyle="-.",linewidth=lw)
        
        total=rg.exact_singles + rg.exact_reps
        zproj=np.sum(total,axis=1)
        #zproj /= np.sum(zproj)
        zproj /= dz
        zproj /= Tfield
        plt.plot(rg.zvals,zproj,label="Total progenitors",color=colours[i],linestyle="--",linewidth=lw)
        
        total=rg.exact_singles + rg.exact_rep_bursts
        zproj=np.sum(total,axis=1)
        #zproj /= np.sum(zproj)
        zproj /= dz
        zproj /= Tfield
        plt.plot(rg.zvals,zproj,label="Total bursts",color=colours[i],linestyle="-",linewidth=3)
        if i==0:
            plt.legend()
        
        # prints the peak in this
        imax=np.argmax(zproj)
        zmax=rg.zvals[imax]
        #print("Redfshift of all burst peak is ",zmax)
        
    plt.tight_layout()
    plt.savefig("Beamtests/beam_comparison.pdf")
    plt.close()
    
    
    
    tx=1.0
    ty=370
    ###### now creates multiple plots - total bursts ######
    plt.figure()
    for i,rg in enumerate(rlist): #Here, T is in days: 10,100,1000 days
        total=rg.exact_singles + rg.exact_rep_bursts
        zproj=np.sum(total,axis=1)
        #zproj /= np.sum(zproj)
        zproj /= dz
        if i==4:
            mult=1.0
        else:
            mult=1.0
        plt.plot(rg.zvals,zproj*mult,label=names[i],color=colours[i],linestyle=styles[i],linewidth=2)
    plt.legend()
    plt.text(tx,ty,"Total bursts")
    set_plot()
    plt.savefig("Beamtests/beam_comparison_total_bursts.pdf")
    plt.close()
    
    ###### now creates multiple plots - total progenitors ######
    plt.figure()
    for i,rg in enumerate(rlist): #Here, T is in days: 10,100,1000 days
        total=rg.exact_singles + rg.exact_reps
        zproj=np.sum(total,axis=1)
        #zproj /= np.sum(zproj)
        zproj /= dz
        plt.plot(rg.zvals,zproj,label=names[i],color=colours[i],linestyle=styles[i],linewidth=2)
    
    set_plot()
    plt.text(tx,ty,"Progenitors")
    plt.savefig("Beamtests/beam_comparison_total_progenitors.pdf")
    plt.close()
    
    ###### now creates multiple plots - repeating FRBs ######
    plt.figure()
    for i,rg in enumerate(rlist): #Here, T is in days: 10,100,1000 days
        total=rg.exact_reps
        zproj=np.sum(total,axis=1)
        #zproj /= np.sum(zproj)
        zproj /= dz
        plt.plot(rg.zvals,zproj,label=names[i],color=colours[i],linestyle=styles[i],linewidth=2)
    plt.text(tx,ty,"Repeaters")
    set_plot()
    plt.savefig("Beamtests/beam_comparison_repeaters.pdf")
    plt.close()
    
    ###### now creates multiple plots - bursts from repeaters ######
    plt.figure()
    for i,rg in enumerate(rlist): #Here, T is in days: 10,100,1000 days
        total=rg.exact_rep_bursts
        zproj=np.sum(total,axis=1)
        #zproj /= np.sum(zproj)
        zproj /= dz
        plt.plot(rg.zvals,zproj,label=names[i],color=colours[i],linestyle=styles[i],linewidth=2)
    plt.text(tx,ty,"Repeat bursts")
    set_plot()
    plt.savefig("Beamtests/beam_comparison_total_rep_bursts.pdf")
    plt.close()
    
    ###### now creates multiple plots - single bursts ######
    plt.figure()
    for i,rg in enumerate(rlist): #Here, T is in days: 10,100,1000 days
        total=rg.exact_singles
        zproj=np.sum(total,axis=1)
        zproj /= dz
        plt.plot(rg.zvals,zproj,label=names[i],color=colours[i],linestyle=styles[i],linewidth=2)
    plt.text(tx,ty,"Single bursts")
    set_plot()
    plt.savefig("Beamtests/beam_comparison_single_bursts.pdf")
    plt.close()
    
def set_plot():
    
    plt.xlabel('z')
    plt.ylabel('N(z)dz (year$^{-1}$ sr$^{-1}$)')
    plt.xlim(0,3)
    plt.ylim(0,420)
    #plt.legend()
    plt.tight_layout()

main()
