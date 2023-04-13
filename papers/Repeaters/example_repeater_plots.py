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
from zdm.craco import loading
from zdm import io
from zdm import repeat_grid as rep

import numpy as np
from zdm import survey
from matplotlib import pyplot as plt

import scipy as sp

import matplotlib
import time

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
    Rset2={"Rmin":1e-4,"Rmax":10,"Rgamma":-2.2}
    Rset1={"Rmin":3.9,"Rmax":4,"Rgamma":-1.1}
    sets=[Rset1,Rset2]
    
    # defines list of surveys to consider, together with Tpoint
    sdir = os.path.join(resource_filename('zdm', 'data'), 'Surveys')
    
    surveys = ["CRAFT_class_I_and_II","parkes_mb_class_I_and_II","FAST"]
    Taskap = [3./24.,1338.9/24.]# units of days
    Tparkes = [270/3600./24.,30./24.]
    Tfast = [13./3600./24.,59.5/24.]
    times = [Taskap,Tparkes,Tfast]
    
    # in case you wish to switch to another output directory
    opdir='ExamplePlots/'
    if not os.path.exists(opdir):
        os.mkdir(opdir)
    
    # for overall loops
    rgs=[]
    for i,name in enumerate(surveys):
        continue
        rgs.append([])
        s,g = loading.survey_and_grid(survey_name=name,NFRB=None,sdir=sdir) # should be equal to actual number of FRBs, but for this purpose it doesn't matter
        
        # updates to latest parameter set - does NOT update repetition parameters
        update_state(g)
        
        rate_check(g,s,name)
        
        for iT,Tfield in enumerate(times[i]):
            rgs[i].append([])
            for iR,Rset in enumerate(sets):
                print("Survey ",name," time ",Tfield," Rparams ",Rset)
                plot_dir = opdir + name + "_" + str(iT) + "_" + str(iR) + "/"
                Tfield=100.
                t0=time.time()
                rg=calc_reps(s,g,Tfield,Rparams=Rset,Nfields=1,opdir=plot_dir)
                t1=time.time()
                rg.calc_Rthresh(doplots=False,Exact=False,MC=True)
                t2=time.time()
                rgs[i][iT].append(rg)
                exit()
    # now does some accumulated plots
     # for overall loops
    rgs=[]
    #name="CRAFT_class_I_and_II"
    
    NMC=100
    generate_MC_plots("CRAFT_ICS",sdir,100.,Rset1,NMC,load=True)
    
    time_effect_for_survey("CRAFT_ICS",sdir,Rset2,suffix="_set2",xmax=1.2,ymax=0.35)#,xmax=1,ymax=2,suffix="_set2")
    time_effect_for_survey("CRAFT_ICS",sdir,Rset1,suffix="_set1",xmax=1.2,ymax=0.35)
    
def generate_MC_plots(name,sdir,Tfield,Rset,NMC,load=True):
    """
    generate MC example
    """
    
    
    opdir='MC/'
    
    
    if load and os.path.exists(opdir+'zsums.npy'):
        zsums=np.load(opdir+'zsums.npy')
        zsumr=np.load(opdir+'zsumr.npy')
        zsumb=np.load(opdir+'zsumb.npy')
        zvals=np.load(opdir+'zvals.npy')
    else:
        s,g = loading.survey_and_grid(survey_name=name,NFRB=None,sdir=sdir) # should be equal to actual number of FRBs, but for this purpose it doesn't matter
        # updates to latest parameter set - does NOT update repetition parameters
        update_state(g)
        told=time.time()
        zsums=np.zeros([NMC,g.zvals.size])
        zsumr=np.zeros([NMC,g.zvals.size])
        zsumb=np.zeros([NMC,g.zvals.size])
        
        g.state.rep.Rmin = Rset["Rmin"]
        g.state.rep.Rmax = Rset["Rmax"]
        g.state.rep.Rgamma = Rset["Rgamma"]
        
        # adds repeating grid
        rg = rep.repeat_Grid(g,Tfield=Tfield,Nfields=1,opdir="MC/",verbose=True,Exact=False,MC=True)
        
        zsums[0,:] = np.sum(rg.MC_singles,axis=1)
        zsumr[0,:] = np.sum(rg.MC_reps,axis=1)
        zsumb[0,:] = np.sum(rg.MC_rep_bursts,axis=1)
        plt.plot(g.zvals,zsums[0,:])
        tnew = time.time()
        print("Initial time ",0," is ",tnew-told)
        told=tnew
        for i in np.arange(NMC-1):
            rg.calc_Rthresh(doplots=False,Exact=False,MC=True)
            zsums[i+1,:] = np.sum(rg.MC_singles,axis=1)
            zsumr[i+1,:] = np.sum(rg.MC_reps,axis=1)
            zsumb[i+1,:] = np.sum(rg.MC_rep_bursts,axis=1)
            tnew = time.time()
            print("Time ",i+1," is ",tnew-told)
            told=tnew
        np.save(opdir+'zsums.npy',zsums)
        np.save(opdir+'zsumr.npy',zsumr)
        np.save(opdir+'zsumb.npy',zsumb)
        np.save(opdir+'zvals.npy',g.zvals)
        zvals=g.zvals
    
    every=10
    # total bursts
    total = zsums + zsumb
    
    array=total
    
    NMC,nz = array.shape
    ncounts=int(nz/10)
    red_array=np.zeros([NMC,ncounts])
    zbar = np.zeros([ncounts])
    
    
    for i in np.arange(ncounts):
        red_array[:,i] = np.sum(array[:,i*every:(i+1)*every],axis=1)
        zbar[i] = np.mean(zvals[i*every:(i+1)*every])
    
    # does statistics on the array
    means = np.mean(red_array,axis=0)
    std = np.std(red_array,axis=0)
    
    ul90 = np.zeros([ncounts])
    ll90 = np.zeros([ncounts])
    for iz in np.arange(ncounts):
        # calculates percentiles in each bin
        counts=np.sort(red_array[:,iz])
        #print("iz is ",counts)
        #ll90[iz]=(counts[4]+counts[5])/2.
        ul90[iz]=(counts[89]+counts[90])/2.
        
    plt.figure()
    #plt.plot(g.zvals,zsums[i+1,:])
    plt.xlabel('z')
    
    plt.plot(zbar,means,linewidth=3,label='$\\left<N_{\\rm bursts}\\right>$ per bin')
    plt.plot(zbar[1:],ul90[1:],linestyle=':',linewidth=2,label='90% upper limit')
    plt.plot(zbar,std/means**0.5,linewidth=3,linestyle='--',label='$\\sigma \\left<N_{\\rm FRB}\\right>^{-0.5}$')
    #plt.plot(zbar,ll90,linestyle=':',color=plt.gca().lines[-1].get_color())
    plt.xlim(0,1.2)
    plt.ylim(0,12)
    plt.legend()
    plt.tight_layout()
    plt.savefig('PaperPlots/MC_all_bursts.pdf')
    plt.close()
    
    
     
def time_effect_for_survey(name,sdir,Rset,xmax=None,ymax=None,suffix=""):
    """
    for CRAFT ICS, generates figures 2
    """
    s,g = loading.survey_and_grid(survey_name=name,NFRB=None,sdir=sdir) # should be equal to actual number of FRBs, but for this purpose it doesn't matter
    # updates to latest parameter set - does NOT update repetition parameters
    update_state(g)
    
    plt.figure()
    plt.xlabel('z')
    plt.ylabel('N(z)dz (day$^{-1}$)')
    if xmax is not None:
        plt.xlim(0,xmax)
    if ymax is not None:
        plt.ylim(0,ymax)
    styles=['-','--','-.',':']
    colours=['red','green','blue','orange','purple']
    lw=1.5
    for i,Tfield in enumerate([10,100,1000]): #Here, T is in days: 10,100,1000 days
        rg=calc_reps(s,g,Tfield,Rparams=Rset,Nfields=1,opdir=None)
        
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
        print("Redshift of repeating sources is ",zmax)
             
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
        plt.plot(rg.zvals,zproj,label="Total bursts",color="black",linestyle="-",linewidth=3)
        if i==0:
            plt.legend()
        
        # prints the peak in this
        imax=np.argmax(zproj)
        zmax=rg.zvals[imax]
        print("Redfshift of all burst peak is ",zmax)
        
    plt.tight_layout()
    plt.savefig('PaperPlots/'+name+"_time_effect"+suffix+".pdf")
    plt.close()
    
def time_effect_for_survey2(name,sdir,Rset):
    
    s,g = loading.survey_and_grid(survey_name=name,NFRB=None,sdir=sdir) # should be equal to actual number of FRBs, but for this purpose it doesn't matter
    # updates to latest parameter set - does NOT update repetition parameters
    update_state(g)
    
    plt.figure()
    plt.xlabel('z')
    plt.ylabel('N(z)dz (day$^{-1}$)')
    plt.xlim(0,0.4)
    plt.ylim(0,0.07)
    styles=['-','--','-.',':']
    colors=['red','green','blue','orange','purple']
    for i,Tfield in enumerate([10,100,1000]):
        rg=calc_reps(s,g,Tfield,Rparams=Rset1,Nfields=1,opdir=None)
        
        dz=rg.zvals[1]-rg.zvals[0]
        
        zproj=np.sum(rg.exact_singles,axis=1)
        #zproj /= np.sum(zproj)
        zproj /= dz
        zproj /= Tfield
        plt.plot(rg.zvals,zproj,label="Singles",color='red',linestyle=styles[i],linewidth=2)
        
        zproj=np.sum(rg.exact_reps,axis=1)
        #zproj /= np.sum(zproj)
        zproj /= dz
        zproj /= Tfield
        plt.plot(rg.zvals,zproj,label="Repeaters",color='purple',linestyle=styles[i],linewidth=2)
                    
        zproj=np.sum(rg.exact_rep_bursts,axis=1)
        #zproj /= np.sum(zproj)
        zproj /= dz
        zproj /= Tfield
        plt.plot(rg.zvals,zproj,label="Bursts from repeaters",color='green',linestyle=styles[i],linewidth=2)
        
        total=rg.exact_singles + rg.exact_reps
        zproj=np.sum(total,axis=1)
        #zproj /= np.sum(zproj)
        zproj /= dz
        zproj /= Tfield
        plt.plot(rg.zvals,zproj,label="Total progenitors",color='orange',linestyle=styles[i],linewidth=2)
        
        total=rg.exact_singles + rg.exact_rep_bursts
        zproj=np.sum(total,axis=1)
        #zproj /= np.sum(zproj)
        zproj /= dz
        zproj /= Tfield
        plt.plot(rg.zvals,zproj,label="Total bursts",color='black',linestyle=styles[i],linewidth=3)
        if i==0:
            plt.legend()
    plt.tight_layout()
    plt.savefig("ASKAP_flys_eye_comparison.pdf")
    plt.close()
    
                    
def rate_check(g,s,name):
    """
    Outputs basic data on the predicted total rate
    """
    # correction for FAST beams
    if name=="FAST":
        rate_mod = (19./13.)*(64./300.)**2
    else:
        rate_mod = 1.
    
    expected_number = rate_mod*np.sum(g.rates)*s.TOBS*10**g.state.FRBdemo.lC
    print("Expected number is ",expected_number) # units: per day, convert to per year! Factor of 365
    
def update_state(g):    
    # approximate best-fit values from recent analysis
    vparams = {}
    vparams['H0'] = 73
    vparams['lEmax'] = 41.63
    vparams['gamma'] = -0.95
    vparams['alpha'] = 1.03
    vparams['sfr_n'] = 1.15
    vparams['lmean'] = 2.23
    vparams['lsigma'] = 0.57
    vparams['lC'] = 1.963
    vparams['RE0']=1e39
    
    # set up new parameters
    g.update(vparams)
    #state = parameters.State()
    


def calc_reps(s,g,Tfield,Rparams=None,Nfields=1,opdir='Repeaters/'):
    """
    Performs the repeat grid calculation
    
    inputs:
        s: survey class objects
        g: grid class object initialised with s
        Tfield: time per field/pointing
        Rparams: sdict of repetition parameters
            None: g.state already initialised
            Else: dict of Rmin, Rmax, and Rgamma
        Nfields: currently redundant number of pointings
    """
    
    g.state.rep.Rmin = Rparams["Rmin"]
    g.state.rep.Rmax = Rparams["Rmax"]
    g.state.rep.Rgamma = Rparams["Rgamma"]
    
    
    # adds repeating grid
    rg = rep.repeat_Grid(g,Tfield=Tfield,Nfields=Nfields,MC=False,opdir=opdir,verbose=True)
    
    return rg #returns the repeat grid object for further plotting fun!
    ############# do 2D plots ##########
    misc_functions.plot_grid_2(g.rates,g.zvals,g.dmvals,
        name=opdir+name+'.pdf',norm=3,log=True,label='$\\log_{10} p({\\rm DM}_{\\rm EG},z)$  [a.u.]',
        project=False,FRBDM=s.DMEGs,FRBZ=s.frbs["Z"],Aconts=[0.01,0.1,0.5],zmax=1.5,
        DMmax=1500)#,DMlines=s.DMEGs[s.nozlist])


main()
