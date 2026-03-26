""" 
This script creates example plots for a combination
of FRB surveys and repeat bursts.

It is used for two figures from the paper.

Firstly, in "Preliminary" results, we show the effects
of ASKAP ICS observations.

Secondly, in "Future Prospects", we analyse expectations
for z and DM of ASKAP and FAST.

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

matplotlib.rcParams['image.interpolation'] = None

defaultsize=14
ds=4
font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : defaultsize}
matplotlib.rc('font', **font)

def main():
    
    # gets the possible states for evaluation
    states,names=st.get_states(ischime=False)
    state=states[0]
    
    
    #defines lists of repeater properties, in order Rmin,Rmax,r
    # units of Rmin and Rmax are "per day above 10^39 erg"
    
    # "strong repeaters" scenario
    Rset1={"Rmin":3.9,"Rmax":4,"Rgamma":-1.1}
    # "distributed repeaters" scenario
    Rset2={"Rmin":1e-4,"Rmax":10,"Rgamma":-2.2}
    # case b
    Rset3={"Rmin":0.056,"Rmax":0.56,"Rgamma":-2.999}
    # case d
    Rset4={"Rmin":2.88e-5,"Rmax":1000,"Rgamma":-2.0999}
    #sets=[Rset1,Rset2,Rset3,Rset4]
    sets=[Rset3,Rset4] # only those sets *after* fitting to CHIME
    
    # defines list of surveys to consider, together with Tpoint
    sdir = os.path.join(resource_filename('zdm','data/'),'Surveys/')
    
    ###### generates the example plots for CRAFT ICS, using multiple times #######
    
    
    # in case you wish to switch to another output directory
    opdir='ExamplePlots/'
    if not os.path.exists(opdir):
        os.mkdir(opdir)
    time_effect_for_survey("CRAFT_ICS",sdir,Rset2,suffix="_set2",xmax=1.2,ymax=0.35,label='(a)')#,xmax=1,ymax=2,suffix="_set2")
    time_effect_for_survey("CRAFT_ICS",sdir,Rset1,suffix="_set1",xmax=1.2,ymax=0.35,label='(b)')
    
    ############### Initial plots -Figs 1 and 2 ##################
    NMC=100
    generate_MC_plots("CRAFT_ICS",sdir,100.,Rset1,NMC,load=True)
    
    ############### Future Predictions Plots - Figs 19 ##################
    opdir = 'Predictions/'
    if not os.path.exists(opdir):
        os.mkdir(opdir)
    
    surveys = ["CRAFT_class_I_and_II","CRAFT_ICS","CRAFT_CRACO_MC_frbs_alpha1_5000","parkes_mb_class_I_and_II","FAST"]
    names = ["ASKAP/FE","ASKAP/ICS","ASKAP/CRACO","Parkes/Mb","FAST"]
    fnames = ["ASKAP_FE","ASKAP_ICS","ASKAP_CRACO","Parkes_Mb","FAST"]
    Tfe = [1338.9/24.]# units of days, 3/24. is lowest
    Tics = [879.1/24.] # from kibana logs of on-sky time, likely CRAFT filler
    Tcraco = [800./24.] # plans for DINGO; could be x2 due to two different bands
    
    # Parkes times, representing HTRU
    Tparkes = [30./24.]  #270/3600./24., is HTRU
    
    # FAST times: representing a single scan, and monitoring of 121102
    Tfast = [59.5/24.] #13./3600./24. is typical drift scan
    
    # all the times!!!
    times = [Tfe,Tics,Tcraco,Tparkes,Tfast]
    zmaxes=[0.5,1,1.5,2,3]
    dmmaxes=[1000,2000,3000,4000,5000]
    FAST_scale = 19/13 * (64/300)**2 # scaled from Parkes: 19 beams not 13, but each beam is smaller
    scales=[1.,1.,1.,1.,FAST_scale]
    
    
    ########### loops over grids, generating plots of repetition effects in each case ############
    rgs=[]
    for i,sname in enumerate(surveys):
        
        name=names[i]
        rgs.append([])
        
        s,g = loading.survey_and_grid(survey_name=sname,NFRB=None,sdir=sdir,init_state=state) # should be equal to actual number of FRBs, but for this purpose it doesn't matter
        
        # updates to latest parameter set - does NOT update repetition parameters
        #update_state(g)
        
        if s.TOBS is not None:
            rate_check(g,s,name)
        
        for iT,Tfield in enumerate(times[i]):
            rgs[i].append([])
            
            for iR,Rset in enumerate(sets):
                
                #plot_dir = opdir + name + "_" + str(iT) + "_" + str(iR) + "/"
                Tfield=100.
                t0=time.time()
                rg=calc_reps(s,g,Tfield,Rparams=Rset,Nfields=1)#,opdir=plot_dir)
                t1=time.time()
                #rg.calc_Rthresh(doplots=False,Exact=False,MC=True) #Why do a Monte Carlo here?
                #t2=time.time()
                
                rgs[i][iT].append(rg)
                # uncomment the following to plot these in 2D zDM space
                #misc_functions.plot_grid_2(rg.exact_singles,g.zvals,g.dmvals,
                #    name='TEMP'+str(iR)+'.pdf',norm=0,log=True,label='$\\log_{10} p({\\rm DM}_{\\rm EG},z)$  [a.u.]',
                #    project=False,zmax=4,
                #    DMmax=4000)#,DMlines=s.DMEGs[s.nozlist])
                
            opfile = opdir+fnames[i]+"z.pdf"
            plot_survey(rgs[i][iT],g,opfile,name,xmax=zmaxes[i],scale=scales[i])
            
            opfile = opdir+fnames[i]+"z_norm.pdf"
            plot_survey(rgs[i][iT],g,opfile,name,xmax=zmaxes[i],norm=True,scale=scales[i])
            
            opfile = opdir+fnames[i]+"dm.pdf"
            plot_survey(rgs[i][iT],g,opfile,name,xmax=dmmaxes[i], axis=0,scale=scales[i])
            
            opfile = opdir+fnames[i]+"dm_norm.pdf"
            plot_survey(rgs[i][iT],g,opfile,name,xmax=dmmaxes[i],norm=True,axis=0,scale=scales[i])
            
            
    
    
def plot_survey(rgs,grid,opfile,name,xmax=None,ymax=None,norm=False,axis=1,scale=1.,printit=False):
    """
    for CRAFT ICS, generates figures 2
    
    
    axis=1: N(z), axis=0: N(DM)
    """
    
    plt.figure()
    if axis == 1:
        plt.xlabel('z')
        if norm:
            plt.ylabel('N(z) [a.u.]')
        else:
            plt.ylabel('N(z)')
        xvals = grid.zvals
        
    else:
        plt.xlabel('DM')
        if norm:
            plt.ylabel('N(DM) [a.u.]')
        else:
            plt.ylabel('N(DM) [1000 pc cm$^{-3}$]$^{-1}$')
        xvals = grid.dmvals
    
    dz = xvals[1]-xvals[0]
    
    styles=['-','--','-.',':']
    lw=1.5
    
    colours=((230./256.,97/256.,0),(93/256.,58/256.,155/256.))
    
    suffixes=[' (b)',' (d)']
    
    for i,rg in enumerate(rgs): #Here, T is in days: 10,100,1000 days
        suffix = suffixes[i]
        
        ### calculates scaling
        if norm:
            total=rg.exact_singles + rg.exact_rep_bursts
            zproj=np.sum(total,axis=axis)
            #zproj /= np.sum(zproj)
            zproj /= dz
            themax=np.max(zproj)
        else:
            if axis == 1:
                themax = scale**-1
            else:
                themax = scale**-1/1000. # so it's "per 1000 DM"
            
        zproj=np.sum(rg.exact_singles,axis=axis)
        #zproj /= np.sum(zproj)
        zproj /= dz
        #zproj /= Tfield
        if i==0:
            label="Single bursts" + suffix
        else:
            label=suffix
        
        plt.plot(xvals,zproj/themax,label=label,color=colours[i],linestyle="-",linewidth=lw)
        
        # prints the peak in this
        imax=np.argmax(zproj)
        zmax=rg.zvals[imax]
        #print("Redshift of single burst peak is ",zmax)
        
        zproj=np.sum(rg.exact_reps,axis=axis)
        #zproj /= np.sum(zproj)
        zproj /= dz
        #zproj /= Tfield
        if i==0:
            label="Repeating sources" + suffix
        else:
            label=suffix
        plt.plot(xvals,zproj/themax,label=label,color=colours[i],linestyle=":",linewidth=lw)
        
        # prints the peak in this
        imax=np.argmax(zproj)
        zmax=rg.zvals[imax]
        #print("Redshift of repeating sources is ",zmax)
             
        zproj=np.sum(rg.exact_rep_bursts,axis=axis)
        #zproj /= np.sum(zproj)
        zproj /= dz
        #zproj /= Tfield
        if i==0:
            label="Bursts from repeaters" + suffix
        else:
            label=suffix
        plt.plot(xvals,zproj/themax,label=label,color=colours[i],linestyle="-.",linewidth=lw)
        
        total=rg.exact_singles + rg.exact_reps
        zproj=np.sum(total,axis=axis)
        #zproj /= np.sum(zproj)
        zproj /= dz
        #zproj /= Tfield
        if i==0:
            label="Total progenitors" + suffix
        else:
            label=suffix
        plt.plot(xvals,zproj/themax,label=label,color=colours[i],linestyle="--",linewidth=lw)
        
        #continue
        total=rg.exact_singles + rg.exact_rep_bursts
        zproj=np.sum(total,axis=axis)
        #zproj /= np.sum(zproj)
        zproj /= dz
        #zproj /= Tfield
        if i==0:
            label="Total bursts"
            plt.plot(xvals,zproj/themax,label=label,color="black",linestyle="-",linewidth=3)
        #plt.legend(title=name)
        
        # prints the peak in this
        #imax=np.argmax(zproj)
        #zmax=rg.zvals[imax]
        #print("Redfshift of all burst peak is ",zmax)
        
        singles = np.sum(rg.exact_singles)*scale
        repeaters = np.sum(rg.exact_reps)*scale
        totals = np.sum(rg.exact_singles + rg.exact_rep_bursts)*scale
        Fs = singles/totals
        Fr = repeaters/totals
        Fb = Fs+Fr
        # information to print
        if printit:
            print(name,i,"Info: ",Fs,Fr,Fb,Fr/Fb,singles,repeaters,singles+repeaters,totals)
        
    ### ADD CONSTANT AND TIME - check that everything is working
    #total = grid.rates * rg.Tfield * 10**grid.state.FRBdemo.lC
    #zproj=np.sum(total,axis=axis)
    #zproj /= dz
    #zproj /= themax
    #plt.plot(xvals,zproj,label="Total bursts v2",color="black",linestyle="-",linewidth=3)
    
    plt.legend(ncol=2,fontsize=12)
    plt.text(0.68,0.57,name,transform = plt.gca().transAxes,fontsize=16)
    if xmax is not None:
        plt.xlim(0,xmax)
    
    if ymax is not None:
        plt.ylim(0,ymax)
    elif norm:
        plt.ylim(0,1.)
    else:
        if axis==1:
            plt.ylim(0.,np.max(zproj)*1.03)
        else:
            plt.ylim(0.,np.max(zproj)*1.03*1000*scale)
    
    plt.tight_layout()
    plt.savefig(opfile)
    plt.close()
    
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
    plt.plot(zbar,std/means**0.5,linewidth=3,linestyle='--',label='$\\sigma \\left<N_{\\rm bursts}\\right>^{-0.5}$')
    #plt.plot(zbar,ll90,linestyle=':',color=plt.gca().lines[-1].get_color())
    plt.xlim(0,1.2)
    plt.ylim(0,12)
    plt.ylabel('$N_{\\rm bursts}$')
    plt.legend()
    plt.tight_layout()
    plt.savefig('ExamplePlots/MC_all_bursts.pdf')
    plt.close()
    
    
     
def time_effect_for_survey(name,sdir,Rset,xmax=None,ymax=None,suffix="",label=''):
    """
    for CRAFT ICS, generates figures 2
    """
    s,g = loading.survey_and_grid(survey_name=name,NFRB=None,sdir=sdir) # should be equal to actual number of FRBs, but for this purpose it doesn't matter
    # updates to latest parameter set - does NOT update repetition parameters
    update_state(g)
    
    plt.figure()
    plt.xlabel('z')
    plt.ylabel('N(z) [day$^{-1}$]')
    plt.text(0.01,0.33,label)
    if xmax is not None:
        plt.xlim(0,xmax)
    if ymax is not None:
        plt.ylim(0,ymax)
    styles=['-','--','-.',':']
    #colours=['red','green','blue','orange','purple']
    lw=1.5
    for i,Tfield in enumerate([10,100,1000]): #Here, T is in days: 10,100,1000 days
        rg=calc_reps(s,g,Tfield,Rparams=Rset,Nfields=1,opdir=None)
        
        dz=rg.zvals[1]-rg.zvals[0]
        
        zproj=np.sum(rg.exact_singles,axis=1)
        #zproj /= np.sum(zproj)
        zproj /= dz
        zproj /= Tfield
        plt.plot(rg.zvals,zproj,label="Single bursts",linestyle="-",linewidth=lw)#color=colours[i],
        
        # prints the peak in this
        imax=np.argmax(zproj)
        zmax=rg.zvals[imax]
        print("Redshift of single burst peak is ",zmax)
        
        zproj=np.sum(rg.exact_reps,axis=1)
        #zproj /= np.sum(zproj)
        zproj /= dz
        zproj /= Tfield
        plt.plot(rg.zvals,zproj,label="Repeating sources",linestyle=":",linewidth=lw,color=plt.gca().lines[-1].get_color())#color=colours[i],
        
        # prints the peak in this
        imax=np.argmax(zproj)
        zmax=rg.zvals[imax]
        print("Redshift of repeating sources is ",zmax)
             
        zproj=np.sum(rg.exact_rep_bursts,axis=1)
        #zproj /= np.sum(zproj)
        zproj /= dz
        zproj /= Tfield
        plt.plot(rg.zvals,zproj,label="Bursts from repeaters",linestyle="-.",linewidth=lw,color=plt.gca().lines[-1].get_color())#color=colours[i],
        
        total=rg.exact_singles + rg.exact_reps
        zproj=np.sum(total,axis=1)
        #zproj /= np.sum(zproj)
        zproj /= dz
        zproj /= Tfield
        plt.plot(rg.zvals,zproj,label="Total progenitors",linestyle="--",linewidth=lw,color=plt.gca().lines[-1].get_color())#color=colours[i],
        
        total=rg.exact_singles + rg.exact_rep_bursts
        zproj=np.sum(total,axis=1)
        #zproj /= np.sum(zproj)
        zproj /= dz
        zproj /= Tfield
        plt.plot(rg.zvals,zproj,label="Total bursts",linestyle="-",linewidth=3,color='black')
        if i==0:
            plt.legend()
        
        # prints the peak in this
        imax=np.argmax(zproj)
        zmax=rg.zvals[imax]
        print("Redfshift of all burst peak is ",zmax)
        
    plt.tight_layout()
    plt.savefig('ExamplePlots/'+name+"_time_effect"+suffix+".pdf")
    plt.close()
    
def time_effect_for_survey2(name,sdir,Rset):
    
    s,g = loading.survey_and_grid(survey_name=name,NFRB=None,sdir=sdir) # should be equal to actual number of FRBs, but for this purpose it doesn't matter
    # updates to latest parameter set - does NOT update repetition parameters
    update_state(g)
    
    plt.figure()
    plt.xlabel('z')
    plt.ylabel('N(z) [day$^{-1}$]')
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
        plt.plot(rg.zvals,zproj,label="Singles",linestyle=styles[i],linewidth=2)#color='red',
        
        zproj=np.sum(rg.exact_reps,axis=1)
        #zproj /= np.sum(zproj)
        zproj /= dz
        zproj /= Tfield
        plt.plot(rg.zvals,zproj,label="Repeaters",linestyle=styles[i],linewidth=2)#color='purple',
                    
        zproj=np.sum(rg.exact_rep_bursts,axis=1)
        #zproj /= np.sum(zproj)
        zproj /= dz
        zproj /= Tfield
        plt.plot(rg.zvals,zproj,label="Bursts from repeaters",linestyle=styles[i],linewidth=2) #color='green',
        
        total=rg.exact_singles + rg.exact_reps
        zproj=np.sum(total,axis=1)
        #zproj /= np.sum(zproj)
        zproj /= dz
        zproj /= Tfield
        plt.plot(rg.zvals,zproj,label="Total progenitors",linestyle=styles[i],linewidth=2)#color='orange',
        
        total=rg.exact_singles + rg.exact_rep_bursts
        zproj=np.sum(total,axis=1)
        #zproj /= np.sum(zproj)
        zproj /= dz
        zproj /= Tfield
        plt.plot(rg.zvals,zproj,label="Total bursts",linestyle=styles[i],linewidth=3)#color='black'
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
    print("Expected number for ",name," is ",expected_number) # units: per day, convert to per year! Factor of 365
    
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
    rg = rep.repeat_Grid(g,Tfield=Tfield,Nfields=Nfields,MC=False,opdir=None,verbose=False)
    
    return rg #returns the repeat grid object for further plotting fun!
    ############# do 2D plots ##########
    #misc_functions.plot_grid_2(g.rates,g.zvals,g.dmvals,
    #    name=opdir+name+'.pdf',norm=3,log=True,label='$\\log_{10} p({\\rm DM}_{\\rm EG},z)$  [a.u.]',
    #    project=False,FRBDM=s.DMEGs,FRBZ=s.frbs["Z"],Aconts=[0.01,0.1,0.5],zmax=1.5,
    #    DMmax=1500)#,DMlines=s.DMEGs[s.nozlist])


main()
