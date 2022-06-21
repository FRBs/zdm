
"""
See README.txt for an explanation
When running, look for do_something=False and do_something=True statements to turn functionalities on/off
Also look for New=True or LOAD=False statements and change these once they have been run; intermediate data
gets saved to Pickle for massive speedups (e.g. if just fine-tuning plots); but it also takes up space!


This file is almost certainly outdated.
"""

import argparse

import numpy as np
import matplotlib
from pkg_resources import resource_filename
import os

import time
from matplotlib.ticker import NullFormatter
from zdm import iteration as it

from zdm import survey
from zdm import cosmology as cos

import pickle

matplotlib.rcParams['image.interpolation'] = None

defaultsize=14
ds=4
font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : defaultsize}
matplotlib.rc('font', **font)

##### definitions of parameters ########
    # pset defined as:
    # [0]:	log10 Emin
    # [1]:	log10 Emax
    # [2]:	alpha (spectrum: nu^alpha)
    # [3]:	gamma
    # [4]:	sfr n
    # [5}: log10 mean host DM
    # [6]: log10 sigma host DM
    
from zdm import misc_functions# import *
#import pcosmic

defaultsize=14
ds=4
font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : defaultsize}
matplotlib.rc('font', **font)

def main():
    print("WARNING: this file is massively out of date, do not use")
    exit()
    ############## Initialise cosmology ##############
    cos.init_dist_measures()
    
    # get the grid of p(DM|z). See function for default values.
    # set new to False once this is already initialised
    zDMgrid, zvals,dmvals,H0=misc_functions.get_zdm_grid(
        new=True,plot=False,method='analytic')
    # NOTE: if this is new, we also need new surveys and grids!
    
    # constants of beam method
    thresh=0
    method=2
    
    
    # sets which kind of source evolution function is being used
    source_evolution=0 # SFR^n scaling
    #source_evolution=1 # (1+z)^(2.7n) scaling
    
    
    # sets the nature of scaling with the 'spectral index' alpha
    alpha_method=0 # spectral index interpretation: includes k-correction. Slower to update
    #alpha_method=1 # rate interpretation: extra factor of (1+z)^alpha in source evolution
    
    ############## Initialise surveys ##############
    
    # constants of intrinsic width distribution
    Wlogmean=1.70267
    Wlogsigma=0.899148
    DMhalo=50
    
    #These surveys combine time-normalised and time-unnormalised samples 
    NewSurveys=True
    #sprefix='Full' # more detailed estimates. Takes more space and time
    sprefix='Std' # faster - fine for max likelihood calculations, not as pretty
    
    if sprefix=='Full':
        Wbins=10
        Wscale=2
        Nbeams=[20,20,20,20]
    elif sprefix=='Std':
        Wbins=5
        Wscale=3.5
        Nbeams=[5,5,5,10]
    
    # location for survey data
    sdir = os.path.join(resource_filename('zdm', 'data'), 'Surveys/')
    if NewSurveys:
        
        print("Generating new surveys, set NewSurveys=False to save time later")
        #load the lat50 survey data
        lat50=survey.survey()
        lat50.process_survey_file(sdir+'CRAFT_class_I_and_II.dat')
        lat50.init_DMEG(DMhalo)
        lat50.init_beam(nbins=Nbeams[0],method=2,plot=False,thresh=thresh) # tells the survey to use the beam file
        pwidths,pprobs=survey.make_widths(lat50,Wlogmean,Wlogsigma,Wbins,scale=Wscale)
        efficiencieslat50=lat50.get_efficiency_from_wlist(dmvals,pwidths,pprobs)
        
        
        # load ICS data
        ICS=survey.survey()
        ICS.process_survey_file(sdir+'CRAFT_ICS.dat')
        ICS.init_DMEG(DMhalo)
        ICS.init_beam(nbins=Nbeams[1],method=2,plot=False,thresh=thresh) # tells the survey to use the beam file
        pwidths,pprobs=survey.make_widths(ICS,Wlogmean,Wlogsigma,Wbins,scale=Wscale)
        efficienciesICS=ICS.get_efficiency_from_wlist(dmvals,pwidths,pprobs)
        
        # load ICS 892 MHz data
        ICS892=survey.survey()
        ICS892.process_survey_file(sdir+'CRAFT_ICS_892.dat')
        ICS892.init_DMEG(DMhalo)
        ICS892.init_beam(nbins=Nbeams[1],method=2,plot=False,thresh=thresh) # tells the survey to use the beam file
        pwidths,pprobs=survey.make_widths(ICS892,Wlogmean,Wlogsigma,Wbins,scale=Wscale)
        efficiencies892=ICS892.get_efficiency_from_wlist(dmvals,pwidths,pprobs)
        
        # load Parkes data
        pks=survey.survey()
        pks.process_survey_file(sdir+'parkes_mb_class_I_and_II.dat')
        pks.init_DMEG(DMhalo)
        pks.init_beam(nbins=Nbeams[2],method=2,plot=False,thresh=thresh) # need more bins for Parkes!
        pwidths,pprobs=survey.make_widths(pks,Wlogmean,Wlogsigma,Wbins,scale=Wscale)
        efficienciesPks=pks.get_efficiency_from_wlist(dmvals,pwidths,pprobs)
        
        
        names=['ASKAP/FE','ASKAP/ICS','Parkes/Mb']
        
        surveys=[lat50,ICS,ICS892,pks]
        if not os.path.isdir('Pickle'):
            os.mkdir('Pickle')
        with open('Pickle/'+sprefix+'surveys.pkl', 'wb') as output:
            pickle.dump(surveys, output, pickle.HIGHEST_PROTOCOL)
            pickle.dump(names, output, pickle.HIGHEST_PROTOCOL)
        
    else:
        with open('Pickle/'+sprefix+'surveys.pkl', 'rb') as infile:
            surveys=pickle.load(infile)
            names=pickle.load(infile)
            lat50=surveys[0]
            ICS=surveys[1]
            ICS892=surveys[2]
            pks=surveys[3]
    print("Initialised surveys ",names)
    
    dirnames=['ASKAP_FE','ASKAP_ICS','Parkes_Mb']
    
    #### these are hard-coded best-fit parameters ####
    # initial parameter values. SHOULD BE LOGSIGMA 0.75! (WAS 0.25!?!?!?)
    # Best-fit parameter values (result from cube iteration)
    lEmin=30. # log10 in erg
    lEmax=41.84 # log10 in erg
    alpha=1.54 # spectral index. WARNING: here F(nu)~nu^-alpha in the code, opposite to the paper!
    gamma=-1.16 # slope of luminosity distribution function
    sfr_n=1.77 #scaling with star-formation rate
    lmean=2.16 # log10 mean of DM host contribution in pc cm^-3
    lsigma=0.51 # log10 sigma of DM host contribution in pc cm^-3
    C=4.19 # log10 constant in number per Gpc^-3 yr^-1 at z=0
    pset=[lEmin,lEmax,alpha,gamma,sfr_n,lmean,lsigma,C,H0]
    
    # This routine takes a *long* time
    # It estimates the difference between a full beam shape (~300 points) and various approximations to it
    # Use 'LOAD=True' if you have already run it
    plot_fbr=False
    if plot_fbr:
        print("Plotting final beam rates")
        #tempnames=['ASKAP/FE','ASKAP/ICS','Parkes/MB']
        #final_plot_beam_values(surveys,zDMgrid,zvals,dmvals,pset,[5,5,10],names,Wlogsigma,Wlogmean,'FinalFitPlots')
        misc_functions.final_plot_beam_rates(surveys,zDMgrid,zvals,dmvals,pset,[5,5,10],names,Wlogsigma,Wlogmean,'Plots',LOAD=False)
    
    
    # This routine is similar to the above
    # It tests different numerical approximations of each beam
    # and estimates various parmeters based on that
    # used only in initial investigations to determine how many points to use
    TestBeams=False
    if TestBeams==True:
        method=2
        zmaxs=[1,2,4]
        DMmaxs=[1000,2000,4000]
        for i,s in enumerate(surveys):
            #test_beam_rates(s,zDMgrid, zvals,dmvals,pset,[0,1,2,5,10,50,100,'all'],method=1)
            outdir=outdir='Plots/'+dirnames[i]+'_BeamTest_'+str(method)+'_'+str(thresh)+'/'
            misc_functions.test_beam_rates(s,zDMgrid, zvals,dmvals,pset,[0,1,2,5,10,50,100,'all'],method=method,outdir=outdir,thresh=thresh,zmax=zmaxs[i],DMmax=DMmaxs[i])
    
    # generates zdm grids for the specified parameter set
    NewGrids=True
    if sprefix=='Full':
        gprefix='best'
    elif sprefix=='Std':
        gprefix='Std_best'
    
    if NewGrids:
        print("Generating new grids, set NewGrids=False to save time later")
        grids=misc_functions.initialise_grids(surveys,zDMgrid, zvals,dmvals,pset,wdist=True,source_evolution=source_evolution,alpha_method=alpha_method)
        with open('Pickle/'+gprefix+'grids.pkl', 'wb') as output:
            pickle.dump(grids, output, pickle.HIGHEST_PROTOCOL)
    else:
        print("Loading grid ",'Pickle/'+gprefix+'grids.pkl')
        with open('Pickle/'+gprefix+'grids.pkl', 'rb') as infile:
            grids=pickle.load(infile)
    glat50=grids[0]
    gICS=grids[1]
    gICS892=grids[2]
    gpks=grids[3]
    print("Initialised grids")
    
    
    Location='Plots'
    if not os.path.isdir(Location):
        os.mkdir(Location)
    prefix='bestfit_'
    
    do2DPlots=True
    if do2DPlots:
        muDM=10**pset[5]
        Macquart=muDM
        # plots zdm distribution
        misc_functions.plot_grid_2(gpks.rates,gpks.zvals,gpks.dmvals,zmax=3,DMmax=3000,
                             name=os.path.join(Location,prefix+'nop_pks_optimised_grid.pdf'),
                             norm=2,log=True,label='$\\log_{10} p({\\rm DM}_{\\rm EG},z)$',project=False,FRBDM=pks.DMEGs,FRBZ=None,Aconts=[0.01,0.1,0.5],Macquart=Macquart)
        misc_functions.plot_grid_2(gICS.rates,gICS.zvals,gICS.dmvals,zmax=1,DMmax=2000,
                             name=os.path.join(Location,prefix+'nop_ICS_optimised_grid.pdf'),
                             norm=2,log=True,label='$\\log_{10} p({\\rm DM}_{\\rm EG},z)$',project=False,FRBDM=ICS.DMEGs,FRBZ=ICS.frbs["Z"],Aconts=[0.01,0.1,0.5],Macquart=Macquart)
        misc_functions.plot_grid_2(gICS892.rates,gICS892.zvals,gICS892.dmvals,zmax=1,DMmax=2000,
                             name=os.path.join(Location,prefix+'nop_ICS892_optimised_grid.pdf'),
                             norm=2,log=True,label='$\\log_{10} p({\\rm DM}_{\\rm EG},z)$',project=False,FRBDM=ICS892.DMEGs,FRBZ=ICS892.frbs["Z"],Aconts=[0.01,0.1,0.5],Macquart=Macquart)
        
        misc_functions.plot_grid_2(glat50.rates,glat50.zvals,glat50.dmvals,zmax=0.6,DMmax=1500,
                             name=os.path.join(Location,prefix+'nop_lat50_optimised_grid.pdf'),
                             norm=2,log=True,label='$\\log_{10} p({\\rm DM}_{\\rm EG},z)$',project=False,FRBDM=lat50.DMEGs,FRBZ=None,Aconts=[0.01,0.1,0.5],Macquart=Macquart)
        
        # plots zdm distribution, including projections onto z and DM axes
        misc_functions.plot_grid_2(gpks.rates,gpks.zvals,gpks.dmvals,zmax=3,DMmax=3000,
                             name=os.path.join(Location,prefix+'pks_optimised_grid.pdf'),
                             norm=2,log=True,label='$\\log_{10} p({\\rm DM}_{\\rm EG},z)$',project=True,FRBDM=pks.DMEGs,FRBZ=None,Aconts=[0.01,0.1,0.5],Macquart=Macquart)
        misc_functions.plot_grid_2(gICS.rates,gICS.zvals,gICS.dmvals,zmax=1,DMmax=2000,
                             name=os.path.join(Location,prefix+'ICS_optimised_grid.pdf'),
                             norm=2,log=True,label='$\\log_{10} p({\\rm DM}_{\\rm EG},z)$',project=True,FRBDM=ICS.DMEGs,FRBZ=ICS.frbs["Z"],Aconts=[0.01,0.1,0.5],Macquart=Macquart)
        misc_functions.plot_grid_2(gICS892.rates,gICS892.zvals,gICS892.dmvals,zmax=1,DMmax=2000,
                             name=os.path.join(Location,prefix+'ICS892_optimised_grid.pdf'),
                             norm=2,log=True,label='$\\log_{10} p({\\rm DM}_{\\rm EG},z)$',project=True,FRBDM=ICS892.DMEGs,FRBZ=ICS892.frbs["Z"],Aconts=[0.01,0.1,0.5],Macquart=Macquart)
        misc_functions.plot_grid_2(glat50.rates,glat50.zvals,glat50.dmvals,zmax=0.5,DMmax=1000,
                             name=os.path.join(Location,prefix+'lat50_optimised_grid.pdf'),
                             norm=2,log=True,label='$\\log_{10} p({\\rm DM}_{\\rm EG},z)$',project=True,FRBDM=lat50.DMEGs,FRBZ=None,Aconts=[0.01,0.1,0.5],Macquart=Macquart)
    
    print("Exiting after generating 2-D plots, remove this exit code to continue")
    exit()
    doMaquart=True
    # generates the Macquart relation for each set
    if doMaquart:
        # generates full imshow dm-z relation
        muDM=10**pset[5]
        Macquart=muDM
        # the badly named variable 'Macquart', if not None, sets the mean host contribution
        misc_functions.make_dm_redshift(glat50,
                                  os.path.join(Location,prefix+'lat50_macquart_relation.pdf'),
                                  DMmax=1000,zmax=0.75,loc='upper right',Macquart=Macquart)
        misc_functions.make_dm_redshift(gICS,
                                  os.path.join(Location,prefix+'ICS_macquart_relation.pdf'),
                                  DMmax=2000,zmax=1,loc='upper right',Macquart=Macquart)
        misc_functions.make_dm_redshift(gpks,
                                  os.path.join(Location,prefix+'pks_macquart_relation.pdf'),
                                  DMmax=4000,zmax=3,loc='upper left',Macquart=Macquart)
    
    
    
    plot_basic_DM_z=True
    if plot_basic_DM_z:
        #uncomment this!
        print("Plotting basic zdm")
        # It is just the intrinsic distribution!
        misc_functions.plot_zdm_basic_paper(grids[0].smear_grid,grids[0].zvals,grids[0].dmvals,zmax=3,DMmax=3000,
                                      name=os.path.join(Location,'dm_EG.pdf'),
                                      norm=1,log=True,ylabel='${\\rm DM_{\\rm EG}}$',label='$\\log_{10} p({\\rm DM_{cosmic}+DM_{host}}|z)$',conts=[0.023, 0.159,0.5,0.841,0.977])
        misc_functions.plot_zdm_basic_paper(grids[0].grid,grids[0].zvals,grids[0].dmvals,zmax=3,DMmax=3000,
                                      name=os.path.join(Location,'dm_cosmic_only.pdf'),
                                      norm=1,log=True,ylabel='${\\rm DM_{\\rm cosmic}}$',label='$\\log_{10} p({\\rm DM_{cosmic}}|z)$',conts=[0.023, 0.159,0.5,0.841,0.977])
    
    
    BasicF0=True
    if BasicF0:
        # here put in simple top-hat beam,
        # this is for a simple demonstration plot with a constant threshold of 1 Jy ms regardless of other effects
        sens=np.zeros([grids[0].dmvals.size])
        sens[:]=1. # relative sensitivity
        Emin=10**pset[0]
        Emax=10**pset[1]
        alpha=pset[2]
        gamma=pset[3]
        muDM=10**pset[5]
        F0=1
        grids[0].calc_thresholds(F0,sens,alpha=alpha)
        
        grids[0].calc_pdv(Emin,Emax,gamma,beam_b=np.array([1]),beam_o=np.array([1]))
        grids[0].calc_rates()
        misc_functions.plot_grid_2(grids[0].rates,grids[0].zvals,grids[0].dmvals,zmax=4,DMmax=5000,norm=0,log=True,name='Plots/basic_F0.pdf',label='$\\log_{10}p({\\rm DM},z) {\\rm [a.u.]}$',project=False,conts=False,FRBZ=None,FRBDM=None,Aconts=[0.01,0.1,0.5],Macquart=muDM)
    
    do_width_test=False
    if do_width_test:
        # Step 1: guess reasonable FRB parameters
        # Step 2: fit_width_test: get best-fit width distribution
        # Step 3: copy the output to cube.py, use to optimise other parameters
        # Step 4: Update best-fit FRB params here, and run width_test to see if there is still a good fit
        # You might also re-run fit_width_test to see how much the best-fit width distribution has changed
        # this first part fits the distribution from https://ui.adsabs.harvard.edu/abs/2021MNRAS.501.5319A/abstract
        # this was made prior to papers by Hao Qiu and Cherie Day, but we find estimates are mostly insensitive
        # to fine details in this distribution. But check it anyway!
        
        # uncomment this if you wish - takes a while though!
        #fit_width_test(pset,surveys,grids,names) # finds something close to log is 1.7026718749999896, 0.8991484374999986
        
        misc_functions.width_test(pset,surveys,grids,names,logmean=1.70267,logsigma=0.899148,outdir='FinalFitPlots',NP=Wbins,scale=Wscale)
    
    exit()
    
    ######### performs likelihood maximisation ##########
    # this was an early attempt to perform a custom likelihood maximisation using standard 'downhill' methods
    # It did not perform well; standard python libraries failed, and my custom routines were slow
    # However, the code is left here as a pointer to anybody wanting to re-try this
    # this is not guaranteed to produce anything sensible at all!
    new_maximise=False
    if new_maximise == True:
        
        # old=True method uses scipy routines, old=False uses my routines
        old=False
        
        if not old:
            
            t0=time.process_time()
            print("Beginning custom maximisation both at once")
            #PenTypes=[0,0,1,0,0,0,0]
            #PenParams=[None,None,[1.6,0.3],None,None,None,None]
            C_ll,C_p=it.my_minimise(pset,grids,surveys,disable=[0,2,4,5,6],psnr=False,PenTypes=None,PenParams=None)
            t1=time.process_time()
            print("My iteration took",t1-t0," seconds")
            print("Results: ",C_ll,C_pbestfit_ICS_macquart_relation.pdf)
            
        
        if old:
            it.update_grid(grids[0],pset)
            t0=time.process_time()
            print("Beginning scipy maximisation")
            lat50_bf=it.maximise_likelihood(grids[0],lat50) # lat50 only
            t1=time.process_time()
            print("Computer took",t1-t0," seconds with some set to zero")
            print("Results: ",lat50_bf)
            
            ICS_bf=it.maximise_likelihood(grids[1],ICS)
            C_bf=it.maximise_likelihood(grids,[lat50,ICS])
            
            lat50_p=lat50_bf["x"]
            ICS_p=ICS_bf["x"]
            C_p=C_bf["x"]
        
        np.save(outdir+'C_p.npy',C_p) # the parameter set that gets maximised
        np.save(outdir+'C_ll.npy',C_ll) #the value of the log-likelihood
        
        print("Best fit parameters for combined sample are ")
        print(C_ll,C_p)
    

main()
