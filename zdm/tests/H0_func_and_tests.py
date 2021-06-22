# -*- coding: utf-8 -*-
"""
Created on Tue Jun 15 18:53:22 2021

@author: Esanmouli Ghosh
"""
import argparse

import numpy as np
from zdm import zdm
#import pcosmic
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm
import matplotlib
from pkg_resources import resource_filename
import os
import sys

import scipy as sp
import time
from matplotlib.ticker import NullFormatter
from zdm import iteration as it

from zdm import survey
from zdm import cosmology as cos
from zdm import pcosmic
from zdm import beams
from zdm import misc_functions
import pickle


####setting up the initial grid and plotting some stuff####

setH0=67.74
cos.set_cosmology(H0=setH0)
       
# get the grid of p(DM|z)
zDMgrid, zvals,dmvals,H0=misc_functions.get_zdm_grid(H0=setH0,new=True,plot=False,method='analytic')
pcosmic.plot_mean(zvals,'Plots/mean_DM.pdf')
    
Wbins=10
Wscale=2
Nbeams=[20,20,20]  #Full beam NOT Std  
thresh=0
method=2
Wlogmean=1.70267
Wlogsigma=0.899148

sdir = os.path.join(resource_filename('zdm', 'data'), 'Surveys/')
#load the lat50 survey data
lat50=survey.survey()
lat50.process_survey_file(sdir+'CRAFT_class_I_and_II.dat')
DMhalo=50
lat50.init_DMEG(DMhalo)
lat50.init_beam(nbins=Nbeams[0],method=2,plot=False,thresh=thresh) # tells the survey to use the beam file
pwidths,pprobs=survey.make_widths(lat50,Wlogmean,Wlogsigma,Wbins,scale=Wscale)
efficiencies=lat50.get_efficiency_from_wlist(dmvals,pwidths,pprobs)
    
weights=lat50.wplist
#misc_functions.plot_efficiencies(lat50)

# create a grid object
grid=zdm.grid()
grid.pass_grid(zDMgrid,zvals,dmvals,H0)
        
# plots the grid of intrinsic p(DM|z)
misc_functions.plot_grid_2(grid.grid,grid.zvals,grid.dmvals,zmax=1,DMmax=1000,name='Plots/p_dm_z_grid_image.pdf',norm=1,log=True,label='$\\log_{10}p(DM_{\\rm EG}|z)$',conts=[0.16,0.5,0.88],title='Rates at H0 ',H0=setH0)
        
# creates a mask of values in DM space to convolve with the DM
# grid
# best-fit values-ish from green curves in fig 3 of cosmic dm paper
mean=10**2.16
sigma=10**0.51
logmean=np.log10(mean)
logsigma=np.log10(sigma)
mask=pcosmic.get_dm_mask(grid.dmvals,(logmean,logsigma),zvals,plot=True)
        
grid.smear_dm(mask,logmean,logsigma)
# plots the grid of intrinsic p(DM|z)
#misc_functions.plot_grid_2(grid.smear_grid,grid.zvals,grid.dmvals,zmax=1,DMmax=1000,name='Plots/DMX_grid_image.pdf',norm=1,log=True,label='$\\log_{10}p(DM_{\\rm EG}|z)$')
misc_functions.plot_zdm_basic_paper(grid.smear_grid,grid.zvals,grid.dmvals,zmax=3,DMmax=3000,
                                          norm=1,log=True,ylabel='${\\rm DM_{\\rm EG}}$',label='$\\log_{10} p({\\rm DM_{cosmic}+DM_{host}}|z)$',conts=[0.023, 0.159,0.5,0.841,0.977],title='Rates at H0 ',H0=setH0)    
#plot_grid_2(grid.smear_grid2,grid.zvals,grid.dmvals,zmax=1,DMmax=1000,name='DMX_grid_image2.pdf',norm=True,log=True,label='$\\log_{10}p(DM_{\\rm EG}|z)$')
        
# plots grid of effective thresholds
alpha=1.54
grid.calc_thresholds(lat50.meta['THRESH'],efficiencies,alpha=alpha, weights=weights)
print ("done1")
#misc_functions.plot_grid_2(grid.thresholds,grid.zvals,grid.dmvals,zmax=1,DMmax=1000,name='Plots/thresholds_dm_z_grid_image.pdf',norm=1,log=True,label='$\\log (E_{\\rm th})$ [erg]')
    
if lat50.TOBS :
    print ("TOBS there")
else:
    print ("TOBS not there")
print("survey dimension ",lat50.nD)

# calculates rates for given gamma etc
gamma=-1.16
Emax=10**(41.84)
Emin=10**(30)
sfr_n=1.77
grid.calc_dV()
grid.calc_pdv(Emin,Emax,gamma,lat50.beam_b,lat50.beam_o) # calculates volumetric-weighted probabilities
    
#print (grid.Emin, grid.Emax,np.log10(float(grid.Emin)),np.log10(float(grid.Emax)))
    
grid.set_evolution(sfr_n) # sets star-formation rate scaling with z - here, no evoltion...
grid.calc_rates() # calculates rates by multiplying above with pdm plot
#misc_functions.plot_grid_2(grid.pdv,grid.zvals,grid.dmvals,zmax=1,DMmax=1000,name='Plots/pdv.pdf',norm=True,log=True,label='$p(DM_{\\rm EG},z)dV$ [Mpc$^3$]')
#misc_functions.plot_grid_2(grid.rates,grid.zvals,grid.dmvals,zmax=1,DMmax=1000,name='Plots/base_rate_dm_z_grid_image.pdf',norm=2,log=True,label='$f(DM_{\\rm EG},z)p(DM_{\\rm EG},z)dV$ [Mpc$^3$]')
misc_functions.plot_grid_2(grid.rates,grid.zvals,grid.dmvals,zmax=1,DMmax=1000,name='Plots/project_rate_dm_z_grid_image.pdf',norm=2,log=True,label='$f(DM_{\\rm EG},z)p(DM_{\\rm EG},z)dV$ [Mpc$^3$]',project=False, title='Rates at H0 ',H0=setH0)
    
C=4.19
pset=[np.log10(float(Emin)),np.log10(float(Emax)),alpha,gamma,sfr_n,logmean,logsigma,C,setH0]
it.print_pset(pset)
    
#from test.py
muDM=10**pset[5]
Macquart=muDM
    
# plots zdm distribution
misc_functions.plot_grid_2(grid.rates,grid.zvals,grid.dmvals,zmax=0.6,DMmax=1500,
                             norm=2,log=True,label='$\\log_{10} p({\\rm DM}_{\\rm EG},z)$',project=False,FRBDM=lat50.DMEGs,FRBZ=None,Aconts=[0.01,0.1,0.5],Macquart=Macquart,title="H0 value ",H0= setH0)
    
    
llsum= it.calc_likelihoods_1D(grid, lat50, pset)

print (grid.Emin, grid.Emax,np.log10(float(grid.Emin)),np.log10(float(grid.Emax)),setH0)    
print ("initial grid setup done")

#set to True to scan H0 likelihoods

scanoverH0=True
if scanoverH0:    
        
    ###### shows how to do a 1D scan of parameter values #######
    pset=[np.log10(float(grid.Emin)),np.log10(float(grid.Emax)),grid.alpha,grid.gamma,grid.sfr_n,grid.smear_mean,grid.smear_sigma,C,grid.H0]
    
    
    #lEmaxs=np.linspace(40,44,21)
    #lscan,lllist,expected=it.scan_likelihoods_1D(grid,pset,lat50,1,lEmaxs,norm=True)
    #print (lscan, lllist, expected)
    #misc_functions.plot_1d(lEmaxs,lscan,'$E_{\\rm max}$','Plots/test_lik_fn_emax.pdf')
    
    #for H0
    t0=time.process_time()
    
    H0iter=np.linspace(50,130,21)
    lscanH0,lllistH0,expectedH0=it.scan_likelihoods_1D(grid,pset,lat50,8,H0iter,norm=True)
    misc_functions.plot_1d(H0iter,lscanH0,'$H_{\\rm 0}$','Plots/test_lik_fn_emax.pdf')
    t1=time.process_time()
    
    print (lscanH0,"done")
    print ("Took ",t1-t0,"seconds")
    
    
def scan_H0(H0_start,H0_stop,n_iterations,survey,surveyname,plots=True):
    """This function is useful for plotting the graphs for different H0 values (and also finding likelihoods but is VERY slow compared to the updated update_grid routine)"""
    
    t0=time.process_time()
    H0values=np.linspace(H0_start,H0_stop,n_iterations)
    H0likes=[]
    for i in H0values:
        setH0=i
        #setH0=67.74
        cos.set_cosmology(H0=setH0)
           
        #parser.add_argument(", help
        # get the grid of p(DM|z)
        zDMgrid, zvals,dmvals,H0=misc_functions.get_zdm_grid(H0=setH0,new=True,plot=False,method='analytic')
        #pcosmic.plot_mean(zvals,'Plots/mean_DM.pdf')
        
       
        efficiencies=survey.get_efficiency_from_wlist(dmvals,pwidths,pprobs)
        weights=survey.wplist
        
        grid=zdm.grid()
        grid.pass_grid(zDMgrid,zvals,dmvals,H0)
        # creates a mask of values in DM space to convolve with the DM
        # grid
        # best-fit values-ish from green curves in fig 3 of cosmic dm paper
        mean=10**2.16
        sigma=10**0.51
        logmean=np.log10(mean)
        logsigma=np.log10(sigma)
        mask=pcosmic.get_dm_mask(grid.dmvals,(logmean,logsigma),zvals,plot=True)
            
        grid.smear_dm(mask,logmean,logsigma)
        # plots the grid of intrinsic p(DM|z)
        #misc_functions.plot_grid_2(grid.smear_grid,grid.zvals,grid.dmvals,zmax=1,DMmax=1000,name='Plots/DMX_grid_image.pdf',norm=1,log=True,label='$\\log_{10}p(DM_{\\rm EG}|z)$')
        
        #plot_grid_2(grid.smear_grid2,grid.zvals,grid.dmvals,zmax=1,DMmax=1000,name='DMX_grid_image2.pdf',norm=True,log=True,label='$\\log_{10}p(DM_{\\rm EG}|z)$')
            
        # plots grid of effective thresholds
        alpha=1.54
        grid.calc_thresholds(survey.meta['THRESH'],efficiencies,alpha=alpha, weights=weights)
        #misc_functions.plot_grid_2(grid.thresholds,grid.zvals,grid.dmvals,zmax=1,DMmax=1000,name='Plots/thresholds_dm_z_grid_image.pdf',norm=1,log=True,label='$\\log (E_{\\rm th})$ [erg]')
        #if surveygiven.TOBS :
           #print ("TOBS there")
        #else:
            #print ("TOBS not there")
        #print("survey dimension ",surveygiven.nD)
            # calculates rates for given gamma etc
        gamma=-1.16
        Emax=10**(41.84)
        Emin=10**(30)
        sfr_n=1.77
        grid.calc_dV()
        grid.calc_pdv(Emin,Emax,gamma,survey.beam_b,survey.beam_o) # calculates volumetric-weighted probabilities
        #print (grid.Emin, grid.Emax,np.log10(float(grid.Emin)),np.log10(float(grid.Emax)))
        grid.set_evolution(sfr_n) # sets star-formation rate scaling with z - here, no evoltion...
        grid.calc_rates() # calculates rates by multiplying above with pdm plot
        #misc_functions.plot_grid_2(grid.pdv,grid.zvals,grid.dmvals,zmax=1,DMmax=1000,name='Plots/pdv.pdf',norm=True,log=True,label='$p(DM_{\\rm EG},z)dV$ [Mpc$^3$]')
        #misc_functions.plot_grid_2(grid.rates,grid.zvals,grid.dmvals,zmax=1,DMmax=1000,name='Plots/base_rate_dm_z_grid_image.pdf',norm=2,log=True,label='$f(DM_{\\rm EG},z)p(DM_{\\rm EG},z)dV$ [Mpc$^3$]')
    
        C=4.19
        pset=[np.log10(float(Emin)),np.log10(float(Emax)),alpha,gamma,sfr_n,logmean,logsigma,C,setH0]
        #it.print_pset(pset)
        
        if plots:
            misc_functions.plot_grid_2(grid.grid,grid.zvals,grid.dmvals,zmax=1,DMmax=1000,name='Plots/p_dm_z_grid_image.pdf',norm=1,log=True,label='$\\log_{10}p(DM_{\\rm EG}|z)$',conts=[0.16,0.5,0.88],title='Rates at H0 ',H0=setH0)
            misc_functions.plot_zdm_basic_paper(grid.smear_grid,grid.zvals,grid.dmvals,zmax=3,DMmax=3000,
                                              norm=1,log=True,ylabel='${\\rm DM_{\\rm EG}}$',label='$\\log_{10} p({\\rm DM_{cosmic}+DM_{host}}|z)$',conts=[0.023, 0.159,0.5,0.841,0.977],title='Rates at H0 ',H0=setH0)    
            misc_functions.plot_grid_2(grid.rates,grid.zvals,grid.dmvals,zmax=1,DMmax=1000,name='Plots/project_rate_dm_z_grid_image.pdf',norm=2,log=True,label='$f(DM_{\\rm EG},z)p(DM_{\\rm EG},z)dV$ [Mpc$^3$]',project=False, title='Rates at H0 ',H0=setH0)   
            #from test.py
            muDM=10**pset[5]
            Macquart=muDM
            misc_functions.plot_grid_2(grid.rates,grid.zvals,grid.dmvals,zmax=0.6,DMmax=1500,
                                 norm=2,log=True,label='$\\log_{10} p({\\rm DM}_{\\rm EG},z)$',project=False,FRBDM=survey.DMEGs,FRBZ=None,Aconts=[0.01,0.1,0.5],Macquart=Macquart,title="H0 value ",H0= setH0)
            misc_functions.make_dm_redshift(grid,
                                  DMmax=1000,zmax=0.75,loc='upper right',Macquart=Macquart,H0=setH0)
        if survey.nD==1:
            llsum= it.calc_likelihoods_1D(grid, survey, pset, psnr=True)
        else:
            llsum= it.calc_likelihoods_2D(grid, survey, pset, psnr=True)
        H0likes.append(llsum)
        #print (grid.Emin, grid.Emax,np.log10(float(grid.Emin)),np.log10(float(grid.Emax)),setH0)    
        print ("done ",setH0)
    
    t1=time.process_time()
    print ("Done.",n_iterations,"iterations took",t1-t0,"seconds")
    H0likes=np.array(H0likes)
    print (H0likes)
    H0likes=-H0likes
    plt.plot(H0values,H0likes)
    plt.title("Likelihood scan while varying H0 for " + surveyname)
    plt.xlabel("H0 value")
    plt.ylabel("- Log Likelihood")
    plt.show()
    plt.close()
    
scan_H0(50,130,21,survey=lat50,surveyname="lat50",plots=True)