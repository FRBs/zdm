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
from scipy import interpolate
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

np.seterr(divide='ignore')


####setting up the initial grid and plotting some stuff####

setH0=67.74
cos.set_cosmology(H0=setH0)
       
# get the grid of p(DM|z)
zDMgrid, zvals,dmvals,H0=misc_functions.get_zdm_grid(H0=setH0,new=True,plot=False,method='analytic')

    
Wbins=10
Wscale=2
Nbeams=[20,20,20]  #Full beam NOT Std  
thresh=0
method=2
Wlogmean=1.70267
Wlogsigma=0.899148

sdir = os.path.join(resource_filename('zdm', 'data'), 'Surveys/')
lat50=survey.survey()
lat50.process_survey_file(sdir+'CRAFT_class_I_and_II.dat')
DMhalo=50
lat50.init_DMEG(DMhalo)
lat50.init_beam(nbins=Nbeams[0],method=2,plot=False,thresh=thresh) # tells the survey to use the beam file
pwidths,pprobs=survey.make_widths(lat50,Wlogmean,Wlogsigma,Wbins,scale=Wscale)
efficiencies=lat50.get_efficiency_from_wlist(dmvals,pwidths,pprobs)
weights=lat50.wplist

ics=survey.survey()
ics.process_survey_file(sdir+'CRAFT_ICS.dat')
DMhalo=50
ics.init_DMEG(DMhalo)
ics.init_beam(nbins=Nbeams[0],method=2,plot=False,thresh=thresh) # tells the survey to use the beam file
pwidths,pprobs=survey.make_widths(ics,Wlogmean,Wlogsigma,Wbins,scale=Wscale)
efficiencies=ics.get_efficiency_from_wlist(dmvals,pwidths,pprobs)
weights=ics.wplist

pks=survey.survey()
pks.process_survey_file(sdir+'parkes_mb_class_I_and_II.dat')
DMhalo=50
pks.init_DMEG(DMhalo)
pks.init_beam(nbins=Nbeams[0],method=2,plot=False,thresh=thresh) # tells the survey to use the beam file
pwidths,pprobs=survey.make_widths(pks,Wlogmean,Wlogsigma,Wbins,scale=Wscale)
efficiencies=pks.get_efficiency_from_wlist(dmvals,pwidths,pprobs)
weights=pks.wplist

ICS892=survey.survey()
ICS892.process_survey_file(sdir+'CRAFT_ICS_892.dat')
ICS892.init_DMEG(DMhalo)
ICS892.init_beam(nbins=Nbeams[0],method=2,plot=False,thresh=thresh) # tells the survey to use the beam file
pwidths,pprobs=survey.make_widths(ICS892,Wlogmean,Wlogsigma,Wbins,scale=Wscale)
efficiencies892=ICS892.get_efficiency_from_wlist(dmvals,pwidths,pprobs)

surveys=[lat50,ics,ICS892,pks]

#updated best-fit values
alpha_method=1
logmean=2.11
logsigma=0.53
alpha=1.55
gamma=-1.09
Emax=10**(41.7)
Emin=10**(30)
sfr_n=1.67
C=3.188

#alpha_method=1
#Emin=10**30
#Emax =10**41.40
#alpha =-0.66
#gamma = -1.01
#sfr_n= 0.73
#logmean=2.18
#logsigma=0.48
#C=2.36 ##it.GetFirstConstantEstimate(grids,surveys,pset)
    
pset=[np.log10(float(Emin)),np.log10(float(Emax)),alpha,gamma,sfr_n,logmean,logsigma,C,setH0]
it.print_pset(pset)

grids=misc_functions.initialise_grids(surveys,zDMgrid, zvals,dmvals,pset,wdist=True,source_evolution=0,alpha_method=1)

plots=False
zmax=[0.6,1,1,3]
DMmax=[1500,2000,2000,3000]
if plots:
    for i in range (len(surveys)):
            grid=grids[i]
            sv=surveys[i]
            pcosmic.plot_mean(zvals,'Plots/mean_DM.pdf')
            #misc_functions.plot_efficiencies(lat50)
            misc_functions.plot_grid_2(grid.grid,grid.zvals,grid.dmvals,zmax=zmax[i],DMmax=DMmax[i],
                                       name='Plots/p_dm_z_grid_image.pdf',norm=1,log=True,
                                       label='$\\log_{10}p(DM_{\\rm EG}|z)$',
                                       conts=[0.16,0.5,0.88],title='Grid at H0 '+str(i),
                                       H0=setH0,showplot=True)
            misc_functions.plot_zdm_basic_paper(grid.smear_grid,grid.zvals,grid.dmvals,zmax=zmax[i],
                                                DMmax=DMmax[i],norm=1,log=True,
                                                ylabel='${\\rm DM_{\\rm EG}}$',
                                                label='$\\log_{10} p({\\rm DM_{cosmic}+DM_{host}}|z)$',
                                                conts=[0.023, 0.159,0.5,0.841,0.977],
                                                title='Smear grid at H0 '+str(i),H0=setH0,
                                                showplot=True)    
            misc_functions.plot_grid_2(grid.pdv,grid.zvals,grid.dmvals,zmax=zmax[i],DMmax=DMmax[i],
                                       name='Plots/pdv.pdf',norm=True,log=True
                                       ,label='$p(DM_{\\rm EG},z)dV$ [Mpc$^3$]',
                                       title="Pdv at H0" + str(i),showplot=True)
            misc_functions.plot_grid_2(grid.rates,grid.zvals,grid.dmvals,zmax=zmax[i],DMmax=DMmax[i],
                                       name='Plots/project_rate_dm_z_grid_image.pdf',norm=2,
                                       log=True,label='$f(DM_{\\rm EG},z)p(DM_{\\rm EG},z)dV$ [Mpc$^3$]',
                                       project=False, title='Rates at H0 '+str(i),H0=setH0,
                                       showplot=True) 
            
            muDM=10**pset[5]
            Macquart=muDM
            misc_functions.plot_grid_2(grid.rates,grid.zvals,grid.dmvals,zmax=zmax[i],DMmax=DMmax[i],
                                norm=2,log=True,label='$\\log_{10} p({\\rm DM}_{\\rm EG},z)$',
                                project=False,FRBDM=sv.DMEGs,FRBZ=None,Aconts=[0.01,0.1,0.5],
                                Macquart=Macquart,title="H0 value "+str(i),H0= setH0,showplot=True)
            misc_functions.make_dm_redshift(grid,
                                DMmax=DMmax[i],zmax=zmax[i],loc='upper right',Macquart=Macquart,
                                H0=setH0,showplot=True) 

      
print ("initial grid setup done")

scanoverH0=False 
# just testing....should NOT be used (update_grid routine should not be modified)
if scanoverH0:    
    for k in range (len(surveys)):
        grid=grids[k]
        sv=surveys[k]
        
        ###### shows how to do a 1D scan of parameter values #######
        pset=[np.log10(float(grid.Emin)),np.log10(float(grid.Emax)),grid.alpha,grid.gamma,grid.sfr_n,grid.smear_mean,grid.smear_sigma,C,grid.H0]
        
        
        #lEmaxs=np.linspace(40,44,21)
        #lscan,lllist,expected=it.scan_likelihoods_1D(grid,pset,lat50,1,lEmaxs,norm=True)
        #print (lscan, lllist, expected)
        #misc_functions.plot_1d(lEmaxs,lscan,'$E_{\\rm max}$','Plots/test_lik_fn_emax.pdf')
        
        #for H0
        t0=time.process_time()
        
        H0iter=np.linspace(50,130,4)
        lscanH0,lllistH0,expectedH0=it.scan_likelihoods_1D(grid,pset,sv,8,H0iter,norm=True)
        misc_functions.plot_1d(H0iter,lscanH0,'$H_{\\rm 0}$','Plots/test_lik_fn_emax.pdf')
        t1=time.process_time()
        
        print (lscanH0,"done")
        print ("Took ",t1-t0,"seconds")
    
    
def scan_H0(H0_start,H0_stop,n_iterations,surveys,plots=False):
    """Routine for scanning over H0 values in 1D"""
    
    t0=time.process_time()
    H0values=np.linspace(H0_start,H0_stop,n_iterations)
    
    H0likes=[]
    for i in H0values:
        setH0=i
        
        cos.set_cosmology(H0=setH0)
        zDMgrid, zvals,dmvals,H0=misc_functions.get_zdm_grid(H0=setH0,new=True,plot=False,
                                                             method='analytic')
        
        mean=10**2.16
        sigma=10**0.51
        logmean=np.log10(mean)
        logsigma=np.log10(sigma)
        alpha=1.54
        gamma=-1.16
        Emax=10**(41.84)
        Emin=10**(30)
        sfr_n=1.77
        C=4.19
        pset=[np.log10(float(Emin)),np.log10(float(Emax)),alpha,gamma,sfr_n,logmean,logsigma,C,setH0]
        it.print_pset(pset)
        
        grids=misc_functions.initialise_grids(surveys,zDMgrid, zvals,dmvals,pset,wdist=True,source_evolution=0,alpha_method=1)
        
        grid=grids[0]
        if plots:
            
            pcosmic.plot_mean(zvals,'Plots/mean_DM.pdf', title="Mean DM at" + str(i))
            
            misc_functions.plot_zdm_basic_paper(grid.grid,grid.zvals,grid.dmvals,zmax=3,DMmax=3000,
                                               name='Plots/p_dm_z_grid_image.pdf',norm=1,log=True,
                                               label='$\\log_{10}p(DM_{\\rm cosmic}|z)$', ylabel='${\\rm DM}_{\\rm cosmic}$',
                                               conts=[0.16,0.5,0.88],title='Cosmological p(z,DM) at $H_{0}$',
                                               H0=setH0,showplot=True)
            misc_functions.plot_zdm_basic_paper(grid.smear_grid,grid.zvals,grid.dmvals,
                                                zmax=3,DMmax=3000,norm=1,log=True,
                                                ylabel='${\\rm DM_{\\rm EG}}$',
                                                label='$\\log_{10} p({\\rm DM_{cosmic}+DM_{host}}|z)$',
                                                conts=[0.023, 0.159,0.5,0.841,0.977],
                                                title='Cosmological + Host p(z,DM) at $H_{0}$ ',H0=setH0,
                                                showplot=True) 
        likessurvey=[]
        for j in range (len(surveys)):
            grid=grids[j]
            sv=surveys[j]
            
            if plots:
                misc_functions.plot_grid_2(grid.pdv,grid.zvals,grid.dmvals,zmax=zmax[j],DMmax=DMmax[j],
                                           name='Plots/pdv.pdf',norm=True,log=True,
                                           label='$p(DM_{\\rm EG},z)dV$ [Mpc$^3$]',showplot=True,
                                           title='Pdv of '+ str(sv.name)+ 'at $H_{0}$ ')
                misc_functions.plot_grid_2(grid.rates,grid.zvals,grid.dmvals,zmax=zmax[j],DMmax=DMmax[j],
                                           name='Plots/project_rate_dm_z_grid_image.pdf',norm=2,
                                           log=True,label='$f(DM_{\\rm EG},z)p(DM_{\\rm EG},z)dV$ [Mpc$^3$]',
                                           project=False, title='Rates of ' + str(sv.name) + ' at $H_{0}$ ',
                                           H0=setH0,showplot=True)   
                #from test.py
                muDM=10**pset[5]
                Macquart=muDM
                misc_functions.plot_grid_2(grid.rates,grid.zvals,grid.dmvals,zmax=zmax[j],DMmax=DMmax[j],
                                     norm=2,log=True,label='$\\log_{10} p({\\rm DM}_{\\rm EG},z)$',
                                     project=False,FRBDM=sv.DMEGs,FRBZ=None,Aconts=[0.01,0.1,0.5],
                                     Macquart=Macquart,title='p(z,DM) of '+ str(sv.name) + ' at $H_{0}$',H0= setH0, showplot=True)
                misc_functions.make_dm_redshift(grid,
                                      DMmax=DMmax[j],zmax=zmax[j],loc='upper right',Macquart=Macquart,
                                      H0=setH0, showplot=True)
            if sv.nD==1:
                llsum= it.calc_likelihoods_1D(grid, sv, pset, psnr=True)
            else:
                llsum= it.calc_likelihoods_2D(grid, sv, pset, psnr=True)
                
            likessurvey.append(llsum)
            
            #print (grid.Emin, grid.Emax,np.log10(float(grid.Emin)),np.log10(float(grid.Emax)),setH0)    
        print ("Calculationg done for $H_{0}$ ",setH0)
            
        likessurvey=np.array(likessurvey)
        H0likes.append(likessurvey)
        
    t1=time.process_time()
    
    print ("Done. ",n_iterations," iterations took ",t1-t0," seconds")
    
    H0likes=np.array(H0likes)
    print (H0likes)
    H0likes=np.transpose(H0likes)
    H0likes=-H0likes
    for a in range(len(surveys)):
        sv=surveys[a]
        H0likesa=H0likes[a]
        plt.plot(H0values,H0likesa)
        plt.title("Likelihood scan while varying H0 for " + str(sv.name))
        plt.xlabel("H0 value")
        plt.ylabel("Log Likelihood")
        plt.show()
        plt.close()
    
    H0likessum=np.sum(H0likes,axis=0)
    plt.plot(H0values,H0likessum)
    tckj = interpolate.splrep(H0values,H0likessum, s=0)
    H0new=np.arange(30,120,0.01)
    ynewj = interpolate.splev(H0new, tckj)
    plt.plot(H0new,ynewj)
    #plt.title("Likelihood scan while varying H0 for " + str(sv.name))
    plt.xlabel("Value of ${\\rm H_{\\rm 0}}$ in km s${^{-1}}$ Mpc{$^-1$}")
    plt.ylabel("log Likelihood")
    plt.show()
    
    H0best=H0new[np.argmin(ynewj)]
    print ('Best fit for alpha method=' +str(grid.alpha_method) +'$H_{0}$ is', H0best)
    
plt.close()


    
scan_H0(50,100,5,surveys,plots=True)