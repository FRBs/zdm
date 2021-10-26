#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 25 02:51:08 2021

@author: esanmoulighosh
"""

import numpy as np
from zdm import zdm
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
from zdm.tests import mc_statistics as mc
from scipy.interpolate import bisplrep, bisplev
import pickle
import copy

def scan_H0_MC(N,mcsample, survey, H0iter, plots=False):
    """"Routine for performing a 1-D scanning over H0 values for a single MC survey
    Input: 
    N -: For N*survey.NFRBs mock FRBs for a specific survey
    mcsample :- The mcsample for a specific survey
    survey :- survey file
    H0iter :- array of values to iterate H0 over
    
    Output:
    lvals : Array of likelihood values (N*H0iter). Considers Pn
    lnoPns : Array of likelihood values (N*H0iter). Doesn't consider Pn """
    
    lvals = []
    lnoPns = []
    longll = []
    for i in H0iter:
        t0 = time.process_time()
        setH0 = i
        cos.set_cosmology(H0=setH0)
        zDMgrid, zvals, dmvals, H0 = misc_functions.get_zdm_grid(
            H0=setH0, new=True, plot=False, method='analytic')

        mean=10**2.16
        sigma=10**0.51
        logmean=np.log10(mean)
        logsigma=np.log10(sigma)
        alpha=1.54
        gamma=-1.16
        Emin=10**(30)
        sfr_n=1.77
        C=4.19
        
        #if i<67.74:
            #Emax=10**(42.44)
        #else:
        Emax=10**(41.84)
    
        
        pset=[np.log10(float(Emin)),np.log10(float(Emax)),alpha,gamma,sfr_n,logmean,logsigma,C,setH0]
        #it.print_pset(pset)
        
        grids=misc_functions.initialise_grids(survey,zDMgrid, zvals,dmvals,pset,wdist=True,
                                              source_evolution=0,alpha_method=1)
        
        grid=grids[0]

        if plots:
            misc_functions.plot_grid_2(grid.grid, grid.zvals, grid.dmvals, zmax=1, DMmax=1000, 
                                       name='Plots/p_dm_z_grid_image.pdf',norm=1, log=True, 
                                       label='$\\log_{10}p(DM_{\\rm EG}|z)$', 
                                       conts=[0.16, 0.5, 0.88], title='Rates at H0 ', H0=setH0)
            misc_functions.plot_zdm_basic_paper(grid.smear_grid, grid.zvals, grid.dmvals, zmax=3,
                                                DMmax=3000,norm=1, log=True, ylabel='${\\rm DM_{\\rm EG}}$', 
                                                label='$\\log_{10} p({\\rm DM_{cosmic}+DM_{host}}|z)$', 
                                                conts=[0.023, 0.159, 0.5, 0.841, 0.977], 
                                                title='Rates at H0 ', H0=setH0)
            misc_functions.plot_grid_2(grid.rates, grid.zvals, grid.dmvals, zmax=1, DMmax=1000,
                                       name='Plots/project_rate_dm_z_grid_image.pdf',norm=2, 
                                       og=True, label='$f(DM_{\\rm EG},z)p(DM_{\\rm EG},z)dV$ [Mpc$^3$]', 
                                       project=False, title='Rates at H0 ', H0=setH0)
            # from test.py
            muDM = 10**pset[5]
            Macquart = muDM
            misc_functions.plot_grid_2(grid.rates, grid.zvals, grid.dmvals, zmax=0.6, DMmax=1500,
                                       norm=2, log=True, label='$\\log_{10} p({\\rm DM}_{\\rm EG},z)$',
                                       project=False, FRBDM=survey.DMEGs, FRBZ=None, Aconts=[0.01, 0.1, 0.5],
                                       Macquart=Macquart, title="H0 value ", H0=setH0)
            misc_functions.make_dm_redshift(grid,DMmax=1000, zmax=0.75, loc='upper right',
                                            Macquart=Macquart, H0=setH0)

        t1 = time.process_time()

        nsamples = mcsample.shape[0]

        # makes a deep copy of the survey
        lvalssurvey = []
        longllsurvey = []
        lnoPn = []
        s = copy.deepcopy(survey)
        NFRBs = survey.NFRB
        nsurveys = int(nsamples/NFRBs)

        #print (NFRBs,nsamples,nsurveys)
        s.NFRB=nsamples 
        # NOTE: does NOT change the assumed normalised FRB total!
        for q in range(nsurveys):
            sstart = q*NFRBs
            sstop = (q+1)*NFRBs
            sampleq = mcsample[sstart:sstop]
            s.DMEGs = sampleq[:, 1]
            s.Ss = sampleq[:, 4]

            if s.nD == 1:  # DM, snr only
                llsum, lllist, expected, longlist = it.calc_likelihoods_1D(
                    grid, s, pset, psnr=True, Pn=True, dolist=2)
            else:
                s.Zs = sampleq[:, 0]
                llsum, lllist, expected, longlist = it.calc_likelihoods_2D(
                    grid, s, pset, psnr=True, Pn=True, dolist=2)

            # we should preserve the normalisation factor for Tobs from lllist
            Pzdm, Pn, Psnr = lllist
            lltot = Pn + np.sum(longlist)
            noPn = np.sum(longlist)

            lnoPn.append(noPn)
            lvalssurvey.append(llsum)
            longllsurvey.append(longlist)

        lvalssurvey = np.array(lvalssurvey)
        lvals.append(lvalssurvey)
        longll.append(longllsurvey)
        lnoPns.append(lnoPn)

    t2 = time.process_time()
    lvals = np.array(lvals)
    longll = np.array(longll)
    lnoPns= np.array(lnoPns)

    print("Iterating over H0 done. Took", t2-t1, "seconds")

    lvals = np.transpose(lvals)
    lnoPns=np.transpose(lnoPns)
    #print(np.shape(lvals))

    return lvals, lnoPns

def MC_H0(N, H0start, H0stop, niter, plot=False):
    """"Routine for computing likelihood of all 4 survey files and plotting stuff"
    Input:
    N -: For N*survey.NFRBs mock FRBs for a specific survey
    H0start, H0stop, niter -: To iterate over np.linspace(H0start, H0stop, niter)
    """
    samples,surveys = mc.main(N)
    print (len(samples), {surveys[i].name for i in range(0,4)})
    lsurvey = []
    H0iter = np.linspace(H0start, H0stop, niter)
    H0new = np.arange(H0start, H0stop, 0.01)
    lspline = []
    H0minima = []
    lnoPnvals = []
    for i in range(len(surveys)):
        print (i)
        lvals, lnoPns = scan_H0_MC(N,samples[i], surveys[i], H0iter)
        
        #incase likelihoods are already calculated
        #lvals=np.load('lvals_survey_'+str(i)+'H0 values='+str(niter)+".npy")
        #lnoPns=np.load('lnopns_survey_'+str(i)+'H0 values='+str(niter)+".npy")
        
        #print ("lvals",lvals)
        #print ("lnoPns",lnoPns)
        
        #saving the likelihoods for each mock survey
        
        np.save('lvals_survey_'+str(i)+'H0 values='+str(niter),lvals)
        np.save('lnopns_survey_'+str(i)+'H0 values='+str(niter),lnoPns)
        
        lsurvey.append(lvals)
        lnoPnvals.append(lnoPns)
        lsplinesurvey = []

        for j in range(len(lvals)):
            lvalsj = lvals[j]
            H0iterj = H0iter[np.isfinite(lvalsj)]
            lvalsj = lvalsj[np.isfinite(lvalsj)]

            if len(lvalsj) > 3:
                tckj = interpolate.splrep(H0iterj, lvalsj, s=0)
                ynewj = interpolate.splev(H0new, tckj)

                minimaj = np.argmax(ynewj)

                if plot:
                    plt.plot(H0iterj, -lvalsj, 'o', color='red')
                    plt.plot(H0new, -ynewj, color='yellow')
                    plt.title(
                        "Likelihood scan while varying H0 survey "+str(i)+" snumber "+str(j))
                    plt.xlabel("H0 value")
                    plt.ylabel("- Log Likelihood")
                    plt.tight_layout()
                    plt.show()
                    plt.close()
    
    lsurvey=np.array(lsurvey)
    print("lsurvey",np.shape(lsurvey)) #should be (4*N*H0iter)
    
    #approx 6% of ICS survey likelihoods come out to be -inf or nan when H0<true H0 for mcsample,
    #so to manage them following two routines can be used
    
    infmanage2 = True
    if infmanage2:
        
        v1 = False   #setting inf values to the average of the other surveys' at that H0value
        if v1:
            for i in range(len(surveys)):
                lsurveyi = lsurvey[i]
                for j in range(len(lsurveyi)):
                    lsurveyij = lsurveyi[j]
                    for k in range(len(lsurveyij)):
                        lsurveyijk = lsurveyij[k]
                        
                        if np.isfinite(lsurveyijk) == False:
                            if np.isfinite(lsurvey[i-1][j][k]) and np.isfinite(lsurvey[i+1][j][k]):
                                value = float(
                                    (lsurvey[i-1][j][k]+lsurvey[i+1][j][k])/2)
                            elif np.isfinite(lsurvey[i-1][j][k]):
                                value = lsurvey[i-1][j][k]
                            elif np.isfinite(lsurvey[i+1][j][k]):
                                value = lsurvey[i+1][j][k]
                                
                            lsurveyij[k] = value

        v2 = True       #setting inf values to the average of neighbouring H0values of the same survey (better results?)
        if v2:
            for i in range(len(surveys)):
                lsurveyi = lsurvey[i]
                for j in range(len(lsurveyi)):
                    lsurveyij = lsurveyi[j]
                    for k in range(len(lsurveyij)):
                        lsurveyijk = lsurveyij[k]
                        
                        if np.isfinite(lsurveyijk) == False:
                            if np.isfinite(lsurveyij[k-1]) and np.isfinite(lsurveyij[k+1]):
                                value = float(
                                    (lsurveyij[k-1]+lsurveyij[k+1])/2)
                            elif np.isfinite(lsurveyij[k-1]):
                                value = lsurveyij[k-1]
                            elif np.isfinite(lsurveyij[k+1]):
                                value = lsurveyij[k+1]
                            lsurveyij[k] = value
                            
                            
    #summing up all the likelihoods obtained for each H0 value for all four surveys
    cumlik = np.sum(lsurvey, axis=0)
    
    #print("cumlik",np.shape(cumlik)) #should be (N*H0iter)
        
    
    plotindv=False
   
    cumlikspline = []
    for q in range(len(cumlik)):
        cumlikq = cumlik[q]
        tckq = interpolate.splrep(H0iter, cumlikq, s=0)
        ynewq = interpolate.splev(H0new, tckq)
        if plotindv:
            plt.title("Cumulative Likelihood function of all surveys")
            plt.xlabel("H0 value")
            plt.ylabel("- Log Likelihood")
            plt.plot(H0iter, -cumlikq, 'o', color='red')
            plt.plot(H0new, -ynewq, color='yellow')
            plt.show()
            plt.close()
        cumlikspline.append(ynewq)

    #print(np.shape(cumlikspline))
    
    #now adding up all the likelihoods for N observations
    # array of (1*H0iter) shape
    sumlik = np.sum(cumlik, axis=0)  
    tckfinal = interpolate.splrep(H0iter, sumlik, s=0)
    likfinal = interpolate.splev(H0new, tckfinal)
    plt.plot(H0iter, -sumlik, 'o', color='red')
    plt.plot(H0new, -likfinal, color='blue')
    plt.title("Likelihood function for " +str(N)+' surveys')
    plt.xlabel("H0 value")
    plt.ylabel("- Log Likelihood")
    plt.xlim(50, 100)
    plt.tight_layout()
    plt.show()
    plt.close()
    
    print ("Best fit value of $H_{0}$ for "+str(N)+" surveys is ", H0new[np.argmax(likfinal)])
    
    
    cumlikspline = np.array(cumlikspline)
    H0mins = []  
    # getting best-fit H0 from each of the N observations from the spline
    for k in range(len(cumlikspline)):
        indexH0 = np.argmax(cumlikspline[k])
        H0minimum = H0new[indexH0]
        H0mins.append(H0minimum)

    truthdiff = []
    H0diff = []  
    #getting the difference between the best-fit H0 and the true H0 for each of the N observations
    for o in range(len(cumlikspline)):
        maxlik = np.max(cumlikspline[o])
        maxlikH0 = np.argmax(cumlikspline[o])
        H0best = H0new[maxlikH0]
        indexactual = np.where(np.isclose(H0new, 67.7))
        actuallik = cumlikspline[o][indexactual]
        actualH0 = H0new[indexactual]
        #print(maxlik, actuallik)
        diff = float(maxlik)-float(actuallik)
        diffH = float(H0best)-float(actualH0)
        truthdiff.append(diff)
        H0diff.append(diffH)

    #print(H0mins)
    #print(truthdiff)

    H0mins = np.array(H0mins)
    print ("Mean value of H0 is", np.mean(H0mins))
    plt.hist(H0mins, bins=15, range=(55, 90))
    plt.title("Histogram of best-fit H0 value for MC sample")
    plt.xlabel("H0 value")
    plt.ylabel("p(H0)")
    plt.show()
    plt.close()

    truthdiff = np.array(truthdiff)
    plt.hist(truthdiff, bins=15)
    plt.title("Histogram of diff in best-fit H0 and true H0 value")
    plt.xlabel("difference")
    plt.ylabel("p(difference)")
    plt.show()
    plt.close()

MC_H0(50,55,90,5)

