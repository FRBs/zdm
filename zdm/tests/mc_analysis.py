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
import seaborn as sns
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

        #updated best-fit values
        alpha_method=0
        lmean=2.11
        lsigma=0.53
        alpha=1.55
        gamma=-1.09
        lEmax=41.7
        lEmin=30
        sfr_n=1.67
        C=3.188
    
    	#alpha_method=1
    	#lEmin=30
    	#lEmax =41.40
    	#alpha =-0.66
    	#gamma = -1.01
    	#sfr_n= 0.73
    	#lmean=2.18
    	#lsigma=0.48
    	#C=2.36 ##it.GetFirstConstantEstimate(grids,surveys,pset)
        
        #if i<67.74:
            #Emax=10**(42.44)
    
        
        pset=[lEmin,lEmax,alpha,gamma,sfr_n,lmean,lsigma,C,setH0]
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

#MC_H0(50,55,90,5)

def scan_H0_lEmax_MC(N,sample, survey, H0iter, Emaxiter, plots=False):
    lvalstotal = []
    lnoPns = []
    longll = []
    for j in Emaxiter:
        Elike=[]
        ElikenoPn=[]
        for i in H0iter:
            t0 = time.process_time()
            setH0 = i
            # setH0=67.74
            cos.set_cosmology(H0=setH0)
    
            zDMgrid, zvals, dmvals, H0 = misc_functions.get_zdm_grid(H0=setH0, new=True, plot=False, method='analytic')
            
            mean=10**2.16
            sigma=10**0.51
            logmean=np.log10(mean)
            logsigma=np.log10(sigma)
            alpha=1.54
            gamma=-1.16
            Emin=10**(30)
            sfr_n=1.77
            C=4.19
            
            Emax=10**j
            
            pset=[np.log10(float(Emin)),np.log10(float(Emax)),alpha,gamma,sfr_n,logmean,logsigma,C,setH0]
            it.print_pset(pset)
            
            grids=misc_functions.initialise_grids(survey,zDMgrid, zvals,dmvals,pset,wdist=True,source_evolution=0,alpha_method=0)
            
            grid=grids[0]
    
            if plots:
                misc_functions.plot_grid_2(grid.grid, grid.zvals, grid.dmvals, zmax=1, DMmax=1000, name='Plots/p_dm_z_grid_image.pdf',
                                           norm=1, log=True, label='$\\log_{10}p(DM_{\\rm EG}|z)$', conts=[0.16, 0.5, 0.88], title='Rates at H0 ', H0=setH0)
                misc_functions.plot_zdm_basic_paper(grid.smear_grid, grid.zvals, grid.dmvals, zmax=3, DMmax=3000,
                                                    norm=1, log=True, ylabel='${\\rm DM_{\\rm EG}}$', label='$\\log_{10} p({\\rm DM_{cosmic}+DM_{host}}|z)$', conts=[0.023, 0.159, 0.5, 0.841, 0.977], title='Rates at H0 ', H0=setH0)
                misc_functions.plot_grid_2(grid.rates, grid.zvals, grid.dmvals, zmax=1, DMmax=1000, name='Plots/project_rate_dm_z_grid_image.pdf',
                                           norm=2, log=True, label='$f(DM_{\\rm EG},z)p(DM_{\\rm EG},z)dV$ [Mpc$^3$]', project=False, title='Rates at H0 ', H0=setH0)
                # from test.py
                muDM = 10**pset[5]
                Macquart = muDM
                misc_functions.plot_grid_2(grid.rates, grid.zvals, grid.dmvals, zmax=0.6, DMmax=1500,
                                           norm=2, log=True, label='$\\log_{10} p({\\rm DM}_{\\rm EG},z)$', project=False, FRBDM=survey.DMEGs, FRBZ=None, Aconts=[0.01, 0.1, 0.5], Macquart=Macquart, title="H0 value ", H0=setH0)
                misc_functions.make_dm_redshift(grid,
                                                DMmax=1000, zmax=0.75, loc='upper right', Macquart=Macquart, H0=setH0)
    
            t1 = time.process_time()
    
            print("iteration done. Took ", t1-t0, " seconds")
            nsamples = sample.shape[0]
    
            # makes a deep copy of the survey
            lvalssurvey = []
            longllsurvey = []
            lnoPn = []
            s = copy.deepcopy(survey)
            NFRBs = survey.NFRB
            nsurveys = int(nsamples/NFRBs)
    
            s.NFRB=nsamples 

            svElike=[]
            svElikenoPn=[]
            for q in range(nsurveys):
                sstart = q*NFRBs
                sstop = (q+1)*NFRBs
                sampleq = sample[sstart:sstop]
                s.DMEGs = sampleq[:, 1]
                s.Ss = sampleq[:, 4]
                
                if s.nD == 1:  # DM, snr only
                    llsum, lllist, expected, longlist = it.calc_likelihoods_1D(
                        grid, s, pset, psnr=True, Pn=True, dolist=2)
                else:
                    s.Zs = sampleq[:, 0]
                    llsum, lllist, expected, longlist = it.calc_likelihoods_2D(
                        grid, s, pset, psnr=True, Pn=True, dolist=2)
    
            
                Pzdm, Pn, Psnr = lllist
                lltot = Pn+ np.sum(longlist)
                noPn = np.sum(longlist)

                svElikenoPn.append(noPn)
                svElike.append(lltot) 
                longllsurvey.append(longlist)

            svElike = np.array(svElike)
            svElikenoPn = np.array(svElikenoPn)

            #print ("lvalssurvey after all nsurveys for H0=",i,lvalssurvey)
            #print ("svElikenoPn",np.shape(svElikenoPn))

            print ("shape of svElike for it is ",np.shape(svElike))  #array of N likelihood values for a certain H0
            print (svElike)
            
            Elike.append(svElike) 
            longll.append(longllsurvey)
            ElikenoPn.append(svElikenoPn)


        ElikenoPn=np.array(ElikenoPn)
        Elike=np.array(Elike)
        print ("ElikenoPn", np.shape(ElikenoPn)) # array of H0iter*N likelihood values for a certain lEmax
        lvalstotal.append(Elike)
        lnoPns.append(ElikenoPn)
        
    
    t2 = time.process_time()
    lvalstotal = np.array(lvalstotal)
    longll = np.array(longll)
    lnoPns= np.array(lnoPns)
    print ("lnoPns", np.shape(lnoPns)) # lEmaxiter*H0iter*N likelihood values

    print("Done.Took", t2-t1, "seconds")

    print(lvalstotal, np.shape(lvalstotal))
    
    return lvalstotal, lnoPns

def MC_H0_lEmax(N, Ns=500,plotall=True,plotindv=False):
    """"This routine performs a 2-D likelihood scan over lEmax and H0 for a MC sample
    N -: Number of total experiments (N*survey.NFRBs mock FRBs for a specific survey)
    Ns -: The number of data points for interpolation of lEmax and H0 values, 
        i.e lEmax=np.linspace(start,stop,Ns)
    plotall: Plot each of the 2D grids for N experiments
    plotindv: NOT SAFE Plots A LOT" of grids(for each survey and experiment)"""
    
    #the spline interpolation works when H0iter and lEmaxiter have the same length
    #ie H0n=lEmaxn
    #
    Nmain=N
    samples, surveys = mc.main(Nmain)
    lsurvey = []
    H0start,H0stop,H0n = 40,110,6   #more than 6 will take much more time
    H0iter=np.linspace(H0start,H0stop,H0n)
    H0iter=np.round(H0iter,1)
    H0labelstr=[str(e) for e in H0iter.tolist()]
    
    lEmaxstart,lEmaxstop,lEmaxn=41.84,42.84,6
    lEmaxiter= np.linspace(lEmaxstart,lEmaxstop,lEmaxn)
    lEmaxiter=np.round(lEmaxiter,2)
    lEmaxlabelstr=[str(e) for e in lEmaxiter.tolist()]
    
    
    H0new = np.linspace(H0start,H0stop,Ns)
    lEmaxnew= np.linspace(lEmaxstart,lEmaxstop,Ns)

    
    for i in range(len(surveys)):
        try:
            lvals=np.load('lvals_lEmax_H0_'+str(i)+'_'+str(len(H0iter))+'.npy')
            lnoPns=np.load('lnoPns_lEmax_H0_'+str(i)+'_'+str(len(H0iter))+'.npy')
        except:
            lvals, lnoPns = scan_H0_lEmax_MC(N,samples[i], surveys[i],H0iter, lEmaxiter)
            np.save('lvals_lEmax_H0_'+str(i)+'_'+ str(len(H0iter)), lvals)
            np.save('lnoPns_lEmax_H0_'+str(i)+'_'+ str(len(H0iter)), lnoPns)

        #print(np.shape(lvals),lvals)
        lsurvey.append(lvals)
        
        #for plotting each individual survey grid - just for testing
        
        for j in range (np.shape(lvals)[2]):
            lvalj=lvals[:,:,j]
            shape=np.shape(lvalj)
            
            #for managing inf and nan values which show up at a rate of around 0.1 % for ICS
            lshape=np.shape(lvalj)
            lvalj=lvalj.flatten()
            lvaljnoinf=lvalj[np.isfinite(lvalj)]
            lmin=np.min(lvaljnoinf)
            lvalj[~np.isfinite(lvalj)]=lmin
            lvalj=np.reshape(lvalj, lshape)
            
            Xn, Yn = np.meshgrid(lEmaxiter, H0iter)
            
            #print (np.shape(Xn),np.shape(Yn),np.shape(lvalj))
            
            #performing a 2D interpolation
            f= bisplrep(Xn, Yn, lvalj, s=8)
            data1 = bisplev(lEmaxnew,H0new,f)
            data1=np.transpose(data1)
            
            #print (np.shape(data1))
            x,y=np.shape(data1)[0],np.shape(data1)[1]
            
            ###just to check-these will create A LOT of plots for each individual mock survey 
            ###- not recommended####
            
            if plotindv:
                plt.figure()
                for k in range (len(lvalj)):
                    lvaljk=lvalj[k]
                    lvaljknoinf=lvaljk[np.isfinite(lvaljk)]
                    H0iterj=H0iter[np.isfinite(lvaljk)]
                    plt.plot(H0iterj,-lvaljknoinf, label='lEmax '+str(lEmaxiter[k]))
    
                plt.xlabel('H0')
                plt.ylabel('-loglik')
                plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                plt.title('For survey '+str(i))
                plt.grid()
                #plt.show()
                plt.close()
                
                ###########coarse grid#############################################
                fig,ax=plt.subplots()
                p1=plt.imshow(lvalj, origin='lower')
                ax.set_yticks(np.arange(0,lEmaxn))
                ax.set_xticks(np.arange(0,H0n))
                ax.set_yticklabels(lEmaxlabelstr)
                ax.set_xticklabels(H0labelstr)
                cbar=plt.colorbar()
                cbar.set_label('loglik')
                plt.title('For survey '+str(i)+' raw')
                #plt.show()
                plt.close()
                
                ############3interpolated 2D grid with projection##################
                plt.figure(1, figsize=(8, 8))
                left, width = 0.1, 0.65
                bottom, height = 0.1, 0.65
                gap=0.11
                woff=width+gap+left
                hoff=height+gap+bottom
                dw=1.-woff-gap
                dh=1.-hoff-gap
                gap=0.11
                
                rect_2D = [left, bottom, width, height]
                rect_1Dx = [left, hoff, width, dh+gap]
                rect_1Dy = [woff-0.03, bottom, dw+gap, height]
                rect_cb = [woff,hoff-0.05,dw*0.95,dh+0.15]
                ax1=plt.axes(rect_2D)
                axx=plt.axes(rect_1Dx)
                axy=plt.axes(rect_1Dy)
                acb=plt.axes(rect_cb)
                axy.set_yticklabels([])
                
                plt.sca(ax1)
                p1=plt.imshow(data1,interpolation='none', origin='lower')
                ax1.set_yticks(np.linspace(0,y,lEmaxn))
                ax1.set_xticks(np.linspace(0,x,H0n))
                ax1.set_yticklabels(lEmaxlabelstr)
                ax1.set_xticklabels(H0labelstr)
                plt.sca(acb)
                
                yonly=np.sum(data1,axis=0)
                xonly=np.sum(data1,axis=1)
                cbar=plt.colorbar(p1,fraction=0.046, shrink=1.2,aspect=20,pad=0.00,cax = acb)
                cbar.set_label('loglik')
                axy.plot(xonly,lEmaxnew)  
                axy.set_yticks(lEmaxiter)
                axy.set_yticklabels(lEmaxlabelstr)
                
                axx.plot(H0new,yonly)
                axx.set_xticks(H0iter)
                axx.set_xticklabels(H0labelstr)
                plt.title('For survey '+str(i)+' imshow_interp')
                #plt.show()
                plt.close()
        

    np.save('lsurvey_lEmax_H0'+str(i), lsurvey)
    print ("lsurvey")
    

    cumlikeli = np.sum(lsurvey, axis=0)
    cshape=np.shape(cumlikeli)
    
    newcumlikeli=np.zeros(cshape)
    H0minimas=[]
    lEmaxminimas=[]
    
    for b in range(np.shape(cumlikeli)[2]):
        cumlikelij=cumlikeli[:,:,b]
        shape=np.shape(cumlikelij)
        
        #for managing inf and nan values which show up at a rate of around 0.1 % for ICS
        cumlikelij=np.ndarray.flatten(cumlikelij)
        cumlikelijnoinf=cumlikelij[np.isfinite(cumlikelij)]
        val=np.min(cumlikelijnoinf)
        cumlikelij[~np.isfinite(cumlikelij)]=val
        cumlikelij=np.reshape(cumlikelij,shape)
        newcumlikeli[:,:,b]=cumlikelij
        
        Xn, Yn = np.meshgrid(lEmaxiter, H0iter)
        
        f= bisplrep(Xn, Yn, cumlikelij, s=8)
        data1 = bisplev(lEmaxnew,H0new,f)
        data1=np.transpose(data1)
        
        if plotall:
            fig,ax=plt.subplots()
            p1=plt.imshow(cumlikelij, origin='lower')
            ax.set_yticks(np.arange(0,lEmaxn))
            ax.set_xticks(np.arange(0,H0n))
            ax.set_yticklabels(lEmaxlabelstr)
            ax.set_xticklabels(H0labelstr)
            cbar=plt.colorbar()
            cbar.set_label('loglik')
            plt.title('2D coarse likelihood grid')
            plt.show()
            plt.close()
     
            ################
            
            plt.figure(1, figsize=(8, 8))
            left, width = 0.1, 0.65
            bottom, height = 0.1, 0.65
            gap=0.11
            woff=width+gap+left
            hoff=height+gap+bottom
            dw=1.-woff-gap
            dh=1.-hoff-gap
            gap=0.11
            
            rect_2D = [left, bottom, width, height]
            rect_1Dx = [left, hoff, width, dh+gap]
            rect_1Dy = [woff-0.03, bottom, dw+gap, height]
            rect_cb = [woff,hoff-0.05,dw*0.95,dh+0.15]
            ax1=plt.axes(rect_2D)
            axx=plt.axes(rect_1Dx)
            axy=plt.axes(rect_1Dy)
            acb=plt.axes(rect_cb)
            axy.set_yticklabels([])
            
            plt.sca(ax1)
            p1=plt.imshow(data1,interpolation='none', origin='lower')
            ax1.set_yticks(np.linspace(0,y,lEmaxn))
            ax1.set_xticks(np.linspace(0,x,H0n))
            ax1.set_yticklabels(lEmaxlabelstr)
            ax1.set_xticklabels(H0labelstr)
            plt.sca(acb)
            
            yonly=np.sum(data1,axis=0)
            xonly=np.sum(data1,axis=1)
            cbar=plt.colorbar(p1,fraction=0.046, shrink=1.2,aspect=20,pad=0.00,cax = acb)
            cbar.set_label('loglik')
            axy.plot(xonly,lEmaxnew)  
            axy.set_yticks(lEmaxiter)
            axy.set_yticklabels(lEmaxlabelstr)
            
            axx.plot(H0new,yonly)
            axx.set_xticks(H0iter)
            axx.set_xticklabels(H0labelstr)
            plt.title('2D likelihood grid')
            plt.show()
            plt.close()
        
        ###################
    
        
        maxllik=np.max(data1)
        maxllikindex=np.where(data1==maxllik)
        E,H = maxllikindex[0],maxllikindex[1]
        E,H = E[0],H[0]
        
        H0minimas.append(H0new[H])
        lEmaxminimas.append(lEmaxnew[E])
        
    #print (H0minimas)
    #print (lEmaxminimas)
    np.save('H0minimas',H0minimas)
    np.save('lEmaxminimas',lEmaxminimas)
    
    
    #print ("H0minima at lEmax 41.84 is ", H0new[np.argmax(data1[0])])

    #H0minimas = np.array(H0minimas)
    plt.hist(H0minimas, bins=15)
    plt.title("Histogram of H0 value for maximum likelihood")
    plt.xlabel("H0 value")
    plt.ylabel("p(H0)")
    plt.show()
    plt.close()

    lEmaxminimas = np.array(lEmaxminimas)
    plt.hist(lEmaxminimas, bins=15)
    plt.title("Histogram of best-fit llEmax value for MC sample")
    plt.xlabel("H0 value")
    plt.ylabel("p(H0)")
    plt.show()
    plt.close()
    
    sns.kdeplot(H0minimas, bw=1)
    plt.title("Histogram of best-fit H0 value for MC sample")
    plt.xlabel("H0 value")
    plt.ylabel("p(H0)")
    plt.show()
    plt.close()
    
    mean=np.mean(H0minimas)
    std=np.std(H0minimas)
    var=np.var(H0minimas)

    print ("H0",mean,std,var)
    
    mean=np.mean(lEmaxminimas)
    std=np.std(lEmaxminimas)
    var=np.var(lEmaxminimas)

    print ("lEmax",mean,std,var)
    
    ########################################
    
    newcumlikeli=np.array(newcumlikeli)
    sumlik = np.sum(newcumlikeli, axis=2)
    
    shape=np.shape(sumlik)
    Xn, Yn = np.meshgrid(lEmaxiter, H0iter)
    f= bisplrep(Xn, Yn, sumlik, s=8)
    data1 = bisplev(lEmaxnew,H0new,f)
    data1=np.transpose(data1)
    
    
    fig,ax=plt.subplots()
    p1=plt.imshow(sumlik, origin='lower')
    ax.set_yticks(np.arange(0,lEmaxn))
    ax.set_xticks(np.arange(0,H0n))
    ax.set_yticklabels(lEmaxlabelstr)
    ax.set_xticklabels(H0labelstr)
    cbar=plt.colorbar()
    cbar.set_label('loglik')
    plt.title('cumulative 2D coarse likelihood grid')
    plt.show()
    plt.close()
    
    ###################
    
    plt.figure(1, figsize=(8, 8))
    left, width = 0.1, 0.65
    bottom, height = 0.1, 0.65
    gap=0.11
    woff=width+gap+left
    hoff=height+gap+bottom
    dw=1.-woff-gap
    dh=1.-hoff-gap
    gap=0.11
    
    rect_2D = [left, bottom, width, height]
    rect_1Dx = [left, hoff, width, dh+gap]
    rect_1Dy = [woff-0.03, bottom, dw+gap, height]
    rect_cb = [woff,hoff-0.05,dw*0.95,dh+0.15]
    ax1=plt.axes(rect_2D)
    axx=plt.axes(rect_1Dx)
    axy=plt.axes(rect_1Dy)
    acb=plt.axes(rect_cb)
    axy.set_yticklabels([])
    
    plt.sca(ax1)
    p1=plt.imshow(data1,interpolation='none', origin='lower')
    ax1.set_yticks(np.linspace(0,y,lEmaxn))
    ax1.set_xticks(np.linspace(0,x,H0n))
    ax1.set_yticklabels(lEmaxlabelstr)
    ax1.set_xticklabels(H0labelstr)
    plt.sca(acb)
    
    yonly=np.sum(data1,axis=0)
    xonly=np.sum(data1,axis=1)
    cbar=plt.colorbar(p1,fraction=0.046, shrink=1.2,aspect=20,pad=0.00,cax = acb)
    cbar.set_label('loglik')
    axy.plot(xonly,lEmaxnew)  
    axy.set_yticks(lEmaxiter)
    axy.set_yticklabels(lEmaxlabelstr)
    
    axx.plot(H0new,yonly)
    axx.set_xticks(H0iter)
    axx.set_xticklabels(H0labelstr)
    plt.title('cumulative 2D likelihood grid')
    plt.show()
    plt.close()
    
    #############################


MC_H0_lEmax(100)


