"""
This file is intended to test the computing performance
when updating grids for a cube evaluation.

It tackles the following questions:
- How long does it take to update a grid for parameter X?
  (how much longer for additional grids beyond the first?)
- What is the optimal parameter order for cubing?
  (is this changed by the number of grids used in the cube?)
- Is the grid update method accurate?
  (and is it still accurate when using the many-cube shortcut?)

"""

import pytest

from pkg_resources import resource_filename
import os
import copy
import pickle

from astropy.cosmology import Planck18

from zdm import cosmology as cos
from zdm import misc_functions
from zdm import parameters
from zdm import survey
from zdm import iteration as it
from zdm.craco import loading
from zdm import io

from IPython import embed

import time
import numpy as np

def main(likelihoods=True,detail=0,verbose=True):
    
    ############## Load up ##############
    input_dict=io.process_jfile('../../../papers/H0_I/Analysis/Cubes/craco_H0_Emax_state.json')

    # Deconstruct the input_dict
    state_dict, cube_dict, vparam_dict = it.parse_input_dict(input_dict)
    
    
    ############## Initialise survey and grid ##############
    # For this purporse, we only need two different surveys
    names = ['CRAFT/FE', 'PKS/Mb']
    #sdir=None # use this to direct to a different survey directory
    
    #names=['CRAFT/FE', 'PKS/Mb','private_CRAFT_ICS','private_CRAFT_ICS_892','private_CRAFT_ICS_1632']
    sdir='../../data/Surveys/'
    
    print("Performing calculations for ",names)
    
    surveys=[]
    grids=[]
    for name in names:
        s,g = loading.survey_and_grid(
            state_dict=state_dict,sdir=sdir,
            survey_name=name,NFRB=None) # should be equal to actual number of FRBs, but for this purpose it doesn't matter
        surveys.append(s)
        grids.append(g)
    
    ################ set up parameters to vary #############
    # retrieves a state
    state=grids[0].state
    #list of parameters to vary
    plist=['H0','F','lEmin','lEmax','gamma','alpha','sfr_n','lmean','lsigma']
    # original values for the loop
    ovals=[state.cosmo.H0,state.IGM.F,
        state.energy.lEmin,state.energy.lEmax,state.energy.gamma,state.energy.alpha,
        state.FRBdemo.sfr_n,
        state.host.lmean,state.host.lsigma]
    
    ################ updates parameters and measures the time ##########
    # number of repetitions for updating - use this to reduce variance is required
    NREPS=1
    for i,param in enumerate(plist):
        print("\n\n########## Timing update of ",param," ########")
        t0=time.time()
        tlik=0.
        tmin=0.
        for j in np.arange(NREPS):
            t1=time.time()
            # set up new parameters
            vparams = {}
            vparams[param] = ovals[i]*(1.09**(j+1)) #increase by 10 % - arbitrary!
            for j,g in enumerate(grids):
                g.update(vparams)
            t2=time.time()
            if likelihoods:
                norm=True
                psnr=True
                
                # these two options should give identical answers
                C1,llC,lltot=it.minimise_const_only(None,grids,surveys,use_prev_grid=False)
                C2,llC,lltot=it.minimise_const_only(vparams,grids,surveys,use_prev_grid=False)
                if np.abs((C2-C1)/(C2+C1)) > 1e-3:
                    print("Two minimisations give different values of the constant! ",C1," and ",C2)
                t3=time.time() #times the minimisation routine
                tmin += t3-t2
                for j,g in enumerate(grids):
                    s=surveys[j]
                    if s.nD==1:
                        results = it.calc_likelihoods_1D(
                        grids[j],s,norm=norm,psnr=psnr,dolist=5)
                    elif s.nD==2:
                        results = it.calc_likelihoods_2D(
                        grids[j],s,norm=norm,psnr=psnr,dolist=5)
                    elif s.nD==3:
                        # mixture of 1 and 2D samples. NEVER calculate Pn twice!
                        results = it.calc_likelihoods_1D(
                            grids[j],s,norm=norm,psnr=psnr,dolist=5)
                        results = it.calc_likelihoods_2D(
                            grids[j],s,norm=norm,psnr=psnr,dolist=5,Pn=False)
                t4=time.time()
                tlik += t4-t3
            if verbose:
                print("    Iteration ",j," took ",t2-t1)
        t5=time.time()
        mean_single=(t5-t0)/NREPS
        tlik = tlik/NREPS
        tmin = tmin/NREPS/2.
        mean_single = mean_single - tlik - tmin
        print("    Updating the grids individually took ",mean_single," s per repetition")
        if likelihoods:
            print("        The minimisation of C took an additional",tmin," s")
            print("        The likelihood calculation took an additional",tlik," s")
        if detail > 0:
            test_update(grids,names)
        
        tlik=0.
        tmin=0.
        t0=time.time()
        for j in np.arange(NREPS):
            t1=time.time()
            vparams = {}
            vparams[param] = ovals[i]*(1.1**(j+1)) #increase by 10 %
            for j,g in enumerate(grids):
                if j==0:
                    g.update(vparams)
                else:
                    g.update(vparams,prev_grid=grids[0])
            t2=time.time()
            
            if likelihoods:
                norm=True
                psnr=True
                
                # these two options should give identical answers
                C1,llC,lltot=it.minimise_const_only(None,grids,surveys,use_prev_grid=True)
                C2,llC,lltot=it.minimise_const_only(vparams,grids,surveys,use_prev_grid=True)
                if np.abs((C2-C1)/(C2+C1)) > 1e-3:
                    print("Two minimisations give different values of the constant! ",C1," and ",C2)
                t3=time.time() #times the minimisation routine
                tmin += t3-t2
                for j,g in enumerate(grids):
                    s=surveys[j]
                    if s.nD==1:
                        results = it.calc_likelihoods_1D(
                        grids[j],s,norm=norm,psnr=psnr,dolist=5)
                    elif s.nD==2:
                        results = it.calc_likelihoods_2D(
                        grids[j],s,norm=norm,psnr=psnr,dolist=5)
                    elif s.nD==3:
                        # mixture of 1 and 2D samples. NEVER calculate Pn twice!
                        results = it.calc_likelihoods_1D(
                            grids[j],s,norm=norm,psnr=psnr,dolist=5)
                        results = it.calc_likelihoods_2D(
                            grids[j],s,norm=norm,psnr=psnr,dolist=5,Pn=False)
                t4=time.time()
                tlik += t4-t3
            
            if verbose:
                print("    Iteration ",j," took ",t2-t1,"\n\n")
        t5=time.time()
        mean_double=(t5-t0)/NREPS
        tlik = tlik/NREPS
        tmin = tmin/NREPS/2.
        mean_double = mean_double - tlik - tmin
        print("    Using prev_grid took ",mean_double," s per repetition")
        print("    This is a time saved of ",-mean_double+mean_single," s")
        if likelihoods:
            print("        The minimisation of C took an additional",tmin," s")
            print("        The likelihood calculation took an additional",tlik," s")
        if detail > 0:
            test_update(grids,names,detail)
        # now generates a new parameter set, and checks that the result is equal
        # to the updated one
    
    ####### Check with a fresh frid ######
    if detail == 0:
        print("Checking update accuracy")
        test_update(grids,names,sdir,detail=1)
    
def test_update(grids,names,sdir,detail=1):
    """
    Compares the grids to fresh grids made using their state parameters
    
    grids:
        list of grid objects
    
    names:
        names of surveys the grids are loaded from
    
    detail:
        int: level of detail to report information
    """
    # create new grids directly from updates state parameters in grid   
    newgrids=[]
    newsurveys=[]
    for name in names:
        s,g = loading.survey_and_grid(
            init_state=grids[0].state, sdir=sdir,
            survey_name=name,NFRB=None) # should be equal to actual number of FRBs, but for this purpose it doesn't matter
        newsurveys.append(s)
        newgrids.append(g)
    
    
    # test that the resulting rates are sufficiently similar
    for i,g in enumerate(grids):
        newg=newgrids[i]
        diff=g.rates-newg.rates
        #rel_diff=0.5*diff/(g.rates+newg.rates)
        print("    For grid ",i," max diff of rates is ",np.max(diff)," from maxes of ",np.max(newg.rates))
        #print("Max relative differences that's not NAN is ",np.nanmax(rel_diff)
        
        if detail==1:
            pass
        diff=g.pdv-newg.pdv
        print("    For grid ",i," max diff of pdv is ",np.max(diff)," from maxes of ",np.max(newg.pdv))
        
        diff=g.dV-newg.dV
        print("    For grid ",i," max diff of dv is ",np.max(diff)," from maxes of ",np.max(newg.dV))
        
        diff=g.fractions-newg.fractions
        print("    For grid ",i," max diff of fractions is ",np.max(diff)," from maxes of ",np.max(newg.fractions))
        
        # widths might be different shapes
        wmin=min(g.thresholds.shape[0],newg.thresholds.shape[0])
        diff=g.thresholds[-wmin:,:,:]-newg.thresholds[-wmin:,:,:]
        print("    For grid ",i," max diff of thresholds is ",np.max(diff)," from maxes of ",np.max(newg.thresholds))
        
        diff=g.sfr_smear-newg.sfr_smear
        print("    For grid ",i," max diff of sfr_smear is ",np.max(diff)," from maxes of ",np.max(newg.sfr_smear))
        
        diff=g.sfr-newg.sfr
        print("    For grid ",i," max diff of sfr is ",np.max(diff)," from maxes of ",np.max(newg.sfr))
        
        diff=g.smear_grid-newg.smear_grid
        print("    For grid ",i," max diff of smear_grid is ",np.max(diff)," from maxes of ",np.max(newg.smear_grid))
        if detail==2:
            pass
        for k,ig in enumerate([g,newg]):
            print("Info for grid ",k)
            print(ig.state.energy.alpha)
            print(ig.nuObs)
            print(ig.nuRef)
            print(ig.state.FRBdemo.alpha_method)
            print(ig.state.FRBdemo.sfr_n)
        
    
main()        
