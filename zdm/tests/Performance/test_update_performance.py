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

def main(detail=0):
    
    ############## Load up ##############
    input_dict=io.process_jfile('../../../papers/H0_I/Analysis/Cubes/craco_H0_Emax_state.json')

    # Deconstruct the input_dict
    state_dict, cube_dict, vparam_dict = it.parse_input_dict(input_dict)
    
    
    ############## Initialise survey and grid ##############
    # For this purporse, we only need two different surveys
    names = ['CRAFT/FE', 'PKS/Mb']
    surveys=[]
    grids=[]
    for name in names:
        s,g = loading.survey_and_grid(
            state_dict=state_dict,
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
    # number of repetitions for updating - just to reduce variance
    NREPS=3
    for i,param in enumerate(plist):
        print("\n\n########## Timing update of ",param," ########")
        t0=time.time()
        for j in np.arange(NREPS):
            t1=time.time()
            # set up new parameters
            vparams = {}
            vparams[param] = ovals[i]*(1.1**(j+1)) #increase by 10 % - arbitrary!
            grids[0].update(vparams)
            grids[1].update(vparams)
            t2=time.time()
            #print("    Iteration ",j," took ",t2-t1)
        t3=time.time()
        mean_single=(t3-t0)/NREPS
        print("    Updating two grids individually took ",mean_single," s per repetition")
        if detail > 0:
            test_update(grids,names)
        
        t0=time.time()
        for j in np.arange(NREPS):
            t1=time.time()
            vparams = {}
            vparams[param] = ovals[i]*(1.1**(j+1)) #increase by 10 %
            grids[0].update(vparams)
            grids[1].update(vparams,prev_grid=grids[0])
            t2=time.time()
            #print("    Iteration ",j," took ",t2-t1,"\n\n")
        t3=time.time()
        mean_double=(t3-t0)/NREPS
        print("    Using prev_grid took ",mean_double," s per repetition")
        print("    This is a time saved of ",-mean_double+mean_single," s per grid")
        if detail > 0:
            test_update(grids,names,detail)
        # now generates a new parameter set, and checks that the result is equal
        # to the updated one
    
    ####### Check with a fresh frid ######
    if detail == 0:
        print("Checking update accuracy")
        test_update(grids,names,detail=1)
    
def test_update(grids,names,detail=1):
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
            init_state=grids[0].state,
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
        
        diff=g.thresholds-newg.thresholds
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
