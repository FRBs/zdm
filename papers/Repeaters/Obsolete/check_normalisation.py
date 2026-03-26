""" 
This script iterates over a grid in Rgamma and Rmax. For each, it:
- calculates Rstar, such that all FRBs repeating at Rstar (=Rmin=Rmax)
    reproduce the correct repetition rate (Rgamma not important)
- iterates over Rgamma, Rmax, to find the Rmin reproducing the correct rate
- simulates the z/DM and declination distribution of repeating CHIME FRBs,
    and calculates likelihood metrics
- saves the resulting output


By default it assumes that 100% of CHIME bursts come from repeaters.
However, you can set FC to vary this "Fraction" to be less than unity.

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
from zdm import energetics

import utilities as ute
import states as st

import pickle
import numpy as np
from zdm import survey
from matplotlib import pyplot as plt

import scipy as sp
from scipy.stats import poisson

import matplotlib
import time
#from zdm import beams
#beams.beams_path = '/Users/cjames/CRAFT/FRB_library/Git/H0paper/papers/Repeaters/BeamData/'

Planck_H0=67.4

matplotlib.rcParams['image.interpolation'] = None

defaultsize=14
ds=4
font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : defaultsize}
matplotlib.rc('font', **font)

def main():
    names = ['CRAFT_class_I_and_II','parkes_mb_class_I_and_II','private_CRAFT_ICS','private_CRAFT_ICS_892','private_CRAFT_ICS_1632']
    
    # gets the possible states for evaluation
    
    psets=st.read_extremes()
    psets.insert(0,st.james_fit())
    states=[]
    for i,pset in enumerate(psets):
        state=st.set_state(pset,chime_response=False)
        states.append(state)
    
    #states,titles=st.get_states(ischime=False)
    sdir='../../zdm/data/Surveys/'
    print("sdir is ",sdir)
    for i,state in enumerate(states):
        state.beam.Bmethod=2 #otherwise, leads to death
        expecteds=[]
        actuals = []
        grids = []
        surveys=[]
        for j,name in enumerate(names):
            
            s,g = loading.survey_and_grid(
                survey_name=name,NFRB=None,sdir=sdir,init_state=state)
            #print("Did survey ",name,i,j)
            expected = np.sum(g.rates)*s.TOBS*(10**state.FRBdemo.lC)
            expecteds.append(expected)
            actuals.append(s.NORM_FRB)
            grids.append(g)
            surveys.append(s)
        
        print("state ",i,"expects ",expecteds," has ",actuals)
        newC,llC = it.minimise_const_only(None,grids,surveys)
        oldC=g.state.FRBdemo.lC
        print("      New constant is ",newC," c.f. ",oldC)
        state.FRBdemo.lC = newC
        meaningful = it.ConvertToMeaningfulConstant(state)
        print("Number above Eref is ",meaningful)
        print("in terms of FRBs/year, not day", meaningful*365)
        print("as log10 ",np.log10(meaningful*365))
        
        with open("changing_constants.txt", "a") as myfile:
            myfile.write(str(i)+"   "+str(newC)+"   "+str(oldC)+"   "+str(meaningful)+"\n")
        
main()
