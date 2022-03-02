#import pytest

#from pkg_resources import resource_filename
import os
#import copy
#import pickle

#from astropy.cosmology import Planck18

#from zdm import cosmology as cos
#from zdm import misc_functions
from zdm import parameters
#from zdm import survey
#from zdm import pcosmic
from zdm import iteration as it
from zdm.craco import loading
from zdm import io


#from IPython import embed

import numpy as np
from zdm import survey
from matplotlib import pyplot as plt

def main():
    # use default parameters
    # Initialise survey and grid 
    # For this purporse, we only need two different surveys
    #names=['CRAFT/FE','CRAFT/ICS','CRAFT/ICS892','CRAFT/ICS1632','PKS/Mb']
    names=['CRAFT/FE','CRAFT/ICS','CRAFT/ICS892','PKS/Mb']
    sdir='../data/Surveys/'
    surveys=[]
    grids=[]
    for name in names:
        s,g = loading.survey_and_grid(
            survey_name=name,NFRB=None,sdir=sdir) # should be equal to actual number of FRBs, but for this purpose it doesn't matter
        surveys.append(s)
        grids.append(g)
    
    
    ### gets cube files
    pfile='../../papers/H0_I/Analysis/Real/Cubes/real_mini_cube.json'
    input_dict=io.process_jfile(pfile)
    
    # Deconstruct the input_dict
    state_dict, cube_dict, vparam_dict = it.parse_input_dict(input_dict)
    
    outfile='output.csv'
    
    # check that the output file does not already exist
    #assert not os.path.exists(outfile)
    
    run=1
    howmany=1
    #run=2 + 2*4 + 5*4*4 + 2*10*4*4 + 1*4*10*4*4 + 5*3*4*10*4*4 + 5*10*3*4*10*4*4
    # I have chosen this to pick normal values of most parameters, and a high Emax
    # should prevent any nans appearing...
    run = 5 + 1*10 + 9*3*10 + 1*10*3*10 + 1*4*10*3*10 + 1*4*4*10*3*10 + 6*4*4*10*3*10
    it.cube_likelihoods(grids,surveys,vparam_dict,cube_dict,run,howmany,outfile)
    
    # now we check that the output file exists
    #assert os.path.exists(outfile)
    
    # output is
    # ith run (counter)
    # variables: 8 of these (Emax, H0, alpha, gamma, n, lmean, lsigma, C)
    # 5 pieces of info for each surveys (ll_survey, Pn, Ps, Pzdm, expected_N)
    # 8 pieces of info at the end (lltot, Pntot, Ps_tot, Pzdm_tot,
    #    pDM|z_tot, pz_tot,pz|DM_tot, pDM_tot
    # =18+Nsurveys*5
    
    ns=len(names)
    expected=17+ns*5
    
    # now check it has the right dimensions
    with open(outfile, 'r') as infile:
        lines=infile.readlines()
        assert len(lines)==howmany+1
        for i,line in enumerate(lines):
            if i==0:
                continue
            words=line.split(',')
            
            # three ways of calculating lltot
            lltot_v0=float(words[-8])
            lltot_v1=0
            for j in np.arange(ns):
                lltot_v1 += float(words[9+5*j])
            lltot_v2=float(words[-5])+float(words[-6])+float(words[-7])
            
            # three ways of calculating p(z,DM)
            zdm_v1=float(words[-3])+float(words[-4])
            zdm_v2=float(words[-1])+float(words[-2])
            
            assert check_accuracy(zdm_v1,zdm_v2)
            
            assert check_accuracy(lltot_v0,lltot_v1)
            assert check_accuracy(lltot_v0,lltot_v2)
            
def check_accuracy(x1,x2,thresh=1e-4):
    diff=x1-x2
    mean=0.5*(x1+x2)
    if np.abs(diff/mean) < thresh:
        return True
    else:
        return False
      
main()
