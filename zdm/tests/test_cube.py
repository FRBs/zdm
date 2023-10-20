"""
File to test that cubing produces the expected output
"""

import os
import pytest

from pkg_resources import resource_filename
import pandas

from zdm import iteration as it
from zdm import real_loading
from zdm import io
from zdm.tests import tstutils

import numpy as np

from IPython import embed

def check_accuracy(x1,x2,thresh=1e-4):
    diff=x1-x2
    mean=0.5*(x1+x2)
    if np.abs(diff/mean) < thresh:
        return True
    else:
        return False

def test_cube_run():
    # use default parameters
    # Initialise survey and grid 
    # For this purporse, we only need two different surveys
    # the defaults are below - passing these will do nothing
    survey_names = ['CRAFT/FE', 
                    'CRAFT_ICS_1632',
                    'CRAFT_ICS_892', 
                    'CRAFT_ICS_1272',
                    'PKS/Mb']
    #sdir = os.path.join(resource_filename('zdm', 'data'), 'Surveys')
    #surveys=[]
    #grids=[]

    '''
    # We should be using real_loading
    for name in names:
        s,g = loading.survey_and_grid(
            survey_name=name,NFRB=None,sdir=sdir) # should be equal to actual number of FRBs, but for this purpose it doesn't matter
        surveys.append(s)
        grids.append(g)
    '''
    surveys, grids = real_loading.surveys_and_grids(
        nz=500, ndm=1400) # Small number to keep this cheap
    
    
    ### gets cube files
    pfile= tstutils.data_path('real_mini_cube.json')
    input_dict=io.process_jfile(pfile)
    
    # Deconstruct the input_dict
    state_dict, cube_dict, vparam_dict = it.parse_input_dict(input_dict)
    
    outfile= tstutils.data_path('output.csv')
    
    # check that the output file does not already exist
    #assert not os.path.exists(outfile)
    
    run=1
    howmany=1
    #run=2 + 2*4 + 5*4*4 + 2*10*4*4 + 1*4*10*4*4 + 5*3*4*10*4*4 + 5*10*3*4*10*4*4
    # I have chosen this to pick normal values of most parameters, and a high Emax
    # should prevent any nans appearing...
    #run = 5 + 1*10 + 9*3*10 + 1*10*3*10 + 1*4*10*3*10 + 1*4*4*10*3*10 + 6*4*4*10*3*10

    # Set cube shape
    ####### counters for each dimensions ######
    parameter_order = cube_dict['parameter_order']
    PARAMS = list(vparam_dict.keys())
    order, iorder = it.set_orders(parameter_order, PARAMS)

    # Shape of the grid (ignoring the constant, lC)
    cube_shape = it.set_cube_shape(vparam_dict, order)
    # Choose parameters in the middle to avoid NANs
    current = [item//2 for item in cube_shape]
    run = np.ravel_multi_index(current, cube_shape, order='F')

    #pytest.set_trace()
    it.cube_likelihoods(grids,surveys,vparam_dict, cube_dict,run,howmany,outfile)
    
    # now we check that the output file exists
    #assert os.path.exists(outfile)
    
    # output is
    # ith run (counter)
    # variables: 8 of these (Emax, H0, alpha, gamma, n, lmean, lsigma, C)
    # 5 pieces of info for each surveys (ll_survey, Pn, Ps, Pzdm, expected_N)
    # 8 pieces of info at the end (lltot, Pntot, Ps_tot, Pzdm_tot,
    #    pDM|z_tot, pz_tot,pz|DM_tot, pDM_tot
    # =18+Nsurveys*5
    
    ns=len(surveys)

    # Load with pandas to assess
    df = pandas.read_csv(outfile)
    ds = df.iloc[0]

    # lls
    lltot_v0= ds.lls
    lltot_v1=0
    for j in np.arange(ns):
        lltot_v1 += ds[f'lls{j}']
    lltot_v2= np.sum(
        [ds[key] for key in ['P_zDM', 'P_n', 'P_s']])
    assert check_accuracy(lltot_v0,lltot_v1)
    assert check_accuracy(lltot_v0,lltot_v2)

    # zdm
    zdm_v1= ds.p_zgDM + ds.p_DM
    zdm_v2= ds.p_DMgz + ds.p_z
    assert check_accuracy(zdm_v1,zdm_v2)
    

#test_cube_run()
