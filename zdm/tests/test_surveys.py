""" Generate, load, etc. Surveys """
import os
import numpy as np
from pkg_resources import resource_filename

import pandas
from astropy.table import Table
from astropy.cosmology import Planck18 

from zdm import survey 
from zdm import survey_data 
from zdm import parameters 
from zdm import cosmology as cos
from zdm.tests import tstutils 
from zdm import misc_functions

import pytest

from IPython import embed

# TODO
#  Update all TNS
#  Confirm THRESH and SNRTHESH
#  Confirm X columns
#  SNRth in Parkes
#  Move all of init into __init__()
#  Refactor the rest of Survey

def run_all():
    """
    Here to allow local tests of all the test routines in this file
    """
    test_load_new_grid()
    test_load_new_survey()
    test_load_old()
    test_refactor()

def test_load_new_grid():
    state = parameters.State()
    # Cosmology
    cos.set_cosmology(state)
    cos.init_dist_measures()
    zDMgrid, zvals,dmvals = misc_functions.get_zdm_grid(
        state, new=True, plot=False, method='analytic', 
        nz=500, datdir=resource_filename('zdm', 'GridData'))
    # Survey
    isurvey = survey.load_survey('CRAFT/FE', state, dmvals)
    # Grid
    grids = misc_functions.initialise_grids(
        [isurvey], zDMgrid, zvals, dmvals, state, wdist=True)

    # Test?
    old_survey = survey.load_survey('CRAFT/FE', state,
                         dmvals, original=True)
    old_grids = misc_functions.initialise_grids(
        [old_survey], zDMgrid, zvals, dmvals, state, wdist=True)
    assert np.all(np.isclose(old_grids[0].rates, grids[0].rates))

def test_load_new_survey():
    state = parameters.State()
    isurvey = survey.load_survey('CRAFT/FE', state,
                        np.linspace(0., 2000., 1000))
    # Original
    old_survey = survey.load_survey('CRAFT/FE', state,
                         np.linspace(0., 2000., 1000),
                         original=True)
    
    # Test
    assert np.all(np.isclose(old_survey.efficiencies, 
                             isurvey.efficiencies))

def test_load_old():
    # Load state
    state = parameters.State()

    # FE
    isurvey = survey.load_survey('CRAFT/FE', state,
                         np.linspace(0., 2000., 1000),
                         original=True)
    assert isurvey.name == 'CRAFT_class_I_and_II'

def test_refactor():
    """
    This test writes a new ecsv, then compares this
    "new" survey with the old one.
    """
    outfile = tstutils.data_path('tmp.ecsv')
    if os.path.isfile(outfile):
        os.remove(outfile)
    # Generate
    survey.refactor_old_survey_file('CRAFT/FE', outfile)
    
    srvy = survey.load_survey('tmp', 
                        parameters.State(),
                         np.linspace(0., 2000., 1000),
                        sdir=tstutils.data_path(''))
    
    # Test
    assert np.isclose(srvy.TOBS, 1274.6) 

    # Clean up
    os.remove(outfile)
    
#run_all()

#def test_gmrt():

zmax = 7.
state = parameters.State()
param_dict={'sfr_n': 0.21, 'alpha': 0.11, 'lmean': 2.18, 'lsigma': 0.42, 'lEmax': 41.37, 
                'lEmin': 39.47, 'gamma': -1.04, 'H0': 70.23, 'halo_method': 0, 'sigmaDMG': 0.0, 'sigmaHalo': 0.0,
                'lC': -7.61, 'min_lat': 0.0}
state.set_astropy_cosmo(Planck18)
state.update_params(param_dict)

# Cosmology
cos.set_cosmology(state)
cos.init_dist_measures()
zDMgrid, zvals,dmvals = misc_functions.get_zdm_grid(
    state, new=True, plot=False, method='analytic', 
    zmax=zmax,
    nz=700, datdir=resource_filename('zdm', 'GridData'))
# Survey
isurvey = survey.load_survey('GMRT_band4', state, dmvals)
#isurvey = survey.load_survey('DSA', state, dmvals)
# Grid
grids = misc_functions.initialise_grids(
    [isurvey], zDMgrid, zvals, dmvals, state, wdist=True)

g = grids[0]
misc_functions.plot_grid_2(
            g.rates,
            g.zvals,
            g.dmvals,
            name="pretty_GMRT_plot.png",
            norm=3,
            log=True,
            label="$\\log_{10} p({\\rm DM}_{\\rm EG},z)$ [a.u.]",
            project=False,
            Aconts=[0.01, 0.1, 0.5],
            zmax=zmax,
            DMmax=4500
        )
# Plot
#embed(header="GMRT")