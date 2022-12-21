""" Generate, load, etc. Surveys """
import os
import numpy as np
from pkg_resources import resource_filename

import pandas
from astropy.table import Table

from zdm import survey 
from zdm import survey_data 
from zdm import parameters 
from zdm import cosmology as cos
from zdm.tests import tstutils 
from zdm import misc_functions

import pytest

# TODO
#  Update all TNS
#  Confirm THRESH and SNRTHESH
#  Confirm X columns
#  SNRth in Parkes
#  Move all of init into __init__()
#  Refactor the rest of Survey

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
    outfile = tstutils.data_path('tmp.ecsv')
    if os.path.isfile(outfile):
        os.remove(outfile)
    # Generate
    survey.refactor_old_survey_file('CRAFT/FE', outfile)
    # Read
    frb_tbl = Table.read(outfile,
                        format='ascii.ecsv')
    srvy_data = survey_data.SurveyData.from_jsonstr(
        frb_tbl.meta['survey_data'])
    frbs = frb_tbl.to_pandas()

    # Vet
    survey.vet_frb_table(frbs, mandatory=True)

    # Test
    assert np.isclose(srvy_data.observing.TOBS, 1274.6) 

    # Clean up
    os.remove(outfile)
    
