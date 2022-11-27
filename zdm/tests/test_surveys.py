""" Generate, load, etc. Surveys """
import os
import numpy as np

import pandas
from astropy.table import Table

from zdm import survey 
from zdm import survey_data 
from zdm import parameters 
from zdm.tests import tstutils 

import pytest


#def test_refactor():
# Generate
survey.refactor_old_survey_file(
    'CRAFT/FE', tstutils.data_path('tmp.ecsv'))
# Read
tbl = Table.read(tstutils.data_path('tmp.ecsv'), 
                    format='ascii.ecsv')
srvy_data = survey_data.SurveyData.from_jsonstr(
    tbl.meta['survey_data'])
# Test
assert np.isclose(srvy_data.observing.TOBS, 1274.6) 

# Clean up
os.remove(tstutils.data_path('tmp.ecsv'))
    
    

def test_load_old():
    # Load state
    state = parameters.State()

    # FE
    isurvey = survey.load_survey('CRAFT/FE', state,
                         np.linspace(0., 2000., 1000))
    assert isurvey.name == 'CRAFT_class_I_and_II'
