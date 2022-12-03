""" Generate, load, etc. Surveys """
import os
import numpy as np
from pkg_resources import resource_filename

import pandas
from astropy.table import Table

from zdm import survey 
from zdm import survey_data 
from zdm import parameters 
from zdm.tests import tstutils 

import pytest

sdir = os.path.join(resource_filename('zdm', 'data'), 
                        'Surveys', 'Original')

def test_load_old():
    # Load state
    state = parameters.State()

    # FE
    isurvey = survey.load_survey('CRAFT/FE', state,
                         np.linspace(0., 2000., 1000),
                         sdir=sdir)
    assert isurvey.name == 'CRAFT_class_I_and_II'

def test_refactor():
    outfile = tstutils.data_path('tmp.ecsv')
    if os.path.isfile(outfile):
        os.remove(outfile)
    # Generate
    survey.refactor_old_survey_file(
        'CRAFT/FE', outfile, sdir=sdir)
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
    
