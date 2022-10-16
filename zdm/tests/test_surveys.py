""" Generate, load, etc. Surveys """
import numpy as np

import pandas
from astropy.table import Table

from zdm import survey 
from zdm import parameters 

import pytest

survey.refactor_old_survey_file('CRAFT/FE', 'tmp.ecsv')

def test_load_old():
    # Load state
    state = parameters.State()

    # FE
    isurvey = survey.load_survey('CRAFT/FE', state,
                         np.linspace(0., 2000., 1000))
    assert isurvey.name == 'CRAFT_class_I_and_II'
