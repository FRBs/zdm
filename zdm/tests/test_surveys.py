""" Generate, load, etc. Surveys """
import numpy as np

import pandas
from astropy.table import Table

from zdm.survey import load_survey
from zdm import parameters 

import pytest

def test_load_old():
    # Load state
    state = parameters.State()

    # FE
    survey = load_survey('CRAFT/FE', state,
                         np.linspace(0., 2000., 1000))
    assert survey.name == 'CRAFT_class_I_and_II'
