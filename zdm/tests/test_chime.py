"""
File to test CHIME
"""

import os
import pytest

from pkg_resources import resource_filename
import pandas

from zdm.tests import tstutils

from zdm.chime import grids

def test_run():
    dmvals, zvals, all_rates, all_singles, all_reps = grids.load()

