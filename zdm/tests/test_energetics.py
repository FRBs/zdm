""" Tests of the energetics.py module """

import numpy as np
import pytest

from zdm import energetics

def test_init_gamma():

    # Run
    energetics.init_igamma_linear([-1.], log=False)

    # Test
    assert -1. in energetics.igamma_linear.keys()
    assert np.isclose(float(
        energetics.igamma_linear[-1](1.)), 0.14860105, atol=2e-4)

    # Run with log
    energetics.init_igamma_linear([-1.], log=True)

    assert np.isclose(float(energetics.igamma_linear_log10[-1](0.)), 
                      float(energetics.igamma_linear[-1](1.)),
                      rtol=1e-3)
                      

