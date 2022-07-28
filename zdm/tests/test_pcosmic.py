import pytest

import numpy as np

from astropy.cosmology import Planck18

from zdm import parameters
from zdm import pcosmic


def test_mean_DM():
    """ Test that DM_cosmic is as we expect it
    This also effectively tests the FRB Repo
    """
    # Set the state
    state = parameters.State()
    state.set_astropy_cosmo(Planck18)

    # Calculate
    zmax, nz = 1., 1000
    dz=zmax/nz
    zvals=(np.arange(nz)+1)*dz
    DMs = pcosmic.get_mean_DM(zvals, state)

    # Test
    assert np.isclose(DMs[-1], 924.81566918)
    