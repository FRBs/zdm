
import pytest

from zdm import cosmology as cos
from zdm import misc_functions
from zdm import parameters

from IPython import embed

def test_make_grids():

    ############## Initialise parameters ##############
    params = parameters.init_parameters()


    ############## Initialise cosmology ##############
    cos.set_cosmology(params)
    cos.init_dist_measures()


    # get the grid of p(DM|z). See function for default values.
    # set new to False once this is already initialised
    zDMgrid, zvals,dmvals,H0 = misc_functions.get_zdm_grid(
        params['cosmo'].H0, new=True, plot=False, method='analytic')
    pytest.set_trace()

