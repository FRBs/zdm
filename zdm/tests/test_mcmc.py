"""
This is a function to test if emcee is running on your computer

It defines a simple probability (log_prob) and 
submits this as a pooled job over your cpus.

Original test code taken from
https://emcee.readthedocs.io/en/stable/tutorials/parallel/
Note a modification to allow emcee to work on mac osx
"""


import time
import numpy as np
import emcee
import os
import multiprocessing as mp
from multiprocessing import cpu_count

from zdm import MCMC
from zdm import grid
from zdm import parameters


def log_prob(theta):
    t = time.time() + np.random.uniform(0.005, 0.008)
    while True:
        if time.time() >= t:
            break
    return -0.5 * np.sum(theta**2)
    

def test_mcmc():
    np.random.seed(42)
    initial = np.random.randn(32, 5)
    nwalkers, ndim = initial.shape
    nsteps = 20

    os.environ["OMP_NUM_THREADS"] = "1"

    # this mod is required for running on mac osx
    Pool = mp.get_context('fork').Pool

    ncpu = cpu_count()
    print("{0} CPUs".format(ncpu))

    with Pool() as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob,pool=pool)
        start = time.time()
        sampler.run_mcmc(initial, nsteps, progress=True)
        end = time.time()
        serial_time = end - start
        print("MP took {0:.1f} seconds".format(serial_time))


def test_broken_power_law_joint_prior():
    state = parameters.State()
    state.energy.luminosity_function = 4
    state.energy.lEmin = 38.0
    state.energy.lEb = 40.0
    state.energy.lEmax = 42.0

    assert MCMC.valid_parameter_combination({}, state)
    assert MCMC.valid_parameter_combination({'lEb': 41.0}, state)
    assert not MCMC.valid_parameter_combination({'lEb': 37.0}, state)
    assert not MCMC.valid_parameter_combination({'lEmin': 41.0}, state)
    assert not MCMC.valid_parameter_combination({'lEmax': 39.0}, state)

    posterior = MCMC.calc_log_posterior(
        [37.0],
        state,
        {'lEb': {'min': 35.0, 'max': 43.0}},
        [[], []],
    )
    assert posterior == -np.inf


def test_broken_power_law_initial_walkers_obey_joint_prior():
    state = parameters.State()
    state.energy.luminosity_function = 4
    params = {
        'lEmin': {'min': 37.0, 'max': 41.0},
        'lEb': {'min': 38.0, 'max': 42.0},
        'lEmax': {'min': 39.0, 'max': 43.0},
        'gamma': {'min': -3.0, 'max': 0.0},
        'gamma2': {'min': -5.0, 'max': 0.0},
    }

    walkers = MCMC.get_initial_walkers(
        state, params, nwalkers=64, rng=np.random.default_rng(1234)
    )
    indices = {name: i for i, name in enumerate(params)}

    assert np.all(
        walkers[:, indices['lEmin']]
        < walkers[:, indices['lEb']]
    )
    assert np.all(
        walkers[:, indices['lEb']]
        < walkers[:, indices['lEmax']]
    )


def test_double_broken_power_law_joint_prior():
    state = parameters.State()
    state.energy.luminosity_function = 5
    state.energy.lEmin = 37.0
    state.energy.lEb = 39.0
    state.energy.lEb2 = 41.0
    state.energy.lEmax = 43.0

    assert MCMC.valid_parameter_combination({}, state)
    assert MCMC.valid_parameter_combination({'lEb2': 42.0}, state)
    assert not MCMC.valid_parameter_combination({'lEb': 42.0}, state)
    assert not MCMC.valid_parameter_combination({'lEb2': 38.0}, state)
    assert not MCMC.valid_parameter_combination({'lEmin': 40.0}, state)
    assert not MCMC.valid_parameter_combination({'lEmax': 40.0}, state)


def test_double_broken_power_law_initial_walkers_obey_joint_prior():
    state = parameters.State()
    state.energy.luminosity_function = 5
    params = {
        'lEmin': {'min': 36.0, 'max': 40.0},
        'lEb': {'min': 37.0, 'max': 41.0},
        'lEb2': {'min': 39.0, 'max': 43.0},
        'lEmax': {'min': 41.0, 'max': 45.0},
        'gamma': {'min': -3.0, 'max': 0.0},
        'gamma2': {'min': -5.0, 'max': 0.0},
        'gamma3': {'min': -7.0, 'max': 0.0},
    }

    walkers = MCMC.get_initial_walkers(
        state, params, nwalkers=64, rng=np.random.default_rng(4321)
    )
    indices = {name: i for i, name in enumerate(params)}

    assert np.all(
        walkers[:, indices['lEmin']] < walkers[:, indices['lEb']]
    )
    assert np.all(
        walkers[:, indices['lEb']] < walkers[:, indices['lEb2']]
    )
    assert np.all(
        walkers[:, indices['lEb2']] < walkers[:, indices['lEmax']]
    )


def test_grid_selects_double_broken_power_law_functions():
    state = parameters.State()
    state.energy.luminosity_function = 5
    state.energy.lEb = 1.0
    state.energy.lEb2 = 2.0
    state.energy.gamma2 = -1.0
    state.energy.gamma3 = -2.0

    test_grid = grid.Grid.__new__(grid.Grid)
    test_grid.state = state
    test_grid.luminosity_function = 5
    test_grid.init_luminosity_functions()

    cumulative = test_grid.vector_cum_lf(
        np.array([1.0, 10.0, 100.0, 1000.0]),
        1.0,
        1000.0,
        -0.5,
    )
    differential = test_grid.vector_diff_lf(
        np.array([10.0, 100.0]),
        1.0,
        1000.0,
        -0.5,
    )

    assert cumulative[0] == 1.0
    assert cumulative[-1] == 0.0
    assert np.all(differential > 0)
