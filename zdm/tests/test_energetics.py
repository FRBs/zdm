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


def test_vector_cum_broken_power_law():
    """The broken power law follows the requested piecewise expression."""
    Emin, Emax = 1.0, 100.0
    gamma1, gamma2, Eb = -1.0, -2.0, 10.0
    Eth = np.array([0.5, Emin, 5.0, Eb, 50.0, Emax, 200.0])

    result = energetics.vector_cum_broken_power_law(
        Eth, Emin, Emax, gamma1, gamma2, Eb
    )

    denominator = (
        (1 - (Emin / Eb) ** gamma1) / gamma1
        + ((Emax / Eb) ** gamma2 - 1) / gamma2
    )
    expected_below_break = (
        (1 - (5.0 / Eb) ** gamma1) / gamma1
        + ((Emax / Eb) ** gamma2 - 1) / gamma2
    ) / denominator
    expected_at_break = (
        ((Emax / Eb) ** gamma2 - 1) / gamma2
    ) / denominator
    expected_above_break = (
        ((Emax / Eb) ** gamma2 - (50.0 / Eb) ** gamma2) / gamma2
    ) / denominator

    expected = np.array([
        1.0,
        1.0,
        expected_below_break,
        expected_at_break,
        expected_above_break,
        0.0,
        0.0,
    ])
    np.testing.assert_allclose(result, expected)
    assert np.all(np.diff(result) <= 0)


def test_vector_cum_broken_power_law_zero_indices():
    """Zero indices use the finite logarithmic limit of the expression."""
    result = energetics.vector_cum_broken_power_law(
        np.array([1.0, 10.0, 100.0]), 1.0, 100.0, 0.0, 0.0, 10.0
    )

    np.testing.assert_allclose(result, [1.0, 0.5, 0.0])


def test_diff_broken_power_law_matches_cumulative_derivative():
    """The differential function is minus the cumulative derivative."""
    params = (1.0, 100.0, -1.0, -2.0, 10.0)
    energies = np.array([2.0, 5.0, 20.0, 50.0])
    step = energies * 1e-5

    upper = energetics.vector_cum_broken_power_law(
        energies + step, *params
    )
    lower = energetics.vector_cum_broken_power_law(
        energies - step, *params
    )
    numerical_density = -(upper - lower) / (2 * step)
    density = energetics.vector_diff_broken_power_law(energies, *params)

    np.testing.assert_allclose(density, numerical_density, rtol=1e-8)


def test_broken_power_law_array_wrappers_preserve_shape():
    params = (1.0, 100.0, -1.0, -2.0, 10.0)
    energies = np.array([[1.0, 5.0], [20.0, 100.0]])

    cumulative = energetics.array_cum_broken_power_law(energies, *params)
    differential = energetics.array_diff_broken_power_law(energies, *params)

    assert cumulative.shape == energies.shape
    assert differential.shape == energies.shape


def test_broken_power_law_vector_functions_accept_scalar():
    params = (1.0, 100.0, -1.0, -2.0, 10.0)

    cumulative = energetics.vector_cum_broken_power_law(10.0, *params)
    differential = energetics.vector_diff_broken_power_law(10.0, *params)

    assert np.isscalar(cumulative)
    assert np.isscalar(differential)


def test_double_broken_power_law_boundaries_and_monotonicity():
    params = (1.0, 1000.0, -0.5, -1.0, -2.0, 10.0, 100.0)
    energies = np.array([0.5, 1.0, 5.0, 10.0, 50.0, 100.0, 500.0,
                         1000.0, 2000.0])

    cumulative = energetics.vector_cum_double_broken_power_law(
        energies, *params
    )

    assert cumulative[0] == 1.0
    assert cumulative[1] == 1.0
    assert cumulative[-2] == 0.0
    assert cumulative[-1] == 0.0
    assert np.all(np.diff(cumulative) <= 0)


def test_double_broken_power_law_is_continuous_at_breaks():
    params = (1.0, 1000.0, -0.5, -1.0, -2.0, 10.0, 100.0)
    for break_energy in (10.0, 100.0):
        epsilon = break_energy * 1e-9
        values = energetics.vector_diff_double_broken_power_law(
            np.array([break_energy - epsilon, break_energy + epsilon]),
            *params,
        )
        np.testing.assert_allclose(values[0], values[1], rtol=1e-7)


def test_diff_double_broken_power_law_matches_cumulative_derivative():
    params = (1.0, 1000.0, -0.5, -1.0, -2.0, 10.0, 100.0)
    energies = np.array([2.0, 5.0, 20.0, 50.0, 200.0, 500.0])
    step = energies * 1e-5

    upper = energetics.vector_cum_double_broken_power_law(
        energies + step, *params
    )
    lower = energetics.vector_cum_double_broken_power_law(
        energies - step, *params
    )
    numerical_density = -(upper - lower) / (2 * step)
    density = energetics.vector_diff_double_broken_power_law(
        energies, *params
    )

    np.testing.assert_allclose(density, numerical_density, rtol=1e-8)


def test_double_broken_power_law_zero_indices():
    params = (1.0, 1000.0, 0.0, 0.0, 0.0, 10.0, 100.0)
    cumulative = energetics.vector_cum_double_broken_power_law(
        np.array([1.0, 10.0, 100.0, 1000.0]), *params
    )
    np.testing.assert_allclose(cumulative, [1.0, 2 / 3, 1 / 3, 0.0])


def test_double_broken_power_law_wrappers_and_scalar():
    params = (1.0, 1000.0, -0.5, -1.0, -2.0, 10.0, 100.0)
    energies = np.array([[1.0, 10.0], [100.0, 1000.0]])

    cumulative = energetics.array_cum_double_broken_power_law(
        energies, *params
    )
    differential = energetics.array_diff_double_broken_power_law(
        energies, *params
    )
    assert cumulative.shape == energies.shape
    assert differential.shape == energies.shape
    assert np.isscalar(
        energetics.vector_cum_double_broken_power_law(10.0, *params)
    )
    assert np.isscalar(
        energetics.vector_diff_double_broken_power_law(10.0, *params)
    )


def test_double_broken_power_law_rejects_invalid_energy_order():
    params = (1.0, 1000.0, -0.5, -1.0, -2.0, 100.0, 10.0)
    with pytest.raises(ValueError, match="Emin < Eb1 < Eb2 < Emax"):
        energetics.vector_cum_double_broken_power_law(10.0, *params)
