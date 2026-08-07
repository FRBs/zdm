"""
FRB luminosity/energy function implementations.

This module provides functions for computing FRB luminosity (energy) distributions,
including both cumulative and differential forms. The main luminosity functions
implemented are:

1. **Power Law**: Simple power-law distribution dN/dE ~ E^gamma between Emin and Emax
2. **Gamma Function**: Upper incomplete gamma function distribution with exponential cutoff

The gamma function implementation uses spline interpolation for efficiency, as
direct evaluation of the incomplete gamma function is computationally expensive
when called many times during grid calculations.

Key Functions
-------------
- `vector_cum_power_law`: Cumulative power-law luminosity function
- `vector_cum_broken_power_law`: Cumulative broken power-law luminosity function
- `vector_cum_double_broken_power_law`: Cumulative two-break power-law luminosity function
- `vector_cum_gamma_spline`: Cumulative gamma function with spline interpolation
- `array_cum_gamma_spline`: N-dimensional array wrapper for gamma function
- `init_igamma_splines`: Initialize spline lookup tables for gamma functions

Module Variables
----------------
igamma_splines : dict
    Cache of spline interpolators keyed by gamma value.
SplineMin, SplineMax : float
    Log10 range for spline interpolation.
SplineLog : bool
    If True, perform interpolation in log-log space (recommended).

Example
-------
>>> from zdm import energetics
>>> Eth = np.logspace(38, 42, 100)  # Energy thresholds in erg
>>> params = (1e38, 1e42, -1.5)  # Emin, Emax, gamma
>>> fraction = energetics.vector_cum_power_law(Eth, *params)
"""

import numpy as np
from scipy import interpolate
import mpmath

from IPython import embed

# Global cache for spline interpolators
igamma_splines = {}
igamma_linear = {}
igamma_linear_log10 = {}

# Spline interpolation settings
SplineMin = -6   # Log10 of minimum argument for incomplete gamma
SplineMax = 6    # Log10 of maximum argument
NSpline = 1000   # Number of spline points
SplineLog = True # Use log-space interpolation (more accurate)

def init_igamma_splines(gammas, reinit=False, k=3):
    """Initialize spline interpolators for the upper incomplete gamma function.

    Pre-computes spline representations of the upper incomplete gamma function
    Gamma(gamma, x) for fast evaluation during grid calculations. Splines are
    cached globally and reused across calls.

    Parameters
    ----------
    gammas : list of float
        Values of gamma (the shape parameter) for which to create splines.
    reinit : bool, optional
        If True, reinitialize splines even if they already exist. Default False.
    k : int, optional
        Degree of spline interpolation. Default is 3 (cubic). Valid range: 1-5.
        Note: k=2 and k=4 not recommended due to numerical issues.

    Notes
    -----
    If module variable `SplineLog` is True (default), interpolation is performed
    in log-log space, which provides better accuracy over the wide dynamic range
    of the incomplete gamma function.
    """
    global SplineMin,SplineMax,NSpline,SplineLog
    for gamma in gammas:
        if gamma not in igamma_splines.keys() or reinit:
            # print(f"Initializing igamma_spline for gamma={gamma}")
            lavals = np.linspace(SplineMin, SplineMax, NSpline)
            avals = 10**lavals
            numer = np.array([float(mpmath.gammainc(
                gamma, a=iEE)) for iEE in avals])
            if SplineLog:
                # check for literal zeros, set them to small values
                zero = np.where(numer == 0.)[0]
                ismall = zero[0]-1
                smallest = numer[ismall]
                numer[zero] = smallest
                lnumer = np.log10(numer)
                igamma_splines[gamma] = interpolate.splrep(lavals, lnumer,k=k)
            else:
                igamma_splines[gamma] = interpolate.splrep(avals, numer,k=k)


def init_igamma_linear(gammas: list, reinit: bool = False,
                       log: bool = False):
    """Initialize linear interpolators for the upper incomplete gamma function.

    Alternative to spline interpolation using scipy's interp1d.

    Parameters
    ----------
    gammas : list of float
        Values of gamma for which to create interpolators.
    reinit : bool, optional
        If True, reinitialize even if interpolator exists. Default False.
    log : bool, optional
        If True, perform interpolation in log10 space. Default False.
    """

    for gamma in gammas:
        if (log and (gamma not in igamma_linear_log10.keys())) \
            or reinit or \
            (not log and (gamma not in igamma_linear.keys())):

            print(f"Initializing igamma_linear for gamma={gamma} with log10")

            # values
            avals = 10**np.linspace(-8, 6., 1000)

            numer = np.array([float(mpmath.gammainc(
                gamma, a=iEE)) for iEE in avals])

            # convert avals to log10 space (init x values)
            if log:
                log_avals = np.log10(avals)
                igamma_linear_log10[gamma] = interpolate.interp1d(log_avals, numer)
            else:
                igamma_linear[gamma] = interpolate.interp1d(avals, numer)

def template_array_cumulative_luminosity_function(Eth,*params):
    """
    Template for a cumulative luminosity function
    Returns fraction of cumulative distribution above Eth
    Luminosity function is defined by *params
    Eth is a multidimensional numpy array
    Always just wraps the vector version
    """
    dims=Eth.shape
    Eth=Eth.flatten()
    result=template_vector_cumulative_luminosity_function(Eth,*params)
    result=result.reshape(dims)
    return result

def template_vector_cumulative_luminosity_function(Eth,*params):
    """
    Template for a cumulative luminosity function
    Returns fraction of cumulative distribution above Eth
    Luminosity function is defined by *params
    Eth is a 1D numpy array
    This example uses a cumulative power law
    """
    #result=f(params)
    #return result
    return None

########### simple power law functions #############

def array_diff_power_law(Eth,*params):
    """ Calculates the differential fraction of bursts for a power law
    at a given Eth, where Eth is an N-dimensional array
    """
    dims=Eth.shape
    Eth=Eth.flatten()
    #if gamma >= 0: #handles crazy dodgy cases. Or just return 0?
    #	result=np.zeros([Eth.size])
    #	result[np.where(Eth < Emax)]=1.
    #	result=result.reshape(dims)
    #	Eth=Eth.reshape(dims)
    #	return result

    result=vector_diff_power_law(Eth,*params)
    result=result.reshape(dims)
    return result


def vector_diff_power_law(Eth,*params):
    Emin=params[0]
    Emax=params[1]
    gamma=params[2]

    result=-(gamma*Eth**(gamma-1)) / (Emin**gamma-Emax**gamma )

    low=np.where(Eth < Emin)[0]
    if len(low) > 0:
        result[low]=0.
    high=np.where(Eth > Emax)[0]
    if len(high) > 0:
        result[high]=0.

    return result


def array_cum_power_law(Eth,*params):
    """ Calculates the fraction of bursts above a certain power law
    for a given Eth, where Eth is an N-dimensional array
    """
    dims=Eth.shape
    Eth=Eth.flatten()
    #if gamma >= 0: #handles crazy dodgy cases. Or just return 0?
    #	result=np.zeros([Eth.size])
    #	result[np.where(Eth < Emax)]=1.
    #	result=result.reshape(dims)
    #	Eth=Eth.reshape(dims)
    #	return result
    result=vector_cum_power_law(Eth,*params)
    result=result.reshape(dims)
    return result

def vector_cum_power_law(Eth, *params):
    """Cumulative power-law luminosity function.

    Computes the fraction of bursts with energy above threshold Eth for
    a power-law distribution dN/dE ~ E^gamma between Emin and Emax.

    Parameters
    ----------
    Eth : ndarray
        Energy threshold values in erg.
    *params : tuple
        (Emin, Emax, gamma) - minimum energy, maximum energy, power-law index.
        Gamma is typically negative (e.g., -1.5).

    Returns
    -------
    ndarray
        Fraction of bursts with E > Eth. Returns 1 for Eth < Emin, 0 for Eth > Emax.
    """
    params=np.array(params)
    Emin=params[0]
    Emax=params[1]
    gamma=params[2]
    result=(Eth**gamma-Emax**gamma ) / (Emin**gamma-Emax**gamma )
    low=np.where(Eth < Emin)[0]
    if len(low) > 0:
        result[low]=1.
    high=np.where(Eth > Emax)[0]
    if len(high)>0:
        result[high]=0.
    return result


########### simple broken power law functions #############

def vector_cum_broken_power_law(Eth, *params):
    """Cumulative broken power-law luminosity function.

    Computes the fraction of bursts with energy above ``Eth`` for a broken
    power law whose indices below and above the break energy are ``gamma1``
    and ``gamma2``, respectively.  The two branches have the same
    normalization at the break.

    Parameters
    ----------
    Eth : ndarray
        One-dimensional array of energy threshold values.
    *params : tuple
        ``(Emin, Emax, gamma1, gamma2, Eb)``: minimum energy, maximum
        energy, lower power-law index, upper power-law index, and break
        energy.

    Returns
    -------
    ndarray
        Fraction of bursts with ``E > Eth``.  The result is 1 below
        ``Emin`` and 0 above ``Emax``.

    Notes
    -----
    Zero-valued indices are evaluated using their logarithmic limits.
    """
    Emin, Emax, gamma1, gamma2, Eb = params

    if not (0 < Emin < Eb < Emax):
        raise ValueError("energies must satisfy 0 < Emin < Eb < Emax")

    Eth = np.asarray(Eth, dtype=float)
    scalar_input = Eth.ndim == 0
    if Eth.ndim > 1:
        raise ValueError("Eth must be a one-dimensional array")
    Eth = np.atleast_1d(Eth)

    def lower_integral(ratio, gamma):
        """Return (1 - ratio**gamma) / gamma stably."""
        if gamma == 0:
            return -np.log(ratio)
        return -np.expm1(gamma * np.log(ratio)) / gamma

    def upper_integral(ratio, gamma):
        """Return (ratio**gamma - 1) / gamma stably."""
        if gamma == 0:
            return np.log(ratio)
        return np.expm1(gamma * np.log(ratio)) / gamma

    upper_at_break = upper_integral(Emax / Eb, gamma2)
    normalization = (lower_integral(Emin / Eb, gamma1) + upper_at_break)

    result = np.empty_like(Eth)
    below_min = Eth < Emin
    above_max = Eth > Emax
    below_break = (Eth >= Emin) & (Eth < Eb)
    above_break = (Eth >= Eb) & (Eth <= Emax)

    result[below_min] = 1.0
    result[above_max] = 0.0
    result[below_break] = (lower_integral(Eth[below_break] / Eb, gamma1) + upper_at_break) / normalization
    result[above_break] = upper_integral(Emax / Eb, gamma2) - upper_integral(Eth[above_break] / Eb, gamma2)
    result[above_break] /= normalization

    if scalar_input:
        return result[0]
    return result


def array_cum_broken_power_law(Eth, *params):
    """N-dimensional wrapper for :func:`vector_cum_broken_power_law`."""
    Eth = np.asarray(Eth)
    dims = Eth.shape
    result = vector_cum_broken_power_law(Eth.flatten(), *params)
    return result.reshape(dims)


def vector_diff_broken_power_law(Eth, *params):
    """Differential broken power-law luminosity function.

    This is the normalized probability density corresponding to
    :func:`vector_cum_broken_power_law`.

    Parameters
    ----------
    Eth : ndarray
        One-dimensional array of energies.
    *params : tuple
        ``(Emin, Emax, gamma1, gamma2, Eb)``.

    Returns
    -------
    ndarray
        Probability density at each energy, with zero density outside
        ``[Emin, Emax]``.
    """
    Emin, Emax, gamma1, gamma2, Eb = params

    if not (0 < Emin < Eb < Emax):
        raise ValueError("energies must satisfy 0 < Emin < Eb < Emax")

    Eth = np.asarray(Eth, dtype=float)
    scalar_input = Eth.ndim == 0
    if Eth.ndim > 1:
        raise ValueError("Eth must be a one-dimensional array")
    Eth = np.atleast_1d(Eth)

    def integral(ratio, gamma):
        if gamma == 0:
            return np.log(ratio)
        return np.expm1(gamma * np.log(ratio)) / gamma

    normalization = (-integral(Emin / Eb, gamma1) + integral(Emax / Eb, gamma2))

    result = np.zeros_like(Eth)
    lower = (Eth >= Emin) & (Eth < Eb)
    upper = (Eth >= Eb) & (Eth <= Emax)
    result[lower] = ((Eth[lower] / Eb) ** (gamma1 - 1) / (Eb * normalization))
    result[upper] = ((Eth[upper] / Eb) ** (gamma2 - 1) / (Eb * normalization))
    if scalar_input:
        return result[0]
    return result


def array_diff_broken_power_law(Eth, *params):
    """N-dimensional wrapper for :func:`vector_diff_broken_power_law`."""
    Eth = np.asarray(Eth)
    dims = Eth.shape
    result = vector_diff_broken_power_law(Eth.flatten(), *params)
    return result.reshape(dims)



########### Double broken power law functions #############


def vector_cum_double_broken_power_law(Eth, *params):
    """Cumulative doubly-broken power-law luminosity function.

    The three differential branches are continuous at both break energies
    and follow the zDM convention ``dP/dE ~ E**(gamma_i - 1)``.

    Parameters
    ----------
    Eth : float or ndarray
        Energy threshold values.
    *params : tuple
        ``(Emin, Emax, gamma1, gamma2, gamma3, Eb1, Eb2)``.

    Returns
    -------
    float or ndarray
        Fraction of bursts with energy greater than ``Eth``.
    """
    Emin, Emax, gamma1, gamma2, gamma3, Eb1, Eb2 = params

    if not (0 < Emin < Eb1 < Eb2 < Emax):
        raise ValueError("energies must satisfy 0 < Emin < Eb1 < Eb2 < Emax")

    Eth = np.asarray(Eth, dtype=float)
    scalar_input = Eth.ndim == 0
    if Eth.ndim > 1:
        raise ValueError("Eth must be a one-dimensional array")
    Eth = np.atleast_1d(Eth)

    def lower_integral(ratio, gamma):
        """Return (1 - ratio**gamma) / gamma stably."""
        if gamma == 0:
            return -np.log(ratio)
        return -np.expm1(gamma * np.log(ratio)) / gamma

    def upper_integral(ratio, gamma):
        """Return (ratio**gamma - 1) / gamma stably."""
        if gamma == 0:
            return np.log(ratio)
        return np.expm1(gamma * np.log(ratio)) / gamma

    break_ratio = Eb2 / Eb1
    middle_total = upper_integral(break_ratio, gamma2)
    # The extra factor follows from continuity of dP/dE at Eb2.
    high_scale = break_ratio ** gamma2
    high_total = high_scale * upper_integral(Emax / Eb2, gamma3)
    normalization = (lower_integral(Emin / Eb1, gamma1) + middle_total + high_total)

    result = np.empty_like(Eth)
    below_min = Eth < Emin
    above_max = Eth > Emax
    first = (Eth >= Emin) & (Eth < Eb1)
    second = (Eth >= Eb1) & (Eth < Eb2)
    third = (Eth >= Eb2) & (Eth <= Emax)

    result[below_min] = 1.0
    result[above_max] = 0.0
    result[first] = (lower_integral(Eth[first] / Eb1, gamma1) + middle_total + high_total) / normalization
    result[second] = (middle_total - upper_integral(Eth[second] / Eb1, gamma2) + high_total) / normalization
    result[third] = high_scale * (upper_integral(Emax / Eb2, gamma3) - upper_integral(Eth[third] / Eb2, gamma3)) / normalization

    if scalar_input:
        return result[0]
    return result


def array_cum_double_broken_power_law(Eth, *params):
    """N-dimensional wrapper for the cumulative doubly-broken power law."""
    Eth = np.asarray(Eth)
    dims = Eth.shape
    result = vector_cum_double_broken_power_law(Eth.flatten(), *params)
    return result.reshape(dims)


def vector_diff_double_broken_power_law(Eth, *params):
    """Differential probability density for the doubly-broken power law."""
    Emin, Emax, gamma1, gamma2, gamma3, Eb1, Eb2 = params

    if not (0 < Emin < Eb1 < Eb2 < Emax):
        raise ValueError("energies must satisfy 0 < Emin < Eb1 < Eb2 < Emax")

    Eth = np.asarray(Eth, dtype=float)
    scalar_input = Eth.ndim == 0
    if Eth.ndim > 1:
        raise ValueError("Eth must be a one-dimensional array")
    Eth = np.atleast_1d(Eth)

    def integral(ratio, gamma):
        if gamma == 0:
            return np.log(ratio)
        return np.expm1(gamma * np.log(ratio)) / gamma

    break_ratio = Eb2 / Eb1
    high_scale = break_ratio ** gamma2
    normalization = (-integral(Emin / Eb1, gamma1)+ integral(break_ratio, gamma2)+ high_scale * integral(Emax / Eb2, gamma3))

    result = np.zeros_like(Eth)
    first = (Eth >= Emin) & (Eth < Eb1)
    second = (Eth >= Eb1) & (Eth < Eb2)
    third = (Eth >= Eb2) & (Eth <= Emax)

    result[first] = ((Eth[first] / Eb1) ** (gamma1 - 1) / (Eb1 * normalization))
    result[second] = ((Eth[second] / Eb1) ** (gamma2 - 1) / (Eb1 * normalization))
    result[third] = (break_ratio ** (gamma2 - 1) * (Eth[third] / Eb2) ** (gamma3 - 1) / (Eb1 * normalization))

    if scalar_input:
        return result[0]
    return result


def array_diff_double_broken_power_law(Eth, *params):
    """N-dimensional wrapper for the differential doubly-broken power law."""
    Eth = np.asarray(Eth)
    dims = Eth.shape
    result = vector_diff_double_broken_power_law(Eth.flatten(), *params)
    return result.reshape(dims)



########### gamma functions #############

def vector_cum_gamma(Eth, *params):
    """Cumulative gamma-function luminosity function (slow, exact version).

    Computes the fraction of bursts with energy above threshold Eth using
    an upper incomplete gamma function distribution. This version evaluates
    the gamma function directly using mpmath - accurate but slow.

    Parameters
    ----------
    Eth : ndarray
        Energy threshold values in erg.
    *params : tuple
        (Emin, Emax, gamma) - minimum energy, characteristic energy, shape parameter.

    Returns
    -------
    ndarray
        Fraction of bursts with E > Eth. Returns 1 for Eth < Emin.

    See Also
    --------
    vector_cum_gamma_spline : Fast spline-interpolated version (recommended).
    """
    params=np.array(params)
    Emin=params[0]
    Emax=params[1]
    gamma=params[2]

    # Calculate
    norm = float(mpmath.gammainc(gamma, a=Emin/Emax))
    Eth_Emax = Eth/Emax
    # If this is too slow, we can adopt scipy + recurrance
    numer = np.array([float(mpmath.gammainc(
        gamma, a=iEE)) for iEE in Eth_Emax])
    result=numer/norm

    # Low end
    low= Eth < Emin
    result[low]=1.
    return result

def vector_cum_gamma_spline(Eth: np.ndarray, *params):
    """Cumulative gamma-function luminosity function using spline interpolation.

    Fast version of `vector_cum_gamma` that uses pre-computed spline
    interpolators. This is the recommended function for grid calculations.

    Parameters
    ----------
    Eth : ndarray
        Energy threshold values in erg.
    *params : tuple
        (Emin, Emax, gamma) - minimum energy, characteristic energy, shape parameter.

    Returns
    -------
    ndarray
        Fraction of bursts with E > Eth. Returns 1 for Eth < Emin.

    Notes
    -----
    Automatically initializes splines for new gamma values if needed.
    """
    global SplineLog

    params=np.array(params)
    Emin=params[0]
    Emax=params[1]
    gamma=params[2]

    # Calculate
    norm = float(mpmath.gammainc(gamma, a=Emin/Emax))
    Eth_Emax = Eth/Emax
    if gamma not in igamma_splines.keys():
        init_igamma_splines([gamma])
    if SplineLog:
        numer = 10**interpolate.splev(np.log10(Eth_Emax), igamma_splines[gamma])
    else:
        numer = interpolate.splev(Eth_Emax, igamma_splines[gamma])
    result=numer/norm

    # Low end
    low = Eth < Emin

    if np.isscalar(result):
        if low:
            result = 1.
    else:
        result[low]=1.
    return result

def vector_cum_gamma_linear(Eth:np.ndarray, *params):
    """ Calculate cumulative Gamma function using a linear interp1d

    Args:
        Eth (np.ndarray): Energy threshold in ergs

    Returns:
        np.ndarray: cumulative probability above Eth
    """
    params=np.array(params)
    Emin=params[0]
    Emax=params[1]
    gamma=params[2]
    log = params[3]

    # Calculate
    norm = float(mpmath.gammainc(gamma, a=Emin/Emax))

    # Branch either with log10 space or without
    if log:
        Eth_Emax = Eth - np.log10(Emax)
        if gamma not in igamma_linear_log10.keys():
            init_igamma_linear([gamma], log=log)
        try:
            numer = igamma_linear_log10[gamma](Eth_Emax)
        except:
            print(Eth_Emax)
            embed(header='248 of energetics.py')
        Emin_temp = np.log10(float(Emin))

    else:
        Eth_Emax = Eth/Emax
        if gamma not in igamma_linear.keys():
            init_igamma_linear([gamma], log=log)

        numer = igamma_linear[gamma](Eth_Emax)
        Emin_temp = Emin

    result=numer/norm

    # Low end
    low= Eth < Emin_temp
    result[low]=1.
    return result

def array_diff_gamma(Eth,*params):
    """ Calculates the differential fraction of bursts for a gamma function
    at a given Eth, where Eth is an N-dimensional array
    """
    dims=Eth.shape
    result=vector_diff_gamma(Eth.flatten(),*params)
    result=result.reshape(dims)
    return result

def array_cum_gamma(Eth,*params):
    """ Calculates the fraction of bursts above a certain gamma function
    for a given Eth, where Eth is an N-dimensional array
    """
    dims=Eth.shape
    result=vector_cum_gamma(Eth.flatten(),*params)
    result=result.reshape(dims)
    return result

def array_cum_gamma_spline(Eth,*params):
    """ Calculates the fraction of bursts above a certain gamma function
    for a given Eth, where Eth is an N-dimensional array
    """
    dims=Eth.shape
    result=vector_cum_gamma_spline(Eth.flatten(),*params)
    result=result.reshape(dims)
    return result

def array_cum_gamma_linear(Eth,*params):
    """ Calculates the fraction of bursts above a certain gamma function
    for a given Eth, where Eth is an N-dimensional array
    """
    dims=Eth.shape
    result=vector_cum_gamma_linear(Eth.flatten(),*params)
    result=result.reshape(dims)
    return result

def vector_diff_gamma(Eth, *params):
    """Differential gamma-function luminosity function.

    Computes dN/dE normalized such that the integral from Emin to infinity
    equals 1. This is the probability density of burst energies.

    Parameters
    ----------
    Eth : ndarray
        Energy values in erg at which to evaluate.
    *params : tuple
        (Emin, Emax, gamma) - minimum energy, characteristic energy, shape parameter.

    Returns
    -------
    ndarray
        Probability density dN/dE at each energy. Returns 0 for E < Emin.
    """
    Emin=params[0]
    Emax=params[1]
    gamma=params[2]

    norm = Emax*float(mpmath.gammainc(gamma, a=Emin/Emax))
    result= (Eth/Emax)**(gamma-1) * np.exp(-Eth/Emax) / norm

    low= Eth < Emin
    result[low]=0.

    return result
