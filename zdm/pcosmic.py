"""
Probability distribution of cosmic dispersion measure given redshift, p(DM|z).

This module implements the Macquart relation - the probability distribution
of cosmic (extragalactic) dispersion measure as a function of redshift. It is
based on the formalism described in Macquart et al. (2020, Nature) and
implements Equation 4 from that work.

The cosmic DM arises from free electrons in the intergalactic medium (IGM).
The distribution is characterized by a feedback parameter F that controls
the variance, and normalization constant C0 that ensures <DM> = DM_cosmic.

Key Features
------------
- p(DM|z) calculation using the modified lognormal distribution
- Mean cosmic DM from the Macquart relation using astropy/frb cosmology
- Lognormal host galaxy DM contribution convolution
- Interpolation utilities for fast grid-based calculations

Main Functions
--------------
- `pcosmic`: Core probability distribution function p(delta|z,F)
- `get_mean_DM`: Compute mean cosmic DM at given redshifts
- `get_pDM_grid`: Generate 2D grid of p(DM|z) probabilities
- `get_dm_mask`: Generate host galaxy DM convolution kernel

References
----------
Macquart et al. (2020), Nature, 581, 391 - "A census of baryons in the
Universe from localized fast radio bursts"

Author: C.W. James
"""
# from fcntl import F_ADD_SEALS
import sys

# sys.path.insert(1, '/Users/cjames/CRAFT/FRB_library/ne2001-master/src/ne2001')

import matplotlib.pyplot as plt
import numpy as np

from astropy.cosmology import FlatLambdaCDM

# from frb import dlas
from frb.dm import igm
from zdm import cosmology as cos
from zdm import parameters

# from zdm import c_code

import scipy as sp

# import astropy.units as u

from IPython import embed

# these are fitted values, which are shown to best-match data
alpha = 3
beta = 3
# F=0.32 ##### feedback best fit, but FIT IT!!! #####


def p(DM, z, F):
    mean = z * 1000  # NOT true, just testing!
    delta = DM / mean
    p = pcosmic(delta, z, F)


def pcosmic(delta, z, logF, C0):
    """Probability density of fractional cosmic DM, p(delta|z).

    Implements Equation 4 from Macquart et al. (2020) for the probability
    distribution of cosmic DM fluctuations. The distribution is parameterized
    such that delta = DM_cosmic / <DM_cosmic> follows a modified log-normal
    with redshift-dependent width.

    Parameters
    ----------
    delta : float or array_like
        Fractional cosmic DM, defined as DM_cosmic / <DM_cosmic>.
        Must be positive.
    z : float
        Redshift. Used to compute the width parameter sigma = F * z^(-0.5).
    logF : float
        Log10 of the feedback/fluctuation parameter F. Controls the width
        of the distribution. Typical values are logF ~ -0.5 to 0.
    C0 : float
        Normalization constant that ensures <delta> = 1. Should be computed
        via `iterate_C0()` for each (z, F) pair.

    Returns
    -------
    float or ndarray
        Probability density p(delta|z). Not normalized; integrate over delta
        for normalization.

    Notes
    -----
    Uses module-level constants alpha=3, beta=3 as shape parameters.
    The standard deviation scales as F * z^(-0.5), reflecting the central
    limit theorem averaging of IGM fluctuations along the line of sight.
    """

    ### logF compensation
    F = 10 ** (logF)
    ###

    global beta, alpha
    sigma = F * z ** -0.5
    return delta ** -beta * np.exp(
        -((delta ** -alpha - C0) ** 2.0) / (2.0 * alpha ** 2 * sigma ** 2)
    )


def p_delta_DM(z, F, C0, deltas=None, dmin=1e-3, dmax=10, ndelta=10000):
    """Calculate the probability distribution of fractional DM.

    Wrapper around `pcosmic` that generates a grid of delta values
    and computes the probability at each point.

    Parameters
    ----------
    z : float
        Redshift.
    F : float
        Log10 of the feedback parameter.
    C0 : float
        Normalization constant.
    deltas : array_like, optional
        Pre-defined delta values. If None, generates a linear grid.
    dmin : float, optional
        Minimum delta value if generating grid. Default is 1e-3.
    dmax : float, optional
        Maximum delta value if generating grid. Default is 10.
    ndelta : int, optional
        Number of delta points if generating grid. Default is 10000.

    Returns
    -------
    deltas : ndarray
        Array of fractional DM values.
    pdeltas : ndarray
        Probability density at each delta value.
    """
    if not deltas:
        deltas = np.linspace(dmin, dmax, ndelta)
    pdeltas = pcosmic(deltas, z, F, C0)
    return deltas, pdeltas


def iterate_C0(z, F, C0=1, Niter=10):
    """Iteratively solve for the normalization constant C0.

    The constant C0 ensures that the mean of the p(delta) distribution
    equals unity, i.e., <delta> = 1. This is required for the distribution
    to correctly represent DM_cosmic / <DM_cosmic>.

    Parameters
    ----------
    z : float
        Redshift.
    F : float
        Log10 of the feedback parameter.
    C0 : float, optional
        Initial guess for C0. Default is 1.
    Niter : int, optional
        Number of iterations. Default is 10.

    Returns
    -------
    float
        Converged value of C0.
    """
    dmin = 1e-3
    dmax = 10
    ndelta = 10000
    # these represent central bin values of delta = DM/<DM>
    deltas = np.linspace(dmin, dmax, ndelta)
    bin_w = deltas[1] - deltas[0]
    for i in np.arange(Niter):
        # pcosmic is a probability density
        # hence, we should calculate this at bin centres
        pdeltas = pcosmic(deltas, z, F, C0)
        norm = bin_w * np.sum(pdeltas)
        mean = bin_w * np.sum(pdeltas * deltas) / norm
        C0 += mean - 1.0

    return C0


def make_C0_grid(zeds, F):
    """Pre-compute C0 normalization constants for a grid of redshifts.

    Parameters
    ----------
    zeds : ndarray
        Array of redshift values.
    F : float
        Log10 of the feedback parameter.

    Returns
    -------
    ndarray
        Array of C0 values, one per redshift.
    """
    C0s = np.zeros([zeds.size])
    for i, z in enumerate(zeds):
        C0s[i] = iterate_C0(z, F)
    return C0s


def get_mean_DM(zeds: np.ndarray, state: parameters.State, Plot=False):
    """Compute the mean cosmic DM at given redshifts (Macquart relation).

    Calculates <DM_cosmic>(z) using the IGM electron density model from
    the frb package. Assumes a FlatLambdaCDM cosmology with parameters
    from the state object.

    Parameters
    ----------
    zeds : ndarray
        Redshifts at which to compute mean DM. Must be linearly spaced
        and represent bin centers (e.g., 0.5*dz, 1.5*dz, 2.5*dz, ...).
    state : parameters.State
        State object containing cosmological parameters (H0, Omega_b, Omega_m).
    Plot : bool, optional
        If True, generate diagnostic plots of DM vs z. Default is False.

    Returns
    -------
    ndarray
        Mean cosmic DM values in pc/cm^3 at each redshift.

    Notes
    -----
    Uses `frb.dm.igm.average_DM` for the underlying calculation.
    The redshifts must be evenly spaced for correct bin assignment.
    """
    # Generate the cosmology
    cosmo = FlatLambdaCDM(
        H0=state.cosmo.H0, Ob0=state.cosmo.Omega_b, Om0=state.cosmo.Omega_m
    )
    #
    dz = zeds[1]-zeds[0]
    zmax = zeds[-1] + dz/2. # top of uppermost zbin
    nz = zeds.size
    
    # this routine offsets zeval by 1 unit. That is, DM[i]
    # is the mean cosmic DM at zeval[i+1]
    # we want this for every 0.5, 1.5 etc
    # hence, we evaluate at 2*nz+1,
    tempDMbar, zeval = igm.average_DM(zmax, cosmo=cosmo, cumul=True, neval=2*nz + 1)
    
    # we now exract the DMbar that we actually want!
    # the zeroeth DMbar corresponds to zeval[1] which
    # since we calculate too many, is zeds[1]
    DMbar = tempDMbar[:-1:2]
    
    # performs a test to check if igm.average_DM has been fixed yet or not
    if np.abs(DMbar[0]/DMbar[1] - 1./3.) > 1e-2:
        print("DMbar is not scaling as expected! Central bins ",
                zeds[0]," and ",zeds[1]," have respective DM of ",
                DMbar[0]," and ",DMbar[1]," . Expected the second ",
                "value to be ",DMbar[0]*3.," . Perhaps ",
                igm.average_DM," has been fixed?",DMbar[0]/DMbar[1] - 1./3.)
        exit()
    
    if Plot:
        plt.figure()
        plt.xlabel('z')
        plt.ylabel('DM')
        dz = zeval[1]-zeval[0]
        plt.plot(zeds,DMbar,marker='+',label="wanted")
        plt.plot(zeval+dz,tempDMbar,linestyle=":",marker='x',label="eval")
        plt.xlim(0,0.1)
        plt.ylim(0,100)
        plt.legend()
        plt.tight_layout()
        plt.show()
        
        plt.xlim(4.9,5.)
        plt.ylim(4900,5500)
        plt.tight_layout()
        plt.show()
        
        plt.close()
    
    
    # Remove this check - now replaced as above
    # assert np.allclose(zeds, zeval[1:])
    
    # now returns the actual values, since
    # we have modified DMbar to exclude the
    # zero value already
    return DMbar.value


def get_log_mean_DM(zeds: np.ndarray, state: parameters.State):
    """Compute mean cosmic DM for arbitrarily-spaced redshifts.

    Similar to `get_mean_DM` but does not require linearly-spaced redshifts.
    Slower because it evaluates each redshift independently.

    Parameters
    ----------
    zeds : ndarray
        Redshifts at which to compute mean DM. Can have any spacing.
    state : parameters.State
        State object containing cosmological parameters.

    Returns
    -------
    ndarray
        Mean cosmic DM values in pc/cm^3 at each redshift.
    """
    # Generate the cosmology
    cosmo = FlatLambdaCDM(
        H0=state.cosmo.H0, Ob0=state.cosmo.Omega_b, Om0=state.cosmo.Omega_m
    )
    #
    nz = zeds.size
    dms = np.zeros([nz])
    for i, z in enumerate(zeds):
        # neval probably should be a function of max z
        DMbar, zeval = igm.average_DM(z, cosmo=cosmo, cumul=True, neval=nz + 1)  #
        dms[i] = DMbar[-1].value
        # wrong dimension
    return dms


def get_C0(z, F, zgrid, Fgrid, C0grid):
    """Interpolate C0 from a pre-computed grid.

    Performs bilinear interpolation on a pre-computed grid of C0 values
    to obtain C0 for arbitrary (z, F) pairs.

    Parameters
    ----------
    z : float
        Redshift at which to interpolate.
    F : float
        Log10 of feedback parameter.
    zgrid : ndarray
        Redshift grid points used to build C0grid.
    Fgrid : ndarray
        Log10(F) grid points used to build C0grid.
    C0grid : ndarray
        2D array of pre-computed C0 values, shape (len(zgrid), len(Fgrid)).

    Returns
    -------
    float
        Interpolated C0 value.
    """

    F = 10 ** F
    Fgrid = 10 ** Fgrid

    if z < zgrid[0] or z > zgrid[-1]:
        print("Z value out of range")
        exit()
    if F < Fgrid[0] or F > Fgrid[-1]:
        print("F value out of range")
        exit()

    iz2 = np.where(zgrid > z)[0]  # gets first element greater than
    iz1 = iz2 - 1
    kz1 = (zgrid[iz2] - z) / (zgrid[iz2] - zgrid[iz1])
    kz2 = 1.0 - kz1

    iF2 = np.where(Fgrid > F)[0]  # gets first element greater than
    iF1 = iF2 - 1
    kF1 = (Fgrid[iF2] - F) / (Fgrid[iF2] - Fgrid[iF1])
    kF2 = 1.0 - kF1

    C0 = (
        kz1 * kF1 * C0grid[iz1, iF1]
        + kz2 * kF1 * C0grid[iz2, iF1]
        + kz1 * kF2 * C0grid[iz1, iF2]
        + kz2 * kF2 * C0grid[iz2, iF2]
    )
    return C0


def get_pDM(z, F, DMgrid, zgrid, Fgrid, C0grid, zlog=False):
    """Compute p(DM|z) for a single redshift using pre-computed tables.

    Parameters
    ----------
    z : float
        Redshift.
    F : float
        Log10 of feedback parameter.
    DMgrid : ndarray
        DM values at which to evaluate the distribution.
    zgrid : ndarray
        Redshift grid for C0 interpolation.
    Fgrid : ndarray
        Feedback parameter grid for C0 interpolation.
    C0grid : ndarray
        Pre-computed C0 values.
    zlog : bool, optional
        If True, use log-spaced z grid. If False (default), use linear spacing.

    Returns
    -------
    ndarray
        Probability density p(DM|z) at each DM grid point.
    """
    C0 = get_C0(z, F, zgrid, Fgrid, C0grid)
    if zlog:
        DMbar = get_log_mean_DM(z)
    else:
        DMbar = get_mean_DM(z)
    deltas = DMgrid / DMbar  # in units of fractional DM
    pDM = pcosmic(deltas, z, F, C0)
    return pDM


def get_pDM_grid(
    state: parameters.State, DMgrid, zgrid, C0s, verbose=False, zlog=False
):
    """Generate a 2D grid of p(DM|z) probabilities.

    Computes the probability distribution of cosmic DM for each redshift
    in the grid. This is the main function for building z-DM grids.

    Parameters
    ----------
    state : parameters.State
        State object containing cosmological and IGM parameters.
    DMgrid : ndarray
        DM values (bin centers) for the output grid, in pc/cm^3.
    zgrid : ndarray
        Redshift values for the output grid.
    C0s : ndarray
        Pre-computed C0 normalization constants for each redshift.
    verbose : bool, optional
        If True, print diagnostic information. Default is False.
    zlog : bool, optional
        If True, zgrid is log-spaced. If False (default), linearly spaced.

    Returns
    -------
    ndarray
        2D array of shape (len(zgrid), len(DMgrid)) containing normalized
        p(DM|z) values. Each row sums to 1.
    """
    # added H0 dependency
    if zlog:
        DMbars = get_log_mean_DM(zgrid, state)
    else:
        DMbars = get_mean_DM(zgrid, state)

    pDMgrid = np.zeros([zgrid.size, DMgrid.size])
    if verbose:
        print("shapes and sizes are ", C0s.size, pDMgrid.shape, DMbars.shape)
    # iterates over zgrid to calculate p_delta_DM
    for i, z in enumerate(zgrid):
        deltas = DMgrid / DMbars[i]  # since pDM is defined such that the mean is 1

        pDMgrid[i, :] = pcosmic(deltas, z, state.IGM.logF, C0s[i])
        pDMgrid[i, :] /= np.sum(pDMgrid[i, :])  # normalisation
    return pDMgrid


#### Host galaxy DM contribution (lognormal distribution) ####

def linlognormal_dlin(DM, *args):
    """Lognormal probability density in linear DM space.

    Computes p(DM) where DM follows a lognormal distribution with
    parameters given in natural log space. Includes the 1/DM Jacobian
    for conversion from log to linear probability density.

    Parameters
    ----------
    DM : float or array_like
        Dispersion measure values (linear scale).
    *args : tuple
        (logmean, logsigma) - mean and standard deviation in natural log space.

    Returns
    -------
    float or ndarray
        Probability density p(DM) such that integral p(DM) dDM = 1.
    """
    logmean = args[0]
    logsigma = args[1]
    logDM = np.log(DM)
    norm = (2.0 * np.pi) ** -0.5 / DM / logsigma
    return norm * np.exp(-0.5 * ((logDM - logmean) / logsigma) ** 2)


def loglognormal_dlog(logDM, *args):
    """Gaussian probability density in log DM space.

    This is simply a Gaussian distribution where the variable happens to
    be log(DM). Does not include Jacobian for log-to-linear conversion.

    Parameters
    ----------
    logDM : float or array_like
        Natural log of dispersion measure.
    *args : tuple
        (logmean, logsigma, norm) - Gaussian parameters in log space.

    Returns
    -------
    float or ndarray
        Probability density p(log DM) such that integral p(log DM) d(log DM) = 1.
    """
    logmean = args[0]
    logsigma = args[1]
    norm = args[2]
    return norm * np.exp(-0.5 * ((logDM - logmean) / logsigma) ** 2)


def plot_mean(zvals, saveas, title="Mean DM"):

    mean = get_mean_DM(zvals)
    plt.figure()
    plt.xlabel("z")
    plt.ylabel("$\\overline{\\rm DM}$")
    plt.plot(zvals, mean, linewidth=2)
    plt.tight_layout()
    plt.title(title)
    plt.savefig(saveas)
    plt.show()
    plt.close()


def get_dm_mask(dmvals, params, zvals=None, plot=False):
    """Generate a convolution kernel for host galaxy DM contribution.

    Creates a probability distribution p(DM_host) following a lognormal
    distribution. This kernel can be convolved with p(DM_cosmic|z) to
    obtain the full p(DM_extragalactic|z) = p(DM_cosmic + DM_host|z).

    The host DM contribution is redshift-corrected: observed DM_host is
    reduced by (1+z) relative to the rest-frame value.

    Parameters
    ----------
    dmvals : ndarray
        DM grid values (bin centers) in pc/cm^3.
    params : array_like
        [log10_mean, log10_sigma] - lognormal parameters for host DM
        distribution in log10 space.
    zvals : ndarray, optional
        If provided, compute a redshift-dependent mask where host DM is
        scaled by 1/(1+z). If None, return a single z-independent mask.
    plot : bool, optional
        If True, generate diagnostic plots. Default is False.

    Returns
    -------
    ndarray
        If zvals is None: 1D array of shape (len(dmvals),) containing p(DM_host).
        If zvals is provided: 2D array of shape (len(zvals), len(dmvals)).
        Each row/vector is normalized to sum to 1.

    Notes
    -----
    The mask is designed for discrete convolution: p(DM_EG)[i] = sum_j p(DM_cosmic)[i-j] * mask[j]
    """

    if len(params) != 2:
        raise ValueError(
            "Incorrect number of DM parameters!",
            params,
            " (expected log10mean, log10sigma)",
        )
    
    # expect the params to be log10 of actual values for simplicity
    # this converts to natural log
    logmean = params[0] / 0.4342944619
    logsigma = params[1] / 0.4342944619

    ddm = dmvals[1] - dmvals[0]

    ##### first generates a mask from the lognormal distribution #####
    # in theory allows a mask up to length of the DN values, but will
    # get truncated.
    # The first value has half weight (0dDM to 0.5dDM) and represents
    # adding no new DM. The rest have width of dDM and represent
    # adding an integer number of dDM intervals.
    
    mask = np.zeros([dmvals.size])
    if zvals is not None:
        ndm = dmvals.size
        nz = zvals.size
        mask = np.zeros([nz, ndm])
        for j, z in enumerate(zvals):
            # with each redshift, we reduce the effects of a 'host' contribution by (1+z)
            # this means that we divide the value of logmean by 1/(1+z)
            # or equivalently, we multiply the ddm by this factor, since a
            # measurable increase of dDM means an intrinsic dDM*(1+z)
            # here we choose the latter, but it is the same
            mask[j, :] = integrate_pdm(ddm * (1.0 + z), ndm, logmean, logsigma)
            mask[j, :] /= np.sum(mask[j, :])  # the mask must integrate to unity
    else:
        # do this for the z=0 case
        # dmmin=0
        # dmmax=ddm*0.5
        # pdm,err=sp.integrate.quad(lognormal,dmmin,dmmax,args=(logmean,logsigma))
        # mask[0]=pdm
        # csum=pdm
        # imax=dmvals.size
        # for i in np.arange(1,dmvals.size):
        #    if csum > CSUMCUT:
        #        imax=i
        #        break
        #    dmmin=(i-0.5)*ddm
        #    dmmax=dmmin+ddm
        #    pdm,err=sp.integrate.quad(lognormal,dmmin,dmmax,args=(logmean,logsigma))
        #    csum += pdm
        #    mask[i]=pdm
        mask = integrate_pdm(ddm, dmvals.size, logmean, logsigma)
        mask /= np.sum(mask)  # ensures correct normalisation
        # mask=mask[0:imax]

    if plot and (not (zvals is not None)):
        plt.figure()
        plt.xlabel("${\\rm DM}_{\\rm X}$")
        plt.ylabel("$p({\\rm DM}_{\\rm X})$")
        label = (
            "$e^\\sigma="
            + str(np.exp(logsigma))[0:4]
            + "$, $e^\\mu="
            + str(np.exp(logmean))[0:4]
            + "$"
        )
        plt.plot(dmvals[0:imax], mask, linewidth=2, label=label)
        plt.tight_layout()
        plt.savefig("Plots/p_DM_X.pdf")
        plt.close()
    return mask


def integrate_pdm(ddm, ndm, logmean, logsigma, quick=True, plot=False):
    """Discretize a lognormal DM distribution onto histogram bins.

    Converts a continuous lognormal probability distribution into discrete
    bin probabilities for use in convolution operations. Two methods are
    available: fast (evaluate at bin centers) and slow (integrate each bin).

    Parameters
    ----------
    ddm : float
        Bin spacing in pc/cm^3.
    ndm : int
        Number of DM bins. Bins span [0, ndm*ddm] with first bin [0, 0.5*ddm].
    logmean : float
        Mean of the lognormal distribution in natural log space.
    logsigma : float
        Standard deviation in natural log space.
    quick : bool, optional
        If True (default), use fast method evaluating at bin centers.
        If False, integrate over each bin (slower but more accurate).
    plot : bool, optional
        If True, generate comparison plot of quick vs slow methods and exit.
        Default is False.

    Returns
    -------
    ndarray
        Array of shape (ndm,) containing probability mass in each bin.
        Not normalized; caller should normalize if needed.
    """
    # do this for the z=0 case (handling of z>0 can be performed at
    # when calling the routine by multiplying dDM values)
    
    # normalisation constant of a normal distribution.
    # Normalisation should probably be redone afterwards
    # anyway.
    norm = (2.0 * np.pi) ** -0.5 / logsigma

    # csum=pdm
    # imax=ndm
    # if quick:
    if plot or quick:
        # does not integrate, takes central values, here in linear space- tiny bias
        dmmeans = np.linspace(ddm / 2.0, ndm * ddm - ddm / 2.0, ndm)
        logdmmeans = np.log(dmmeans)
        dlogs = ddm / dmmeans
        m1 = (
            loglognormal_dlog(logdmmeans, logmean, logsigma, norm) * dlogs
        )  # worst errors in lowest bins
    # else:
    if plot or not quick:
        m2 = np.zeros([ndm])
        args = (logmean, logsigma, norm)
        # performs integration of first bin in log space
        # Does this for the first bin: probability from
        # "0" (-logsigma*10) to ddm*0.5
        pdm, err = sp.integrate.quad(
            loglognormal_dlog,
            np.log(ddm * 0.5) - logsigma * 10,
            np.log(ddm * 0.5),
            args=args,
        )
        m2[0] = pdm
        
        # performs the integration for all other bins;
        # goes from lower to upper bin bounds 
        for i in np.arange(1, ndm):
            # if csum > CSUMCUT:
            #    imax=i
            #    break
            dmmin = (i - 0.5) * ddm
            dmmax = dmmin + ddm
            pdm, err = sp.integrate.quad(
                loglognormal_dlog, np.log(dmmin), np.log(dmmax), args=args
            )
            m2[i] = pdm
    if quick:
        mask = m1
    else:
        mask = m2
    if plot:
        plt.figure()
        plt.plot(dmmeans, m2, label="quick")
        plt.plot(dmmeans, mask, label="slow")
        plt.xlabel("DM")
        plt.ylabel("p(DM)")
        plt.legend()
        plt.xlim(0, 1000)
        plt.tight_layout()
        plt.savefig("dm_mask_comparison_plot.pdf")
        plt.close()
        print("Generated plot of dm masks, exiting...")
        # Quit to avoid infinite plots. This is just a saftey measure.
        exit()
    return mask
