########### P(DM|z) ############

# this library implements Eq 33 from
# Macquart et al (Nature, submitted)
# It also includes other functions and so on and so forth related to that work
# Imported by C.W. James


#############################
from fcntl import F_ADD_SEALS
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
    """ Equation 4 page 33
    
    delta: = DM_cosmic/ <DM_cosmic>
           i.e. fractional cosmic DM
    alpha, beta: fitted parameters
    
    constraints: std dev must be f*z^0.5
    
    A: arbitrary amplitude of relative probability, ignore...
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
    """ 
    Calculates probability distribution of delta DM as function of feedback
    redshift and the constant C0
    """
    if not deltas:
        deltas = np.linspace(dmin, dmax, ndelta)
    pdeltas = pcosmic(deltas, z, F, C0)
    return deltas, pdeltas


def iterate_C0(z, F, C0=1, Niter=10):
    """ Iteratively solves for C_0 as a function of z and F """
    dmin = 1e-3
    dmax = 10
    ndelta = 10000
    deltas = np.linspace(dmin, dmax, ndelta)
    for i in np.arange(Niter):
        pdeltas = pcosmic(deltas, z, F, C0)
        bin_w = deltas[1] - deltas[0]
        norm = bin_w * np.sum(pdeltas)
        mean = bin_w * np.sum(pdeltas * deltas) / norm
        C0 += mean - 1.0

    return C0


def make_C0_grid(zeds, F):
    """ Pre-generates normalisation constants for C0
    Does this from a grid of z and a given F
    """
    C0s = np.zeros([zeds.size])
    for i, z in enumerate(zeds):
        C0s[i] = iterate_C0(z, F)
    return C0s


def get_mean_DM(zeds: np.ndarray, state: parameters.State):
    """ Gets mean average z to which can be applied deltas 

    Args:
        zeds (np.ndarray): redshifts (must be linearly spaced)
        state (parameters.State): 

    Returns:
        np.ndarray: DM_cosmic
    """
    # Generate the cosmology
    cosmo = FlatLambdaCDM(
        H0=state.cosmo.H0, Ob0=state.cosmo.Omega_b, Om0=state.cosmo.Omega_m
    )
    #
    zmax = zeds[-1]
    nz = zeds.size
    DMbar, zeval = igm.average_DM(zmax, cosmo=cosmo, cumul=True, neval=nz + 1)

    # Check
    assert np.allclose(zeds, zeval[1:])
    # wrong dimension
    return DMbar[1:].value


def get_log_mean_DM(zeds: np.ndarray, state: parameters.State):
    """ Gets mean average z to which can be applied deltas
        Does NOT assume that the zeds are linearly spaced. 

    Args:
        zeds (np.ndarray): redshifts (any order/spacing).
        state (parameters.State): 

    Returns:
        np.ndarray: DM_cosmic
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
    """ Takes a pre-generated table of C0 values,
    and calculates the p(DM) distribution based on this """

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
    """ Gets pDM for an arbitrary z value 
    
    zlog (bool): True if zs are log-spaced
                     False if linearly spaced
    """
    C0 = get_C0(z, F, zgrid, Fgrid, C0grid)
    if zlog:
        DMbar = get_mean_DM(z)
    else:
        DMbar = get_log_mean_DM(z)
    deltas = DMgrid / DMbar  # in units of fractional DM
    pDM = pcosmic(deltas, z, F, C0)
    return pDM


def get_pDM_grid(
    state: parameters.State, DMgrid, zgrid, C0s, verbose=False, zlog=False
):
    """ Gets pDM when the zvals are the same as the zgrid
    state
    C0grid: C0 values obtained by convergence
    DMgrid: range of DMs for which we are generating a histogram
    zgrid: redshifts. These do not have to be in any particular
        order or spacing.
    zlog (bool): True if zs are log-spaced
                 False if linearly spaced
    
    
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


#### defines DMx (excess contribution) ####
# family of lognormal curves. These tae either linear or logarithmic arguments,
# and either do or do not include the 1/x fatcor when converting log to dlin.
# the DM part is because integral p(logDM)dlogDM = 1
# DM is normal, logmean and logsigma are natural logs of these parameters
def linlognormal_dlin(DM, *args):
    """ x values are in linear space,
    args in logspace,
    returns p dx """
    logmean = args[0]
    logsigma = args[1]
    logDM = np.log(DM)
    norm = (2.0 * np.pi) ** -0.5 / DM / logsigma
    return norm * np.exp(-0.5 * ((logDM - logmean) / logsigma) ** 2)


def loglognormal_dlog(logDM, *args):
    """x values, mean and sigma are already in logspace
    returns p dlogx
    That is, this function is simply a Gaussian,
    and the arguments happen to be in log space.
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
    """ Generates a mask over which to integrate the lognormal
    Apply this mask as DM[i] = DM[set[i]]*mask[i]
    DMvals: these give local probabilities of p(DM).
    We simply assign lognormal values at the midpoints
    The renormalisation constants then give some idea of the
    error in this procedure
    This requires parameters to be passed as a vector
    
    
    dmvals: numpy array storing the dms over which to calculate the mask
    
    params [vector, 2]: mean and sigma of the lognormal (log10) distribution
    """

    if len(params) != 2:
        raise ValueError(
            "Incorrect number of DM parameters!",
            params,
            " (expected log10mean, log10sigma)",
        )
        exit()
    # expect the params to be log10 of actual values for simplicity
    # this converts to natural log
    logmean = params[0] / 0.4342944619
    logsigma = params[1] / 0.4342944619

    ddm = dmvals[1] - dmvals[0]

    ##### first generates a mask from the lognormal distribution #####
    # in theory allows a mask up to length of the DN values, but will
    # get truncated
    # the first value has half weight (0 to 0.5)
    # the rest have width of 1
    mask = np.zeros([dmvals.size])
    if zvals is not None:
        ndm = dmvals.size
        nz = zvals.size
        mask = np.zeros([nz, ndm])
        for j, z in enumerate(zvals):
            # with each redshift, we reduce the effects of a 'host' contribution by (1+z)
            # this means that we divide the value of logmean by 1/(1+z)
            # or equivalently, we multiply the ddm by this factor
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
    """
    Assigns probabilities of DM smearing (e.g. due to the host galaxy contribution)
    to a histogram in dm space.
    
    Two methods: quick (use central values of DM bins), and slow (integrate bins)
    
    Arguments:
    
    ddm (float) [pc/cm3]: spacing of dm bins
    
    ndm (int): number of dm bins. Bins assumed to start at 0
    
    logmean: natural logarithm of the mean of the DM distribution
    
    logsigma: sigma in natural log space of DM distribution
    
    quick (bool): True uses the speedup, False takes things slowly
    
    plot (bool): If True, compares quick and slow methods, then exits
        to avoid generating infinite plots.
    
    Returns:
        mask (np.ndarray)
    """
    # do this for the z=0 case

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
        pdm, err = sp.integrate.quad(
            loglognormal_dlog,
            np.log(ddm * 0.5) - logsigma * 10,
            np.log(ddm * 0.5),
            args=args,
        )
        m2[0] = pdm

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
        exit()  # quit to avoid infinite plots
    return mask
