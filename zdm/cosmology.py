"""
Cosmology module for the zdm package.

This module implements standard Lambda CDM cosmology calculations for
Fast Radio Burst (FRB) redshift-dispersion measure analysis. It provides
functions for computing cosmological distance measures, volume elements,
and energy-fluence conversions.

The module uses a convention where the spectral index alpha is defined such
that F_nu ~ nu^(-alpha), i.e., positive alpha corresponds to steeper spectra
at higher frequencies.

Key Features
------------
- Hubble parameter and scale factor calculations
- Distance measures: comoving (DM), angular diameter (DA), luminosity (DL)
- Cosmological volume elements (dV) and time-volume elements (dVdtau)
- Interpolated lookup tables for fast array operations
- Energy-fluence conversions for FRB analysis
- Source evolution functions (SFR, power-law)

Usage
-----
Before using interpolated functions (dm, da, dl, dv, dvdtau), call
`init_dist_measures()` to populate the lookup tables. The cosmology
can be configured via `set_cosmology()`.

Example
-------
>>> from zdm import cosmology as cos
>>> cos.init_dist_measures()
>>> z = 0.5
>>> d_L = cos.dl(z)  # Luminosity distance in Mpc

Author: Clancy W. James (clancy.w.james@gmail.com)
"""


import scipy.constants as constants
import numpy as np
import scipy.integrate as integrate
from zdm.zdm import parameters



'''
#### defines some default cosmological parameters ###
#cosmological parameters to use here 
# check are these Wmap
# photo density. Ignored here (we do not go back far enough)
DEF_Omega_k=0.
# dark energy / cosmological constant (in current epoch)
DEF_Omega_lambda=0.685
# matter density in current epoch
DEF_Omega_m=0.315 #Plank says 0.315
# baryon density
DEF_Omega_b=0.044
DEF_Omega_b_h2=0.0224 #Planck says 0.0224, WMAP 0.02264
# hubble constant in current epoch
DEF_H0 = igm.Planck15.H0.value #km s^-1 Mpc^-1 #planck15 used in frb.igm
'''


# default value for calculation of cosmological distance measures
DEF_ZMIN=0
DEF_ZMAX=10
DEF_NZ=1000
DZ=(DEF_ZMAX-DEF_ZMIN)/(DEF_NZ-1.)


c_light_kms=constants.c/1e3

'''
Omega_m=DEF_Omega_m
Omega_k=DEF_Omega_k
Omega_lambda=DEF_Omega_lambda
H0=DEF_H0
DH=c_light_kms/H0
'''

# dummy variables so Python recognises them as being global
dms=1
das=1
dls=1
zs=1
dvs=1
dvdtaus=1 

# tracks whether or not this module has been initialised
INIT=False

cosmo = parameters.CosmoParams()


def print_cosmology(params):
    """Print the current cosmological parameters.

    Parameters
    ----------
    params : dict or State
        Parameter dictionary or State object containing 'cosmo' key
        with CosmoParams object.
    """
    print("Hubble constant in default cosmology, H0: ",params['cosmo'].H0," [km/s/Mpc]")
    #print("Hubble constant in current epoch, H0: ",params['cosmo'].current_H0," [km/s/Mpc]")

def set_cosmology(params):
    """Set the global cosmology parameters for this module.

    This function must be called before using distance measure functions
    if non-default cosmology is desired.

    Parameters
    ----------
    params : dict or State
        Parameter dictionary or State object containing 'cosmo' key
        with CosmoParams object specifying H0, Omega_m, Omega_lambda, Omega_k.
    """
    global cosmo
    cosmo = params['cosmo']

'''
def set_cosmology(H0=DEF_H0,Omega_k=DEF_Omega_k,
    Omega_lambda=DEF_Omega_lambda, Omega_m=DEF_Omega_m):
    """ Sets cosmological constants
    
    This routine allows the user to set various relevant cosmological
    parameters, within the framework of lambda CDM cosmology.
    It uses a stupid trick to get around having to re-define
    variable names to set global variables with a function.
    """
    
    stupid_trick(Omega_k,Omega_lambda,Omega_m,H0)
    init_dist_measures(this_ZMIN=DEF_ZMIN,this_ZMAX=DEF_ZMAX,this_NZ=DEF_NZ)
    #added init_dist_measure to be called from set_cosmology

def stupid_trick(a,b,c,d):
    global Omega_k,Omega_lambda,Omega_m,H0,DH,c_light_kms
    Omega_k=a
    Omega_lambda=b
    Omega_m=c
    H0=d
    DH=c_light_kms/H0
'''


# Routines to accurately evaluate cosmological
# parameters. They are designed to be
# sufficiently accurate but potentially
# slow.


def H(z):
    """Hubble parameter (km/s/Mpc)

    Args:
        z (float): redshift

    Returns:
        float: Hubble parameter [km/s/Mpc]
    """
    return E(z)*cosmo.H0


def E(z):
    """Dimensionless Hubble parameter E(z) = H(z)/H0.

    Computes the normalized expansion rate assuming flat Lambda CDM cosmology.

    Parameters
    ----------
    z : float or array_like
        Redshift(s) at which to evaluate E(z).

    Returns
    -------
    float or ndarray
        Dimensionless Hubble parameter E(z) = sqrt(Omega_m*(1+z)^3 + Omega_k*(1+z)^2 + Omega_lambda).
    """
    a=1.+z #inverse scale factor
    return (cosmo.Omega_m*a**3+cosmo.Omega_k*a**2+cosmo.Omega_lambda)**0.5


def inv_E(z):
    """Inverse of dimensionless Hubble parameter, 1/E(z).

    Used as integrand for computing comoving distance.

    Parameters
    ----------
    z : float
        Redshift.

    Returns
    -------
    float
        1/E(z), where E(z) is the dimensionless Hubble parameter.
    """
    return E(z)**-1


def DM(z):
    """Comoving distance (line-of-sight) via numerical integration.

    This is the slow but accurate version. For array operations,
    use the interpolated version `dm()` after calling `init_dist_measures()`.

    Parameters
    ----------
    z : float
        Redshift.

    Returns
    -------
    float
        Comoving distance in Mpc.
    """
    res,err=integrate.quad(inv_E,0,z)
    DH=c_light_kms/cosmo.H0
    return DH*res

def DA(z):
    """Angular diameter distance via numerical integration.

    This is the slow but accurate version. For array operations,
    use the interpolated version `da()` after calling `init_dist_measures()`.

    Parameters
    ----------
    z : float
        Redshift.

    Returns
    -------
    float
        Angular diameter distance in Mpc.
    """
    return DM(z)/(1+z)


def DL(z):
    """Luminosity distance via numerical integration.

    This is the slow but accurate version. For array operations,
    use the interpolated version `dl()` after calling `init_dist_measures()`.

    Parameters
    ----------
    z : float
        Redshift.

    Returns
    -------
    float
        Luminosity distance in Mpc.
    """
    return DM(z)*(1+z)

def dV(z):
    """Comoving volume element per unit redshift per steradian.

    Computes dV/dz/dOmega, the differential comoving volume element.
    This does not include any time dilation factor for rate conversions.

    Parameters
    ----------
    z : float
        Redshift.

    Returns
    -------
    float
        Volume element in Mpc^3 per unit redshift per steradian.
    """
    DH=c_light_kms/cosmo.H0
    return DH*(1+z)**2*DA(z)**2/E(z)


def dVdtau(z):
    """Time-weighted comoving volume element per unit redshift per steradian.

    Similar to dV(z) but includes an extra (1+z)^-1 factor to account for
    cosmological time dilation when converting between source-frame and
    observer-frame event rates.

    Parameters
    ----------
    z : float
        Redshift.

    Returns
    -------
    float
        Time-weighted volume element in Mpc^3 per unit redshift per steradian.
    """
    DH=c_light_kms/cosmo.H0
    return DH*(1+z)*DA(z)**2/E(z) #changed (1+z)**2 to (1+z)

#################### SECTION 2 ###################
# these functions are numerical, and require initialisation


def init_dist_measures(this_ZMIN=DEF_ZMIN,this_ZMAX=DEF_ZMAX,this_NZ=DEF_NZ,
                       verbose=False):
    """Initialize interpolation tables for fast distance measure calculations.

    This function populates lookup tables for comoving distance, angular
    diameter distance, luminosity distance, and volume elements. After
    initialization, the lowercase functions (dm, da, dl, dv, dvdtau) can
    be used for fast array operations.

    Must be called before using the interpolated distance functions.
    Uses current cosmological parameters set via `set_cosmology()`.

    Parameters
    ----------
    this_ZMIN : float, optional
        Minimum redshift for interpolation grid. Default is 0.
    this_ZMAX : float, optional
        Maximum redshift for interpolation grid. Default is 10.
    this_NZ : int, optional
        Number of redshift points in interpolation grid. Default is 1000.
    verbose : bool, optional
        If True, print confirmation message. Default is False.
    """
    
    #sets up initial grid of DMZ
    global zs, das, dms, dls, dvs,DZ, dvdtaus
    global ZMAX,ZMIN,NZ
    global INIT
    NZ=this_NZ
    ZMAX=this_ZMAX
    ZMIN=this_ZMIN
    DZ=(ZMAX-ZMIN)/(NZ-1.)
    
    zs=np.linspace(ZMIN,ZMAX,NZ)
    dms=np.zeros([NZ])
    dvs=np.zeros([NZ])
    dvdtaus=np.zeros([NZ])
    for i,z in enumerate(zs):
        dms[i]=DM(z)
        dvs[i]=dV(z)
        dvdtaus[i]=dVdtau(z) #created for iterpolation
    das=dms/(1.+zs)
    dls=dms*(1.+zs)
    INIT=True
    if verbose:
        print("Initialised distance measures")
    

# The following are interpolation functions
# they are designed to operate on np array inputs
# and return corresponding interpolated outputs.
# They are not 'safe', and input bounds should
# be checked independently


def dm(z):
    """Comoving distance via linear interpolation (fast, array-compatible).

    Requires prior call to `init_dist_measures()` to populate lookup tables.

    Parameters
    ----------
    z : float or array_like
        Redshift(s). Must be within [ZMIN, ZMAX] set during initialization.

    Returns
    -------
    float or ndarray
        Comoving distance in Mpc.
    """
    iz=np.array(np.floor(z/DZ)).astype('int')
    kz=z/DZ-iz
    return dms[iz]*(1.-kz)+dms[iz+1]*kz


def dl(z):
    """Luminosity distance via linear interpolation (fast, array-compatible).

    Requires prior call to `init_dist_measures()` to populate lookup tables.

    Parameters
    ----------
    z : float or array_like
        Redshift(s). Must be within [ZMIN, ZMAX] set during initialization.

    Returns
    -------
    float or ndarray
        Luminosity distance in Mpc.
    """
    iz=np.array(np.floor(z/DZ)).astype('int')
    kz=z/DZ-iz
    return dls[iz]*(1.-kz)+dls[iz+1]*kz


def da(z):
    """Angular diameter distance via linear interpolation (fast, array-compatible).

    Requires prior call to `init_dist_measures()` to populate lookup tables.

    Parameters
    ----------
    z : float or array_like
        Redshift(s). Must be within [ZMIN, ZMAX] set during initialization.

    Returns
    -------
    float or ndarray
        Angular diameter distance in Mpc.
    """
    iz=np.array(np.floor(z/DZ)).astype('int')
    kz=z/DZ-iz
    return das[iz]*(1.-kz)+das[iz+1]*kz


def dv(z):
    """Comoving volume element via linear interpolation (fast, array-compatible).

    Requires prior call to `init_dist_measures()` to populate lookup tables.

    Parameters
    ----------
    z : float or array_like
        Redshift(s). Must be within [ZMIN, ZMAX] set during initialization.

    Returns
    -------
    float or ndarray
        Volume element in Mpc^3 per unit redshift per steradian.
    """
    iz=np.array(np.floor(z/DZ)).astype('int')
    kz=z/DZ-iz
    return dvs[iz]*(1.-kz)+dvs[iz+1]*kz


def dvdtau(z):
    """Time-weighted volume element via linear interpolation (fast, array-compatible).

    Includes (1+z)^-1 factor for source-frame to observer-frame rate conversion.
    Requires prior call to `init_dist_measures()` to populate lookup tables.

    Parameters
    ----------
    z : float or array_like
        Redshift(s). Must be within [ZMIN, ZMAX] set during initialization.

    Returns
    -------
    float or ndarray
        Time-weighted volume element in Mpc^3 per unit redshift per steradian.
    """
    iz=np.array(np.floor(z/DZ)).astype('int')
    kz=z/DZ-iz
    return (dvdtaus[iz]*(1.-kz)+dvdtaus[iz+1]*kz) #removed the 1/(1+z) dependency

#################### SECTION 3 ###################
# functions related to FRB scaling


def E_to_F(E,z,alpha=0, bandwidth=1e9):
    """Convert isotropic-equivalent energy to observed fluence.

    Uses the formula from Macquart & Ekers (2018) to convert burst energy
    at the source to observed fluence at the telescope.

    Parameters
    ----------
    E : float or array_like
        Isotropic-equivalent burst energy in erg.
    z : float or array_like
        Redshift of the source.
    alpha : float, optional
        Spectral index where F_nu ~ nu^(-alpha). Default is 0 (flat spectrum).
    bandwidth : float, optional
        Observation bandwidth in Hz. Default is 1e9 (1 GHz).

    Returns
    -------
    float or ndarray
        Fluence in Jy ms.
    """
    F=E/(4*np.pi*(dl(z))**2/(1.+z)**(2.-alpha))
    F /= 9.523396e22*bandwidth # see below for constant calculation
    return F


def F_to_E(F,z,alpha=0, bandwidth=1e9, Fobs=1.3e9, Fref=1.3e9):
    """Convert observed fluence to isotropic-equivalent energy.

    Uses the formula from Macquart & Ekers (2018) to convert observed
    fluence to source-frame burst energy. Supports array inputs for z.

    Parameters
    ----------
    F : float or array_like
        Observed fluence in Jy ms.
    z : float or array_like
        Redshift of the source.
    alpha : float, optional
        Spectral index where F_nu ~ nu^(-alpha). Default is 0 (flat spectrum).
        Note: This uses the convention F ~ nu^(-alpha), opposite to some papers.
    bandwidth : float, optional
        Observation bandwidth in Hz. Default is 1e9 (1 GHz).
    Fobs : float, optional
        Observation frequency in Hz. Default is 1.3e9 (1.3 GHz, ASKAP/Parkes).
    Fref : float, optional
        Reference frequency for energy normalization in Hz. Default is 1.3e9.

    Returns
    -------
    float or ndarray
        Isotropic-equivalent burst energy in erg.

    Notes
    -----
    Unit conversion factor 9.523396e22 accounts for:
    - 10^-26 from Jy to W/m^2/Hz
    - 1e-3 from ms to s
    - (3.086e22 m/Mpc)^2 for distance conversion
    - 1e7 from J to erg
    """
    E=F*4*np.pi*(dl(z))**2/(1.+z)**(2.-alpha)
	# now convert from dl in MPc and F in Jy ms
	# 10^-26 from Jy to W per m2 per Hz
	# 1e-3 from Jy ms to J per m2 per Hz
	# (3.086e16 m in 1 pc x 10^6 Mpc)^2 for dl in m
	# 1e7 from J to erg
	# total factor is 9.523396e22
    E *= 9.523396e22*bandwidth

	# now corrects for reference frequency
	# according to value of alpha
	# effectively: if fluence was X at F0, it was X*(F0/Fref)**alpha at Fref
	# i.e. if alpha is positive (stronger at low frequencies), we reduce E
	# This acts to reduce the telescope threshold at higher frequencies
    E *= (Fobs/Fref)**alpha

    return E


def dFnu_to_dEnu(z,alpha=0,bandwidth=1.e9):
    """Compute the Jacobian dE/dF for fluence-to-energy transformations.

    Useful for converting "per fluence" statistics to "per energy" statistics.
    This is the derivative of energy with respect to fluence at fixed z.

    Parameters
    ----------
    z : float or array_like
        Redshift.
    alpha : float, optional
        Spectral index where F_nu ~ nu^(-alpha). Default is 0.
    bandwidth : float, optional
        Observation bandwidth in Hz. Default is 1e9.

    Returns
    -------
    float or ndarray
        Jacobian dE/dF for the transformation.
    """
    #Fnu is Jy
    #Jy: 1e-26 J/s /m2 /Hz
    #   E in J/s (Mpc/m)^2 /Hz
    dE = 4*np.pi*(dl(z))**2/(1.+z)**(2.-alpha)
    return dE

######### possible source evolution functions go here ##########

def choose_source_evolution_function(which=0):
    """Select a source evolution function for FRB population modeling.

    Returns a function that computes the relative FRB source density
    as a function of redshift. These functions are parameterized to
    allow for different evolution scenarios.

    Parameters
    ----------
    which : int, optional
        Model selection:
        - 0: Star formation rate from Madau & Dickinson (2014) raised to power n.
             SFR(z)^n where SFR follows the cosmic star formation history.
        - 1: Simple power law (1+z)^(2.7*n), without the high-z turnover.
             Useful for comparison with SFR model.
        Default is 0.

    Returns
    -------
    callable
        Source evolution function with signature f(z, n) where z is redshift
        and n is the evolution power parameter.

    Raises
    ------
    ValueError
        If `which` is not 0 or 1.
    """
    if which==0:
        source_evolution=sfr_evolution
    elif which==1:
        source_evolution=opz_evolution
    else:
        raise ValueError("Undefined source evolution function ",which," choose 0 or 1")
    return source_evolution

def sfr_evolution(z,*params):
    """Star formation rate evolution model from Madau & Dickinson (2014).

    Computes the cosmic star formation rate raised to a power n, normalized
    to unity at z=0.

    Parameters
    ----------
    z : float or array_like
        Redshift.
    *params : float
        First parameter is n, the power to which the SFR is raised.
        Typically n=1 for direct SFR tracking.

    Returns
    -------
    float or ndarray
        Relative source density at redshift z, normalized to 1 at z=0.
    """
    return (1.0025738*(1+z)**2.7 / (1 + ((1+z)/2.9)**5.6))**params[0]


def opz_evolution(z,*params):
    """Simple power-law source evolution model.

    Computes (1+z)^(2.7*n), which matches the low-z behavior of the SFR
    model but lacks the high-z turnover. The factor 2.7 ensures that
    n-values are comparable between models.

    Parameters
    ----------
    z : float or array_like
        Redshift.
    *params : float
        First parameter is n, the evolution power parameter.

    Returns
    -------
    float or ndarray
        Relative source density at redshift z.
    """
    return (1+z)**(2.7*params[0])


def sfr(z):
    """Star formation rate density from Madau & Dickinson (2014).

    .. deprecated::
        Use `sfr_evolution(z, 1)` instead for parameterized models.

    Parameters
    ----------
    z : float or array_like
        Redshift.

    Returns
    -------
    float or ndarray
        Star formation rate in solar masses per year per cubic Mpc,
        normalized to approximately 1 at z=0.
    """
    return 1.0025738*(1+z)**2.7 / (1 + ((1+z)/2.9)**5.6)
