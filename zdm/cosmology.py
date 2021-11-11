################ COSMOLOGY.PY ###############

# Author: Clancy W. James
# clancy.w.james@gmail.com

# This file contains functions relating to
# cosmological evaluations. It simply 
# implements standard Lambda CDM cosmology
# written with alpha such that F_nu ~ \nu**-alpha
##############################################


import scipy.constants as constants
import numpy as np
import scipy.integrate as integrate



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

cosmo = None


def print_cosmology(params):
    """ Print cosmological parameters
    
    The current values of cosmological parameters are printed.
    """
    print("Hubble constant in default cosmology, H0: ",params['cosmo'].H0," [km/s/Mpc]")
    print("Hubble constant in current epoch, H0: ",params['cosmo'].current_H0," [km/s/Mpc]")

def set_cosmology(params):
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
    """Hubble parameter (km/s/Mpc)"""
    return E(z)*cosmo.H0


def E(z):
    """scale factor, assuming a simplified cosmology."""
    a=1.+z #inverse scale factor
    return (cosmo.Omega_m*a**3+cosmo.Omega_k*a**2+cosmo.Omega_lambda)**0.5


def inv_E(z):
    """ inverse of scale factor"""
    return E(z)**-1


def DM(z):
    """comoving distance [Mpc]"""
    res,err=integrate.quad(inv_E,0,z)
    DH=c_light_kms/cosmo.H0
    return DH*res

def DA(z):
    """angular diameter distance [Mpc]"""
    return DM(z)/(1+z)


def DL(z):
    """luminosity distance [Mpc]"""
    return DM(z)*(1+z)

#cosmological volume element calculation
# comoving Mpc^3 per dz dOmega
# note this is simply a volume - no rate adjustment
# is taken into account.
def dV(z):
    """ cosmological volume element [Mpc^3 /redshift /sr]"""
    DH=c_light_kms/cosmo.H0
    return DH*(1+z)**2*DA(z)**2/E(z)


def dVdtau(z):
    """ cosmological time-volume element [Mpc^3 /redshift /sr]
    it is weighted by an extra (1+z) factor to reflect the rate
    in the rest frame vs the observer frame
    """
    DH=c_light_kms/cosmo.H0
    return DH*(1+z)*DA(z)**2/E(z) #changed (1+z)**2 to (1+z)

#################### SECTION 2 ###################
# these functions are numerical, and require initialisation


def init_dist_measures(this_ZMIN=DEF_ZMIN,this_ZMAX=DEF_ZMAX,this_NZ=DEF_NZ,
                       verbose=False):
    """ Initialises cosmological distance measures.
    
    Fills in look-up tables that can operate on input numpy arrays.
    For speed.
    It will use default values of cosmological parameters if
    nothing has yet been specified.
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
    """ Comoving distance [Mpc].
    Must have initialised w. init_dist_measures
    """
    iz=np.array(np.floor(z/DZ)).astype('int')
    kz=z/DZ-iz
    return dms[iz]*(1.-kz)+dms[iz+1]*kz


def dl(z):
    """ Luminosity distance [Mpc].
    Must have initialised w. init_dist_measures
    """
    iz=np.array(np.floor(z/DZ)).astype('int')
    kz=z/DZ-iz
    return dls[iz]*(1.-kz)+dls[iz+1]*kz


def da(z):
    """ Angular diameter  distance [Mpc].
    Must have initialised w. init_dist_measures
    """
    iz=np.array(np.floor(z/DZ)).astype('int')
    kz=z/DZ-iz
    return das[iz]*(1.-kz)+das[iz+1]*kz


def dv(z):
    """ Comoving volume element [Mpc^3 dz sr^-1].
    Must have initialised w. init_dist_measures
    """ 
    iz=np.array(np.floor(z/DZ)).astype('int')
    kz=z/DZ-iz
    return dvs[iz]*(1.-kz)+dvs[iz+1]*kz


def dvdtau(z):
    """ Comoving volume element [Mpc^3 dz sr^-1].
    normalsied by proper time (1+z)^-1 to convert
    rates.
    """ 
    iz=np.array(np.floor(z/DZ)).astype('int')
    kz=z/DZ-iz
    return (dvdtaus[iz]*(1.-kz)+dvdtaus[iz+1]*kz) #removed the 1/(1+z) dependency

#################### SECTION 3 ###################
# functions related to FRB scaling


def E_to_F(E,z,alpha=0, bandwidth=1e9):
    """ Converts an energy to a fluence
    Formula from Macquart & Ekers 2018
    Energy is assumed to be in ergs
    Fluence returned in Jy ms
    """
    F=E/(4*np.pi*(dl(z))**2/(1.+z)**(2.-alpha))
    F /= 9.523396e22*bandwidth # see below for constant calculation
    return F


# inverse of above
def F_to_E(F,z,alpha=0, bandwidth=1e9, Fobs=1.3e9, Fref=1.3e9):
	""" Converts a fluence in Jy ms to an energy in erg
	Formula from Macquart & Ekers 2018
	Works with an array of z.
	
	Arguments are:
		Fluence: of an FRB [Jy ms]
		
		Redshift: assumed redshift of an FRB producing the fluence F.
			Standard cosmological definition [unitless]
		
		alpha: F(\nu)~\nu^-\alpha. Note that this is an internal definition.
			The paper uses ^alpha, not ^-alpha. [unitless]
	
		Bandwidth: over which to integrate fluence [Hz] 
		
		Fobs: the observation frequency [Hz]
		
		Fref: reference frequency at which FRB energies E are normalised.
			It defaults to 1.3 GHz (ASKAP lat50, Parkes).
	
	Return value: energy [erg]
	
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


# calculates the 'bin size scaling factor'
# essentially differentiates the above expression
# for observed dF to emitted dF
# does this include the rate factor??? NO!
def dFnu_to_dEnu(z,alpha=0,bandwidth=1.e9):
    """ Converts differential dF to differential dE
    Good for probing "per fluence" stats to "per energy"
    """
    #Fnu is Jy
    #Jy: 1e-26 J/s /m2 /Hz
    #   E in J/s (Mpc/m)^2 /Hz
    dE = 4*np.pi*(dl(z))**2/(1.+z)**(2.-alpha)
    return dE

######### possible source evolution functions go here ##########

def choose_source_evolution_function(which=0):
    """
    Selects which source evolution function to use
    These are now generalised to take multiple parameters
    Could implement arbitrarily many of these
    
    Arguments:
        which (int). Selects which pre-defined model
            to use for FRB source evolution.
            Currently implemented values are:
            0: star-formation rate from Madau
                & Dickenson, to the power n
            1: (1+z)^2.7n, i.e. 0 but without the
                denominator
            
    """
    if which==0:
        source_evolution=sfr_evolution
    elif which==1:
        source_evolution=opz_evolution
    else:
        raise ValueError("Undefined source evolution function ",which," choose 0 or 1")
    return source_evolution

def sfr_evolution(z,*params):
    """
    Madau & dickenson 2014
    Arguments:
        z (float): redshift
        params: n (float) Scaling parameter.
    """
    return (1.0025738*(1+z)**2.7 / (1 + ((1+z)/2.9)**5.6))**params[0]
    

def opz_evolution(z,*params):
    """
    Same as SFR, but without denominator, i.e. just (1+z)**2.7
    Factor of 2.7 is kept so that resulting n-values are comparable
    Arguments:
        z:(float, numpy array) redshift 
        params: n (float) Scaling parameter.
    """
    return (1+z)**(2.7*params[0])



# outdated code
# returns a population density proportional to the star-formation rate to the power n
#Madau & dickenson 2014
# units: solar masses per year per cubic Mpc
def sfr(z):
    return 1.0025738*(1+z)**2.7 / (1 + ((1+z)/2.9)**5.6)
