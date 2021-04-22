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
DEF_H0 = 67.4 #km s^-1 Mpc^-1 #planck


# default value for calculation of cosmological distance measures
DEF_ZMIN=0
DEF_ZMAX=10
DEF_NZ=1000
DZ=(DEF_ZMAX-DEF_ZMIN)/(DEF_NZ-1.)


c_light_kms=constants.c/1e3
Omega_m=DEF_Omega_m
Omega_k=DEF_Omega_k
Omega_lambda=DEF_Omega_lambda
H0=DEF_H0
DH=c_light_kms/H0

# dummy variables so Python recognises them as being global
dms=1
das=1
dls=1
zs=1
dvs=1

# tracks whether or not this module has been initialised
INIT=False


def print_cosmology():
	""" Print cosmological parameters
	
	The current values of cosmological parameters are printed.
	"""
	print("Hubble constant in current epoch, H0: ",H0," [km/s/Mpc]")
	print("Hubble constant in current epoch, H0: ",H0," [km/s/Mpc]")

def set_cosmology(H0=DEF_H0,Omega_k=DEF_Omega_k,
	Omega_lambda=DEF_Omega_lambda, Omega_m=DEF_Omega_m):
	""" Sets cosmological constants
	
	This routine allows the user to set various relevant cosmological
	parameters, within the framework of lambda CDM cosmology.
	It uses a stupid trick to get around having to re-define
	variable names to set global variables with a function.
	"""
	
	stupid_trick(Omega_k,Omega_lambda,Omega_m,H0)

def stupid_trick(a,b,c,d):
	global Omega_k,Omega_lambda,Omega_m,H0,DH,c_light_kms
	Omega_k=a
	Omega_lambda=b
	Omega_m=c
	H0=d
	DH=c_light_kms/H0



# Routines to accurately evaluate cosmological
# parameters. They are designed to be
# sufficiently accurate but potentially
# slow.


def H(z):
	"""Hubble parameter (km/s/Mpc)"""
	return E(z)*H0


def E(z):
	"""scale factor, assuming a simplified cosmology."""
	a=1.+z #inverse scale factor
	return (Omega_m*a**3+Omega_k*a**2+Omega_lambda)**0.5


def inv_E(z):
	""" inverse of scale factor"""
	return E(z)**-1


def DM(z):
	"""comoving distance [Mpc]"""
	res,err=integrate.quad(inv_E,0,z)
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
	return DH*(1+z)**2*DA(z)**2/E(z)


def dVdtau(z):
	""" cosmological time-volume element [Mpc^3 /redshift /sr]
	it is weighted by an extra (1+z) factor to reflect the rate
	in the rest frame vs the observer frame
	"""
	return DH*(1+z)**2*DA(z)**2/E(z)

#################### SECTION 2 ###################
# these functions are numerical, and require initialisation


def init_dist_measures(this_ZMIN=DEF_ZMIN,this_ZMAX=DEF_ZMAX,this_NZ=DEF_NZ):
	""" Initialises cosmological distance measures.
	
	Fills in look-up tables that can operate on input numpy arrays.
	For speed.
	It will use default values of cosmological parameters if
	nothing has yet been specified.
	"""
	
	#sets up initial grid of DMZ
	global zs, das, dms, dls, dvs,DZ
	global ZMAX,ZMIN,NZ
	global INIT
	NZ=this_NZ
	ZMAX=this_ZMAX
	ZMIN=this_ZMIN
	DZ=(ZMAX-ZMIN)/(NZ-1.)
	
	zs=np.linspace(ZMIN,ZMAX,NZ)
	dms=np.zeros([NZ])
	dvs=np.zeros([NZ])
	for i,z in enumerate(zs):
		dms[i]=DM(z)
		dvs[i]=dV(z)
	das=dms/(1.+zs)
	dls=dms*(1.+zs)
	INIT=True
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
	return (dvs[iz]*(1.-kz)+dvs[iz+1]*kz)/(1.+z)

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
def F_to_E(F,z,alpha=0, bandwidth=1e9):
	""" Converts a fluence to an energy
	Formula from Macquart & Ekers 2018
	Bandwidth in Hz
	Spectrum: F(\nu)~\nu^\alpha
	Fluence assumed to be in Jy ms
	Energy assumed to be in erg
	"""
	E=F*4*np.pi*(dl(z))**2/(1.+z)**(2.-alpha)
	# now convert from dl in MPc and F in Jy ms
	# 10^-26 from Jy to W per m2 per Hz
	# 1e-3 from Jy ms to J per m2 per Hz
	# (3.086e16 m in 1 pc x 10^6 Mpc)^2 for dl in m
	# 1e7 from J to erg
	# total factor is 9.523396e22
	E *= 9.523396e22*bandwidth
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

# returns a population density proportional to the star-formation rate to the power n
#Madau & dickenson 2014
# units: solar masses per year per cubic Mpc
def sfr(z):
	return 1.0025738*(1+z)**2.7 / (1 + ((1+z)/2.9)**5.6)
