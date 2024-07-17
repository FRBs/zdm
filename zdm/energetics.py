import numpy as np
from scipy import interpolate
import mpmath
from astropy.cosmology import Planck18 as cosmo

from IPython import embed

igamma_splines = {}
igamma_linear = {}
igamma_linear_log10 = {}
SplineMin = -6
SplineMax = 6
NSpline = 1000

############## this section defines different luminosity functions ##########

def init_igamma_splines(gammas, reinit=False):
    global SplineMin,SplineMax,NSpline
    for gamma in gammas:
        if gamma not in igamma_splines.keys() or reinit:
            print(f"Initializing igamma_spline for gamma={gamma}")
            avals = 10**np.linspace(SplineMin, SplineMax, NSpline)
            numer = np.array([float(mpmath.gammainc(
                gamma, a=iEE)) for iEE in avals])
            # iGamma
            igamma_splines[gamma] = interpolate.splrep(avals, numer,k=3)

def init_igamma_linear(gammas:list, reinit:bool=False, 
                       log:bool=False):
    """ Setup the linear interpolator for gamma

    Args:
        gammas (list): values of gamma
        reinit (bool, optional): If True, redo the calculation.
        log (bool, optional): Perform in log10 space
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

############## this section defines different luminosity functions ##########

########### simple power law functions #############

def vector_cum_power_law(Eth,*params):
    """ Calculates the fraction of bursts above a certain power law
    for a given Eth.
    """
    #params=np.array(params)
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

    
def array_cum_power_law(Eth,*params):
    """ Calculates the fraction of bursts above a certain power law
    for a given Eth, where Eth is an N-dimensional array
    """
    dims=Eth.shape
    Eth=Eth.flatten()
    #if gamma >= 0: #handles crazy dodgy cases. Or just return 0?
    #    result=np.zeros([Eth.size])
    #    result[np.where(Eth < Emax)]=1.
    #    result=result.reshape(dims)
    #    Eth=Eth.reshape(dims)
    #    return result
    result=vector_cum_power_law(Eth,*params)
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


########### gamma functions #############

def vector_cum_gamma(Eth,*params):
    """ Calculates the fraction of bursts above a certain gamma function
    for a given Eth.
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

def vector_cum_gamma_spline(Eth:np.ndarray, *params):
    """ Calculate cumulative Gamma function using a spline

    Args:
        Eth (np.ndarray): [description]

    Returns:
        np.ndarray: [description]
    """
    params=np.array(params)
    Emin=params[0]
    Emax=params[1]
    gamma=params[2]

    # Calculate
    norm = float(mpmath.gammainc(gamma, a=Emin/Emax))
    Eth_Emax = Eth/Emax
    if gamma not in igamma_splines.keys():
        init_igamma_splines([gamma])
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

def vector_diff_gamma(Eth,*params):
    """ Calculates the differential fraction of bursts for a gamma function
    """
    Emin=params[0]
    Emax=params[1]
    gamma=params[2]
    
    norm = Emax*float(mpmath.gammainc(gamma, a=Emin/Emax))
    result= (Eth/Emax)**(gamma-1) * np.exp(-Eth/Emax) / norm
    
    low= Eth < Emin
    result[low]=0.
    return result

########### lensing modified #######

def distanceFraction(zD, zS):
    Dds = cosmo.angular_diameter_distance_z1z2(zD,zS)
    Ds = cosmo.angular_diameter_distance(zS)
    return Dds/Ds

def lensingPDF(mu, zD, zS, bPosNum, beami, opdir):
    formatted_redshift = "{:03.2f}".format(zS)
    x = np.load(opdir+'mux.npy')
    yFull = np.load(opdir+'pmux_BP_'+str(bPosNum)+str(formatted_redshift)+'.npy')
    if np.sum(np.isnan(yFull[:,beami]))==len(yFull[:,beami]):
        return np.ones(len(mu))*np.nan
    zMap = 9
    y = yFull[:,beami]
    interpFunc = interpolate.interp1d(x,y, bounds_error=False, fill_value=0)
    return interpFunc(np.log10(mu))

def vector_cum_lensed_power_law(Eth,*params):
    """ Calculates the fraction of bursts above a certain power law
    for a given Eth.
    """
    #params=np.array(params)
    Emin=params[0]
    Emax=params[1]
    gamma=params[2]
    zvals=params[4]
    beami = params[5]
    surveyName = params[6]
    zD = params[7]
    opdir = params[8]
    bPosNum = params[9]
    #print(Eth, Emin, Emax, gamma)
    logEn = np.log(Emin)
    logEx = np.log(Emax)
    logSpacing = 0.01
    logERange = np.arange(logEn, logEx+logSpacing, logSpacing)
    muNum = int((10+2)/logSpacing)+1
    #print(muNum)
    logMu = np.arange(-2, muNum*logSpacing, logSpacing)
    #print(len(logERange), len(logMu))
    result = np.zeros(Eth.shape)
    for i in range(len(zvals)):
        if zD < zvals[i]:
            probGrid = lensingPDF(np.e**logMu, zD, zvals[i], bPosNum, beami, opdir)
            if np.sum(np.isnan(probGrid))==len(probGrid):
                result[i,:]=vector_cum_power_law(Eth[i,:],*params)
            else:
                phiGrid = vector_diff_power_law(np.e**logERange, *params)
                phiL = np.convolve(probGrid, phiGrid)*logSpacing
                logE_muRange = np.arange(logEn-2, np.amax(logERange)+np.amax(logMu),logSpacing)
                #print(np.amin(np.e**logE_muRange), np.amax(np.e**logE_muRange))
                phiLCumConv = np.cumsum(np.flip((np.e**logE_muRange)*phiL*logSpacing))
                interpFunc = interpolate.interp1d(np.flip(np.e**logE_muRange), phiLCumConv, bounds_error=False, fill_value=(1.0,0.0))
                #iprint(interpFunc(1e31),interpFunc(1e32))
                result[i,:] = interpFunc(Eth[i,:])
        else:
            result[i,:]=vector_cum_power_law(Eth[i,:],*params)

    return result



def array_cum_lensed_power_law(Eth,*params):
    """ Calculates the fraction of bursts above a certain power law
    for a given Eth, where Eth is an N-dimensional array
    """
    dims=Eth.shape
    #if gamma >= 0: #handles crazy dodgy cases. Or just return 0?
    #    result=np.zeros([Eth.size])
    #    result[np.where(Eth < Emax)]=1.
    #    result=result.reshape(dims)
    #    Eth=Eth.reshape(dims)
    #    return result
    result=vector_cum_lensed_power_law(Eth,*params)
    return result
