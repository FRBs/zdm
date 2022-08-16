import numpy as np
from scipy import interpolate
import mpmath

from IPython import embed

igamma_splines = {}
igamma_linear = {}
igamma_linear_log10 = {}

############## this section defines different luminosity functions ##########

def init_igamma_splines(gammas, reinit=False):
    for gamma in gammas:
        if gamma not in igamma_splines.keys() or reinit:
            print(f"Initializing igamma_spline for gamma={gamma}")
            # values
            avals = 10**np.linspace(-6, 6., 1000)
            numer = np.array([float(mpmath.gammainc(
                gamma, a=iEE)) for iEE in avals])
            # iGamma
            igamma_splines[gamma] = interpolate.splrep(avals, numer)

def init_igamma_linear(gammas, reinit=False, log=False):

    for gamma in gammas:
        
        if log:
            if gamma not in igamma_linear_log10.keys() or reinit:
                print(f"Initializing igamma_linear for gamma={gamma} with log10")
                 # values
                avals = 10**np.linspace(-6, 6., 1000)

                numer = np.array([float(mpmath.gammainc(
                    gamma, a=iEE)) for iEE in avals])

                log_avals = np.log10(avals)

                igamma_linear_log10[gamma] = interpolate.interp1d(log_avals, numer)

        else:
            if gamma not in igamma_linear.keys() or reinit:    
                print(f"Initializing igamma_linear for gamma={gamma}")
                 # values
                avals = 10**np.linspace(-6, 6., 1000)

                numer = np.array([float(mpmath.gammainc(
                    gamma, a=iEE)) for iEE in avals])

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
        result[low]=1.  # This was 0 and I think it was wrong -- JXP
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
    low= Eth < Emin
    result[low]=1.
    return result

def vector_cum_gamma_linear(Eth:np.ndarray, *params):
    """ Calculate cumulative Gamma function using a linear interp1d

    Args:
        Eth (np.ndarray): [description]

    Returns:
        np.ndarray: [description]
    """
    params=np.array(params)
    Emin=params[0]
    Emax=params[1]
    gamma=params[2]
    log = params[3]

    # Calculate
    norm = float(mpmath.gammainc(gamma, a=Emin/Emax))

    if log:
        Eth_Emax = Eth - np.log10(Emax)
        if gamma not in igamma_linear_log10.keys():
            init_igamma_linear([gamma], log=log)
        # embed()
        numer = igamma_linear_log10[gamma](Eth_Emax)

    else:
        Eth_Emax = Eth/Emax
        if gamma not in igamma_linear.keys():
            init_igamma_linear([gamma], log=log)
        
        numer = igamma_linear[gamma](Eth_Emax)
    
    result=numer/norm

    # Low end

    if log:
        Eth = 10**Eth

    low= Eth < Emin
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
    result[low]=1.  # This was 0 and I think it was wrong
    
    return result
