import numpy as np
from zdm import cosmology as cos
import time

############## this section defines different luminosity functions ##########

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

#def array_power_law2(Eth,Emin,Emax,gamma):		
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

########### simple power law functions #############
    
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


############### unused - to delete ##########
# power-laws here are differential
#def power_law_norm(Emin,Emax,gamma):
#	""" Calculates the normalisation factor for a power-law """
#	return Emin**gamma-Emax**-gamma

#def power_law(Eth,Emin,Emax,gamma):
#	""" Calculates the fraction of bursts above a certain power law
#	for a given Eth.
#	"""
#	if Eth <= Emin:
#		return 1
#	elif Eth >= Emax:
#		return 0
#	else:
#		return (Eth**gamma-Emax**gamma ) / (Emin**gamma-Emax**gamma )


######### misc function to load some data - do we ever use it? ##########

def load_data(filename):
    if filename.endswith('.npy'):
        data=np.load(filename)
    elif filename.endswith('.txt') or filename.endswith('.txt'):
        # assume a simple text file with whitespace separator
        data=np.loadtxt(filename)
    else:
        raise ValueError('unrecognised type on z-dm file ',filename,' cannot read data')
    return data


