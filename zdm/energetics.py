import numpy as np
from scipy import interpolate
import mpmath

from IPython import embed

igamma_splines = {}
igamma_linear = {}
igamma_linear_log10 = {}
SplineMin = -6
SplineMax = 6
NSpline = 1000
SplineLog = True

############## this section defines different luminosity functions ##########

def init_igamma_splines(gammas, reinit=False,k=3):
    """
    gammas [list of floats]: list of values of gamma at which splines
        must be created
    reinit [bool]: if True, will re-initialise even if a spline for
        that gamma has already been created
    k [int]: degree of spline to use. 3 by default (cubic splines).
        Formal range: integers 1 <= k <= 5. Do NOT use 2 or 4.
    
    If SplineLog is set, interpolations are performed in log-space,
        i.e. the results is a spline interpolation of the log10 of the
        answer in terms of the log10 of the input
    """
    global SplineMin,SplineMax,NSpline,SplineLog
    for gamma in gammas:
        if gamma not in igamma_splines.keys() or reinit:
            print(f"Initializing igamma_spline for gamma={gamma}")
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
            


def test_spline_accuracy(gamma = -1.5837, Ntest = 100):
    """
    Function to test the accuracy of the spline interplation
    It explores different methods of spline interpolation
    
    Ntest [int]: number of random values to generate
    gamma [float]: value of the index gamma to test at
    
    Output: This produces three plots, being
        spline_test.pdf: compares splines to truth
        spline_errors.pdf: plots absolute value of errors
        spline_rel_errors.pdf: plots absolute value of relative errors
    """
    global SplineMin,SplineMax,NSpline,SplineLog
    
    # generate random values in the range of the splines
    lnewavals = np.random.rand(Ntest) * (SplineMax - SplineMin) + SplineMin
    lnewavals = np.sort(lnewavals)
    newavals = 10**lnewavals
    
    
    # generate the true values at these avalues via direct calculation
    truth = np.zeros(Ntest)
    for i,av in enumerate(newavals):
        truth[i] = mpmath.gammainc(gamma, a=av)
    
    # sets up for plotting results
    from matplotlib import pyplot as plt
    plt.figure()
    ax1=plt.gca()
    plt.plot(newavals,truth,label='truth')
    plt.xlabel('a')
    plt.ylabel('cumulative gamma function')
    plt.xscale('log')
    plt.yscale('log')
    
    # sets up for plotting differences between truth and interpolation
    plt.figure()
    ax2=plt.gca()
    plt.xlabel('a')
    plt.ylabel('|interpolation - truth|')
    plt.xscale('log')
    plt.yscale('log')
    
    # sets up for plotting relative differences between truth and interpolation
    plt.figure()
    ax3=plt.gca()
    plt.xlabel('a')
    plt.ylabel('|interpolation - truth|/truth')
    plt.xscale('log')
    plt.yscale('log')
    
    # now use different spline methods to evaluate
    # standard method: cubic, linear
    
    SplineLog=False
    init_igamma_splines([gamma],reinit=True,k=3)
    result = interpolate.splev(newavals, igamma_splines[gamma])
    ax1.plot(newavals,result,label='cubic spline, linear')
    diff = np.abs(result - truth)
    ax2.plot(newavals,diff,label='cubic spline, linear')
    rdiff = diff/truth
    ax3.plot(newavals,rdiff,label='cubic spline, linear')
    
    # now use different spline methods to evaluate
    # standard method: cubic, linear
    init_igamma_splines([gamma],reinit=True,k=1)
    result = interpolate.splev(newavals, igamma_splines[gamma])
    ax1.plot(newavals,result,label='linear spline, linear')
    diff = np.abs(result - truth)
    ax2.plot(newavals,diff,label='linear spline, linear')
    rdiff = diff/truth
    ax3.plot(newavals,rdiff,label='linear spline, linear')
    
    
    SplineLog=True
    init_igamma_splines([gamma],reinit=True,k=3)
    result = 10**interpolate.splev(lnewavals, igamma_splines[gamma])
    ax1.plot(newavals,result,label='cubic spline, log')
    diff = np.abs(result - truth)
    ax2.plot(newavals,diff,label='cubic spline, log')
    rdiff = diff/truth
    ax3.plot(newavals,rdiff,label='cubic spline, log')
    
    
    # now use different spline methods to evaluate
    # standard method: cubic, linear
    init_igamma_splines([gamma],reinit=True,k=1)
    result = 10**interpolate.splev(lnewavals, igamma_splines[gamma])
    ax1.plot(newavals,result,label='linear spline, log')
    diff = np.abs(result - truth)
    ax2.plot(newavals,diff,label='linear spline, log')
    rdiff = diff/truth
    ax3.plot(newavals,rdiff,label='linear spline, log')
    
    plt.sca(ax1)
    plt.legend()
    plt.tight_layout()
    plt.savefig('spline_test.pdf')
    plt.close()
    
    plt.sca(ax2)
    plt.legend()
    plt.tight_layout()
    plt.savefig('spline_errors.pdf')
    
    plt.sca(ax3)
    plt.legend()
    plt.ylim(1e-14,1)
    plt.tight_layout()
    plt.savefig('spline_rel_errors.pdf')
    plt.close()
    

def time_splines(gamma = -1.5837, Ntimetest = 100000, Nreps=100):
    """
    Function to time different methods of spline interpolation.
    It explores different methods of spline interpolation
    
    gamma [float]: value of the index gamma to test at
    Ntimetest [int]: number of random values to generate to test on
    Nreps [int]: number of repetitions for timing
    """
    global SplineMin,SplineMax,NSpline
    
    import time
    
    # begins with very large array of values to evaluate on
    lnewavals = np.random.rand(Ntimetest) * (SplineMax - SplineMin) + SplineMin
    newavals = 10**lnewavals
    
    # now use different spline methods to evaluate
    # standard method: cubic, linear
    
    SplineLog=False
    t0=time.time()
    for i in np.arange(Nreps):
        init_igamma_splines([gamma],reinit=True,k=3)
    t1=time.time()
    for i in np.arange(Nreps):
        result = interpolate.splev(newavals, igamma_splines[gamma])
    t2=time.time()
    dt1_ci = t1-t0
    dt2_ci = t2-t1
    
    # now use different spline methods to evaluate
    # standard method: cubic, linear
    t0=time.time()
    for i in np.arange(Nreps):
        init_igamma_splines([gamma],reinit=True,k=1)
    t1=time.time()
    for i in np.arange(Nreps):
        result = interpolate.splev(newavals, igamma_splines[gamma])
    t2=time.time()
    dt1_li = t1-t0
    dt2_li = t2-t1
    
    SplineLog=True
    t0=time.time()
    for i in np.arange(Nreps):
        init_igamma_splines([gamma],reinit=True,k=3)
    t1=time.time()
    for i in np.arange(Nreps):
        result = 10**interpolate.splev(lnewavals, igamma_splines[gamma])
    t2=time.time()
    dt1_co = t1-t0
    dt2_co = t2-t1
    
    # now use different spline methods to evaluate
    # standard method: cubic, linear
    t0=time.time()
    for i in np.arange(Nreps):
        init_igamma_splines([gamma],reinit=True,k=1)
    t1=time.time()
    for i in np.arange(Nreps):
        result = 10**interpolate.splev(lnewavals, igamma_splines[gamma])
    t2=time.time()
    dt1_lo = t1-t0
    dt2_lo = t2-t1
    
    print("Performing ",Nreps," spline initialisations took...")
    print("Cubic spline in linear space:  ",dt1_ci)
    print("Linear spline in linear space: ",dt1_li)
    print("Cubic spline in log space:     ",dt1_co)
    print("Linear spline in log space:    ",dt1_lo)
    
    print("Performing ",Nreps," x ",Ntimetest," spline evaluations took...")
    print("Cubic spline in linear space:  ",dt2_ci)
    print("Linear spline in linear space: ",dt2_li)
    print("Cubic spline in log space:     ",dt2_co)
    print("Linear spline in log space:    ",dt2_lo)
    
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
