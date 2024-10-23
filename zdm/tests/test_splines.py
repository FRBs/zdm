""" 
This script produces plots to test spline accuracy,
and test the time taken to create and evaluate splines.

"""

from zdm import energetics
import mpmath
import numpy as np
from scipy import interpolate

def main():
    
    
    # this compares the accuracy of the default spline method, and
    # complains if anything if wrong by more than 0.01 %
    max_rdiff,lowest = test_default_spline()
    assert(max_rdiff < 1e-4)
    assert(lowest >= 0.)
    
    # this creates a bunch of plots to test the accuracy of different spline methods
    # off by default
    #test_spline_accuracy()
    
    # this times how long it takes to produce splines
    # off by default
    #time_splines(Nreps=1)

def test_default_spline(gamma=-1.5837, amin=1e-6, amax=1e2,
        Ntest=1000, plot=False):
    """
    Function to test the accuracy of the spline interplation
    It checks the accuracy vs direct calculation
    It does *not* check the accuracy of the underlying Python routine
    
    Input:
        gamma [float]: value of the index gamma to test at
        amin [float]: minimum value relative to Emax at which to calculate
        amax [float]: maximum value relative to Emax at which to calculate
        Ntest [int]: number of random values to generate
        plot [bool]: turn on to show a quick plot of results
    
    Returns: 
        phys_worst: worst relative error in the "physical" range where
            the true result is > 1e-200
        min: absolute minimum value, to check this never goes negative
    """
    
    # completely arbitrary choices of Emin and Emax
    Emin = 1e30
    Emax = 1e42
    
    # the range over which to generate test values of Eth
    lEthmin = np.log10(Emax) + np.log10(amin)
    lEthmax = np.log10(Emax) + np.log10(amax)
    
    
    # generate random values in the range of the splines
    lEth = np.random.rand(Ntest) * \
        (lEthmax - lEthmin) + lEthmin
    Eth = 10**lEth
    Eth = np.sort(Eth)
    
    # generate a default spline
    energetics.init_igamma_splines([gamma])
    # evaluate that spline
    result = energetics.vector_cum_gamma_spline(Eth,Emin,Emax,gamma)
    # evaluate the true values
    truth = energetics.vector_cum_gamma(Eth,Emin,Emax,gamma)
    
    rdiff = np.abs(result - truth)/truth
    
    # gets the greatest error in the "physical" range (> 1e-200)
    phys = np.where(truth > 1e-200)[0]
    phys_rdiff = rdiff[phys]
    
    phys_worst = np.max(phys_rdiff)
    
    # gets the minimum value over the entire range
    lowest = np.min(result)
    
    if plot:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(Eth,truth,linestyle='',marker='x')
        
        plt.plot(Eth,result,linestyle='',marker='o')
        plt.xscale('log')
        plt.yscale('log')
        plt.show()
    
    return phys_worst, lowest


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
    
    # generate random values in the range of the splines
    lnewavals = np.random.rand(Ntest) * \
        (energetics.SplineMax - energetics.SplineMin) + energetics.SplineMin
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
    
    energetics.SplineLog=False
    energetics.init_igamma_splines([gamma],reinit=True,k=3)
    result = interpolate.splev(newavals, energetics.igamma_splines[gamma])
    ax1.plot(newavals,result,label='cubic spline, linear')
    diff = np.abs(result - truth)
    ax2.plot(newavals,diff,label='cubic spline, linear')
    rdiff = diff/truth
    ax3.plot(newavals,rdiff,label='cubic spline, linear')
    
    # now use different spline methods to evaluate
    # standard method: cubic, linear
    energetics.init_igamma_splines([gamma],reinit=True,k=1)
    result = interpolate.splev(newavals, energetics.igamma_splines[gamma])
    ax1.plot(newavals,result,label='linear spline, linear')
    diff = np.abs(result - truth)
    ax2.plot(newavals,diff,label='linear spline, linear')
    rdiff = diff/truth
    ax3.plot(newavals,rdiff,label='linear spline, linear')
    
    
    energetics.SplineLog=True
    energetics.init_igamma_splines([gamma],reinit=True,k=3)
    result = 10**interpolate.splev(lnewavals, energetics.igamma_splines[gamma])
    ax1.plot(newavals,result,label='cubic spline, log')
    diff = np.abs(result - truth)
    ax2.plot(newavals,diff,label='cubic spline, log')
    rdiff = diff/truth
    ax3.plot(newavals,rdiff,label='cubic spline, log')
    
    
    # now use different spline methods to evaluate
    # standard method: cubic, linear
    energetics.init_igamma_splines([gamma],reinit=True,k=1)
    result = 10**interpolate.splev(lnewavals, energetics.igamma_splines[gamma])
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
    import time
    
    # begins with very large array of values to evaluate on
    lnewavals = np.random.rand(Ntimetest) * \
        (energetics.SplineMax - energetics.SplineMin) + energetics.SplineMin
    newavals = 10**lnewavals
    
    # now use different spline methods to evaluate
    # standard method: cubic, linear
    
    energetics.SplineLog=False
    t0=time.time()
    for i in np.arange(Nreps):
        energetics.init_igamma_splines([gamma],reinit=True,k=3)
    t1=time.time()
    for i in np.arange(Nreps):
        result = interpolate.splev(newavals, energetics.igamma_splines[gamma])
    t2=time.time()
    dt1_ci = t1-t0
    dt2_ci = t2-t1
    
    # now use different spline methods to evaluate
    # standard method: cubic, linear
    t0=time.time()
    for i in np.arange(Nreps):
        energetics.init_igamma_splines([gamma],reinit=True,k=1)
    t1=time.time()
    for i in np.arange(Nreps):
        result = interpolate.splev(newavals, energetics.igamma_splines[gamma])
    t2=time.time()
    dt1_li = t1-t0
    dt2_li = t2-t1
    
    energetics.SplineLog=True
    t0=time.time()
    for i in np.arange(Nreps):
        energetics.init_igamma_splines([gamma],reinit=True,k=3)
    t1=time.time()
    for i in np.arange(Nreps):
        result = 10**interpolate.splev(lnewavals, energetics.igamma_splines[gamma])
    t2=time.time()
    dt1_co = t1-t0
    dt2_co = t2-t1
    
    # now use different spline methods to evaluate
    # standard method: cubic, linear
    t0=time.time()
    for i in np.arange(Nreps):
        energetics.init_igamma_splines([gamma],reinit=True,k=1)
    t1=time.time()
    for i in np.arange(Nreps):
        result = 10**interpolate.splev(lnewavals, energetics.igamma_splines[gamma])
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
  

main()
