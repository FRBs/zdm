"""
This file contains routines for analysing MCMC results

Routines here were written by Jordan Hoffmann

"""


import numpy as np
from matplotlib import pyplot as plt

# Here are different plotting functions

def plot_walkers(samples,labels,outfile,burnin=None):
    """
    Puts all walkers from all samples on one plot
    If you want different samples per plot, call this function
    multiple times
    """
    plt.rcParams['font.size'] = 16
    # get the number of parameters
    fig, axes = plt.subplots(len(labels), 1, figsize=(20,30), sharex=True)
    #plt.title("Sample: " + filenames[j])
    for j,sample in enumerate(samples):
        for i,ax in enumerate(axes):
            for k in range(sample.shape[1]):
                if burnin is None:
                    ax.plot(sample[:,k,i], '.-', label=str(k))
                else:
                    ax.plot(sample[burnin[j]:,k,i], '.-', label=str(k))
        
            ax.set_ylabel(labels[i])
    
        axes[-1].set_xlabel("Step number")
        axes[-1].legend()
    
    plt.tight_layout()
    plt.savefig(outfile)
    plt.close()  

def plot_autocorrelations(samples,opfile):
    """
    Plot the autocorrelation time to estimate the burnin
    Do this once bad walkers have been discarded.
    To be done: proper explanation of what this routine is doing
    """
    burnin = []
    fig, ax = plt.subplots(1,1,figsize=(8,6))
    for isample,sample in enumerate(samples):
        # Compute the estimators for a few different chain lengths
        N = np.exp(np.linspace(np.log(10), np.log(sample.shape[0]), 10)).astype(int)
        new = np.empty(len(N))
        for i, n in enumerate(N):
            new[i] = autocorr(sample[:, :n, 0].T)
    
        # Plot the comparisons
        
        ax.loglog(N, new, "o-", label="new")
        ylim = ax.get_ylim()
        ax.plot(N, N / 50.0, "--k", label=r"$\tau = N/50$")
        ax.set_ylim(ylim)
        ax.legend(fontsize=14);

        burnin.append(int(1.5*new[-1]))
    
    ax.set_xlabel("number of samples, $N$")
    ax.set_ylabel(r"$\tau$ estimates")
    plt.tight_layout()
    plt.savefig(opfile)
    plt.close()


# # Implement burnin and change priors
# 
# - Changes prior to discard samples outside the specified prior range
# - Implements the burnin using either the predefined burnin or a constant specified

# Enforce more restrictive priors on a parameter
# get rid of burnin first!
def change_priors(sample, param_num, max=np.inf, min=-np.inf):

    condition = np.logical_and(sample[:,param_num] > min, sample[:,param_num] < max)
    good_idxs = np.flatnonzero(condition)

    return sample[good_idxs, :]

# Here we present different methods to get the burnin from
# https://emcee.readthedocs.io/en/stable/tutorials/autocorr/#a-more-realistic-example
# however we note that in actuality it is generally easier and more useful to specify
# burnin=200 or something similar which is done further below.

def next_pow_two(n):
    i = 1
    while i < n:
        i = i << 1
    return i

def autocorr_func_1d(x, norm=True):
    x = np.atleast_1d(x)
    if len(x.shape) != 1:
        raise ValueError("invalid dimensions for 1D autocorrelation function")
    n = next_pow_two(len(x))

    # Compute the FFT and then (from that) the auto-correlation function
    f = np.fft.fft(x - np.mean(x), n=2 * n)
    acf = np.fft.ifft(f * np.conjugate(f))[: len(x)].real
    acf /= 4 * n

    # Optionally normalize
    if norm and acf[0] != 0:
        acf /= acf[0]

    return acf

# Automated windowing procedure following Sokal (1989)
def auto_window(taus, c):
    m = np.arange(len(taus)) < c * taus
    if np.any(m):
        return np.argmin(m)
    return len(taus) - 1

def autocorr(y, c=5.0):
    f = np.zeros(y.shape[1])
    for yy in y:
        f += autocorr_func_1d(yy)
    f /= len(y)
    taus = 2.0 * np.cumsum(f) - 1.0
    window = auto_window(taus, c)
    return taus[window]



# - Discards any walkers that do not converge
# Reject walkers with bad autocorrelation values
def auto_corr_rej(samples, burnin=0):
    good_samples = []

    # Loop through each sample and generate a list of good walkers and bad walkers
    for j,sample in enumerate(samples): 
        # burnin=200
        good_walkers = []
        bad_walkers = []


        # for i in range(sample.shape[1]):
        #     # if np.all(sample[burnin:burnin+30,i,0] == sample[burnin,i,0]):
        #     if ( np.std(sample[burnin:burnin+30,i,0] ) )
        #         bad_walkers.append(i)
        #     else:
        #         good_walkers.append(i)

        # Loop through each walker in the current sample
        for i in range(sample.shape[1]):
            bad = False

            # Loop through each parameter for the walker
            for k in range(sample.shape[2]):

                # If any of the parameters have a bad autocorrelation function then set as a bad walker
                acf = autocorr_func_1d(sample[burnin:,i,k], norm=False)
                if np.max(acf) < 1e-10:
                    bad = True
                    break

            if bad:
                bad_walkers.append(i)
            else:
                good_walkers.append(i)


        print("Discarded walkers for sample " + str(j) + ": " + str(bad_walkers))

        # Add the new sample with the bad walkers discarded to the good_samples list
        good_samples.append(sample[burnin:,good_walkers,:])

    return good_samples

# Reject walkers with small standard deviations
def std_rej(samples, burnin=0):
    good_samples = []

    if not type(burnin) == list:
        burnin = [burnin for i in range(len(samples))]

    # Loop through each sample
    for i, sample in enumerate(samples):
        bad_walkers = []
        good_walkers = []

        # For each parameter
        for k in range(sample.shape[2]):
            sd = []

            # Loop through every walker and get a list of the standard deviations
            for j in range(sample.shape[1]):
                sd.append(np.std(sample[burnin[i]:burnin[i]+100,j,k]))

            # Normalise standard deviation
            sd = sd / np.max(sd)

            # Flag any walkers with standard deviations less than 1e-2
            bad_walkers = [] # np.flatnonzero(sd < 1e-2)
            temp = []
            for m in range(len(sd)):
                if sd[m] < 1e-2:
                    bad_walkers.append(m)

        bad_walkers = np.unique(np.array(bad_walkers))
        
        print("Discarded walkers for sample " + str(i) + ": " + str(bad_walkers))
        for l in range(sample.shape[1]):
            if l not in bad_walkers:
                good_walkers.append(l)

        # Add the new sample with the bad walkers discarded to the good_samples list
        good_samples.append(sample[:,good_walkers,:])

    return good_samples
