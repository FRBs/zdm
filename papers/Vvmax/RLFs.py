"""
Radio luminosity functions used to fit histograms of (1/Vmax)
as a function of burst energy

Throughout, we use the notation:
    gamma: differential slope of the luminosity function
    Emax: maximum FRB energy (power-law) or characteristic
        turn-over energy (Schechter)
    A: Amplitude, being the normalisation/leading constant of the RLF
    
    Method: Key to indicate fitting method.
        0: a Schechter function, evaluated at bin centres
        1: Schechter function, integrated over the bin
        2: Schechter function fit in log-space, integrated over the bin
        3: power-law, evaluated at bin centre
                
"""

import numpy as np
from scipy.integrate import quad

global E0
# normalisation energy - no physical meaning, just makes units more sensible
E0=1e22


def make_first_guess(E,L, params,method):
    """
    Makes a first guess by setting the amplitude
    to that of the first bin
    
    Inputs:
        E [array, float]: array of energies at bin centres
        L [float]: histogram of luminosity function L(E) ~ p(E)dE
        params [array,float]: vector of power-law slope gamma, and Emax
    
    Outputs:
        guess for parameter fits: intercept, gamma, Emax
    
    """
    if method==0:
        mag = Schechter(E[0],1.,params[0],params[1])
        guess = [L[0]/mag,params[0],params[1]]
    elif method==1:
        mag = integrateSchechter(E[0],1.,params[0],params[1])
        guess = [L[0]/mag,params[0],params[1]]
    elif method==2:
        # returns the log10 magnitude expected with an amplitude of unity
        mag = logIntegrateSchechter(np.log10(E[0]),1.,params[0],params[1])
        guess = [L[0]/10.**mag,params[0],params[1]]   
    else:
        mag = power_law(E[0],1.,params[0],params[1])
        guess = [L[0]/mag,params[0],params[1]]
    
    
    return guess
    
def logIntegrateSchechter(log10E,A,gamma,log10Emax):
    """
    Returns the logarithms of the IntegrateSchechter function
    Requires log10 E to be sent
    """
    res = integrateSchechter(10.**log10E,A,gamma,log10Emax)
    res = np.log10(res)
    return res

def logIntegratePowerLaw(log10E,A,gamma):
    """
    Returns the logarithms of the IntegrateSchechter function
    Requires log10 E to be sent
    """
    res = integratePowerLaw(10.**log10E,A,gamma)
    res = np.log10(res)
    return res
    
def integrateSchechter(E,A,gamma,log10Emax):
    """
    Schechter function
    Gamma is differential
    Relative to E0, which is arbitrary
    
    An extra E/E0 due to log binning of histogram
    """
    Emax = 10**log10Emax
    bw = 10**0.5
    if isinstance(E,np.ndarray):
        y=np.zeros(E.size)
        for i,bc in enumerate(E):
            res = quad(Schechter,bc/bw,bc*bw,args=(A,gamma,Emax))
            y[i]=res[0]
    else:
        res = quad(Schechter,E/bw,E*bw,args=(A,gamma,Emax))
        y = res[0]
    return y

def integratePowerLaw(E,A,gamma):
    """
    Schechter function
    Gamma is differential
    Relative to E0, which is arbitrary
    
    An extra E/E0 due to log binning of histogram
    """
    bw = 10**0.5
    if isinstance(E,np.ndarray):
        y=np.zeros(E.size)
        for i,bc in enumerate(E):
            res = A * ((bc*bw)**(gamma+1) - (bc/bw)**(gamma+1))/(gamma+1)
            #res = quad(Schechter,bc/bw,bc*bw,args=(A,gamma,Emax))
            y[i]=res
    else:
        #res = quad(Schechter,E/bw,E*bw,args=(A,gamma,Emax))
        res = A * ((E*bw)**(gamma+1) - (E/bw)**(gamma+1))/(gamma+1)
        y = res
    return y

def Schechter(E,A,gamma,Emax):
    """
    Schechter function
    Gamma is differential
    Relative to E0, which is arbitrary
    
    An extra E/E0 due to log binning of histogram
    """
    global E0
    return A*(E/E0)**gamma * np.exp(-E/Emax)

def best_fits(JHz = True):
    """
    Returns best-fit parameters of the Schechter function according to
    different papers.
    
    Only gives gamma and Emax
    
    JHz means divide by 1e7 (erg) and 1e9 (GHz bandwidth)
    """
    JamesFit = [-1.95,41.26]
    RyderFit = [-1.95,41.7]
    ShinFit = [-1.3,41.38]
    LuoFit = [-1.79,41.46]
    for item in [JamesFit,RyderFit,ShinFit,LuoFit]:
        #item[1] = 10.**item[1]
        if JHz:
        # convert from erg to J Hz
            item[1] = item[1] - 16
    return JamesFit,RyderFit,ShinFit,LuoFit

