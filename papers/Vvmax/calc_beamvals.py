"""
Script to calculate V and Vmax for CRAFT FRBs for the "VVmax" paper (Arcus et al)
"""

# imports from zdm
from zdm import pcosmic
from zdm import loading as loading
from zdm import vvmax

# standard python imports
from scipy import interpolate
import numpy as np
import os
from matplotlib import pyplot as plt
from scipy.special import jv as BessJ
from scipy import constants
from pkg_resources import resource_filename
from zdm import cosmology as cos
from scipy.integrate import quad



# initialises cosmology and distance measures 


def main():
    """
    
    Main program to calculate V and Vmax values for the FRBs in fname
    
    INPUTS:
        fname [string]: data file containing the FRBs of interest
        outfile [string]: output file base for writing output
        Nsfr [float]: scaling with star-formation rate
        alpha [float]: frequency scaling of FRBs: F~nu^alpha
        loc [bool]: True if using localised sample
        False if z should be estimated through Macquart relation
        minz [None or float]: if float, minimum redshift for FRBs
            with low DM
        write [bool]: if True, write data to outfile
        laststrings [None or list of strings]: list of strings
            to append for writing output
        LimZ [None or float]: maximum redshift for V/Vmax calculations
        zfrac [float]: if using minz, fractional distance to assume
            between minz and maximum redshift for FRBs with otherwise
            negative Macquart redshifts
        
    RETURNS:
        laststrings: returns strings which otherwise would have been written
            if laststrings is None, or otherwise returns input value
    """
    # initialises cosmology and distance measures 
    #state = init_cos()
    
    # loads unlocalised data, gets Macquart z
    names,DM,freqMHz,SNR,SNRth,DMG,tsamp,width,BeamThetaDeg,newFnu,zloc,Nants = load_unloc_frb_data("Data/all_unloc_frb_data.dat")
    
    # loads beamshape data
    bvals,omegab = load_beamshape()
    Bvals = calc_Bvals(BeamThetaDeg,freqMHz)
    
    for i,name in enumerate(names):
        print(name,Bvals[i])
    
    

def load_data(fname):
    """
    Edits the data used for the V/Vmax calculation according to various criteria
    
    Returns edited FRB properties
    """
    
    names,DM,freqMHz,SNR,SNRth,DMG,tsamp,width,BeamThetaDeg,Fnu,zDM,Nants \
             = load_loc_frb_data(fname)
    
    return names,freqMHz,BeamThetaDeg,Fnu
         
def integrate_volume(zmax,Nsfr=0.):
    """
    Integrates dV/dz from 0 to z
    Weights according to SFR**n
    """
    result = quad(dVdtaudz,0,zmax,args=[Nsfr])
    return result[0]
    
def dVdtaudz(z,Nsfr):
    """
    Volume element per dz per solid angle
    """
    weighted_dVdtau = cos.dVdtau(z) * cos.sfr_evolution(z,Nsfr[0])
    return weighted_dVdtau

def plot_fluence(DM,Fluences,Fnu):
    """
    Creates a plot of the "Fluences" calculated by Clancy 
    vs the "Fnu" calculated by Wayne
    """
    plt.figure()
    plt.scatter(DM, Fluences/Fnu)
    plt.xlabel("DM")
    plt.ylabel("James F / Arcus F")
    plt.ylim(0,2)
    #plt.xscale("log")
    #plt.yscale("log")
    plt.savefig("frb_fluences.pdf")
    plt.close()

def calc_fluence(F0,Bvals,SNR,w,tsamp,DM,freq,Nants):
    """
    Calculates fluence of the burst
    Uses dodgy 4.15 ms dispersion measure calc
    """
    SNRthresh = 9.5
    
    dmsmear = DM * 4.15 * 1e6 * ((freq-0.5)**-2 - (freq+0.5)**-2)
    
    #ArcusEfficiency
    #eta0=1.
    #efficiency = eta0 * (1 + (tsamp/w)**2 + (dmsmear/w)**2)**-0.5
    
    # only if we forget that the measured width is actually inclusive
    # of these factors
    #weff = (w**2 + tsamp**2 + dmsmear**2)**0.5
    #efficiency = weff**-0.5
    
    # realises that we already include tsamp and dmsmear in w
    # but wmin *must* be at least equal to dm smearing and tsamp
    wmin = (dmsmear**2 + tsamp**2)**0.5
    efficiency = np.zeros([wmin.size])
    toolow = np.where(w<wmin)[0]
    OK = np.where(w>=wmin)[0]
    efficiency[toolow] = wmin[toolow]**-0.5
    efficiency[OK] = w[OK]**-0.5
    #if w < wmin:
    #    efficiency = wmin**-0.5
    #else:
    #    efficiency = w**-0.5
    
    #efficiency=1.
    
    #efficiency = 1.
    F = F0 * (SNR/SNRthresh) /Nants**0.5 /Bvals / efficiency
    return F

def calc_Bvals(thetasDeg,freqMHz):
    """
    Calculates the FRB fluence assuming a certain threshold, SNR, and
    beam centre distance
    """
    # coefficient of Airy function is lambda/D
    wavelength = constants.speed_of_light/1e6/freqMHz
    radius = 6 # ASKAP diameter in m
    Diam = radius * 2.
    thetasRad = thetasDeg * np.pi/180.
    stheta = np.sin(thetasRad)
    k = 2. * np.pi/wavelength
    # uses an airy function we should use a Gaussian like everybody else...
    sigma = 1.1 * constants.speed_of_light/Diam/freqMHz/1e6/(2 * (2*np.log(2))**0.5)
    
    Bvals = np.exp(-0.5*(thetasRad/sigma)**2)
    #inputs = stheta * k * radius
    #Bvals = airy(inputs)
    return Bvals


def plot_beamshape(theta,B):
    """
    plots beamshape
    """
    plt.figure()
    plt.scatter(theta,B)
    plt.xlabel("Offset (deg)")
    plt.ylabel("B")
    plt.tight_layout()
    plt.savefig("frb_b_values.pdf")
    plt.close()

def airy(x):
    """
    airy disk function
    """
    
    val = (2. * BessJ(1,x)/x)**2
    return val
  
def load_beamshape():
    """
    loads askap beamshape info
    """
    import os
    sdir = os.path.join(resource_filename('zdm', 'data'), 'BeamData/')
    bvals = np.load(sdir+"lat50_log_bins.npy")
    omegab = np.load(sdir + "lat50_log_hist.npy")
    return bvals,omegab

def load_loc_frb_data(fname):
    """
    Loads table of localised FRBs
    """
    data = np.loadtxt(fname)
    names = data[:,0]
    DM = data[:,2]
    freqMHz = data[:,4]
    SNR = data[:,1]
    #SNRth = data[:,4]
    DMG = data[:,3]
    tsamp = data[:,5]
    width = data[:,6]
    BeamThetaDeg = data[:,7]
    oldFnu = data[:,8]
    zloc = data[:,9]
    Nants = data[:,10]
    newFnu = data[:,11]
    
    SNRth = np.full([newFnu.size],9.5)
    
    return names,DM,freqMHz,SNR,SNRth,DMG,tsamp,width,BeamThetaDeg,newFnu,zloc,Nants
    

def load_unloc_frb_data(filename):
    """
    Loads table of unlocalised FRBs
    
    Uses new fluence data from Ryan
    """
    data = np.loadtxt(filename)
    names = data[:,0]
    DM = data[:,1]
    freqMHz = data[:,2]
    SNR = data[:,3]
    SNRth = data[:,4]
    DMG = data[:,5]
    tsamp = data[:,6]
    width = data[:,7]
    BeamThetaDeg = data[:,8]
    oldFnu = data[:,9]
    zDM = data[:,10]
    Nants = data[:,11]
    newFnu = data[:,12]
    return names,DM,freqMHz,SNR,SNRth,DMG,tsamp,width,BeamThetaDeg,newFnu,zDM,Nants

main()
