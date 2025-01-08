"""
Script to calculate V and Vmax for CRAFT FRBs for the "VVmax" paper (Arcus et al)

This script is modified so that the FRBs are *always* assumed to be at beam centre
just like a tophat function used in original FRB calculation.

Modification is "Inserted systematic test here"

Writes output to "BeamCentreUnbiasedLocalisedOutput"
(this is to compare to the unbiased localised output)


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


def main(fname,outfile,Nsfr,alpha=0.,loc=False,minz=None,write=True,
    laststrings=None,LimZ=None,zfrac = 0.):
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
    global macquartz
    global macquartDM
    global state
    
    # initialises cosmology and distance measures 
    #state = init_cos()
    
    # loads unlocalised data, gets Macquart z
    
    
    names,DM,freqMHz,SNR,SNRth,DMG,tsamp,width,BeamThetaDeg,Fnu,\
        zDM,Nants,Enu = curate_data(loc,fname,minz,LimZ)
    
    
    # gets DM cosmic contributions for data
    DMcosmic = vvmax.get_DM_cosmic(zDM,state)
    
    # loads beamshape data
    bvals,omegab = load_beamshape()
    Bvals = calc_Bvals(BeamThetaDeg,freqMHz)
    
    # calculates value of beam at point of detection
    # this is my own calculation, but incorrect
    # Ryan's has been updated, and is much better
    if False:
        plot_beamshape(BeamThetaDeg,Bvals)
        
        # gets the fluences
        F0 = 22 # Jyms
        Fluences = calc_fluence(F0,Bvals,SNR,width,tsamp,DM,freqMHz,Nants)
        plot_fluence(DM,Fluences,Fnu)
        Fnu = Fluences
    
    if laststrings is None:
        tempstrings = []
    
    # speedup!!! (or is it???)
    #zarray = np.linspace(0,9.8,9801)
    #varray = np.zeros([9801])
    print("Initialising dVdOmega array...")
    N=9801
    zarray = np.linspace(0,9.8,N)
    varray = np.zeros([N])
    for i,z in enumerate(zarray):
        dVdOmega = integrate_volume(z,Nsfr=Nsfr) #weighted cubic Mpc per steradian
        varray[i] = dVdOmega
    Vsplines = interpolate.CubicSpline(zarray,varray)
    print("Done")
    # now set
    if write:
        f = open(outfile,"a")
    print("FRB  Fnu  zmaxB  zmaxC  BLd  CLd")
    
    # initialises calculation times - for testing
    #dt1=0
    #dt2=0
    
    # tries getting zmax for one FRB
    for i,FRB in enumerate(names):
        #if not (FRB == 20180110):
        #    continue
        print("Starting frb ",FRB)
        # calculates zmax at the B of detection
        s = SNR[i]/SNRth[i]
        Bzmax = vvmax.calc_zmax(s,zDM[i],DM[i],DMcosmic[i],width[i],freqMHz[i],
            tsamp[i],macquartz,macquartDM,alpha)
        
        s = SNR[i]/SNRth[i] / Bvals[i]
        Czmax = vvmax.calc_zmax(s,zDM[i],DM[i],DMcosmic[i],width[i],freqMHz[i],
            tsamp[i],macquartz,macquartDM,alpha)
        
        # note: last one is beam centre. This is a speedup, mostly for testing
        if write:
            LoopB = bvals
            LoopO = omegab
            
        else:
            LoopB = bvals[-3:]
            LoopO = omegab[-3:]
        
        s=SNR[i]/SNRth[i]
        
        ########## Inserted systematic test here ##########
        # we falsify the B-values for each FRB, and *also* curate the B,O values
        # to loop over
        LoopB = np.array([0.])
        HWHP = 0.9*np.pi/180. # Approx half width half power
        Area = np.pi * HWHP**2 * 36 # area of 36 beams (approx!!!)
        LoopO = np.array([Area]) # approx area at half-width half-power
        Bval = 1.
        
        Volume,Vmax=vvmax.get_vvmax(LoopB,LoopO,Bval,s,zDM[i],DM[i],DMcosmic[i],
            width[i],freqMHz[i],tsamp[i],macquartz,macquartDM,alpha,Vsplines,LimZ)
        
        # in Gpc
        Bdl = cos.DL(Bzmax)/1e3
        Cdl = cos.DL(Czmax)/1e3
        
        energyErg = cos.F_to_E(Fnu[i],zDM[i],alpha,bandwidth=1.) # bandwidth of unity
        JHz = energyErg / 1e7
        
        
        ### constructs string to write to data file ####
        string = '{0:8} {1:1.3e} {2:6.5f} {3:6.5f} {4:1.3e} {5:1.2e} {6:6.3f}\n'.format(
            int(FRB),JHz,Bzmax,Czmax,Volume,Vmax,Volume/Vmax)
        # printing out Czmax and Cdl would be super useful for the paper! Wait, why Dl? Who cares!
        
        if write:
            f.write(string),SNR[i],DM[i],DMG[i],freqMHz[i],tsamp[i],width[i],BeamThetaDeg[i]
        
        # just a proxy
        tempzmax = -1
        
        #### constructs string to write to table in paper ####
        if minz is not None:
            string = '\\FRB{0:8} & {1:1.5f} & {2:1.2e} & {3:6.5f} & {4:1.2e} & {5:1.2e} & {6:6.3f} &'.format \
            (names[i],cos.DL(zDM[i])/1e3,Enu[i],tempzmax,Volume,Vmax,Volume/Vmax)
        else:
            string = '\\FRB{0:8} & {1:1.3f} & {2:1.2e} & {3:6.3f} & {4:1.2e} & {5:1.2e} & {6:6.3f} &'.format \
                (names[i],cos.DL(zDM[i])/1e3,Enu[i],tempzmax,Volume,Vmax,Volume/Vmax)
        if laststrings is None:
            tempstrings.append(string)
            print(string)
        else:
            extra = ' {0:1.2e} & {1:6.3f} & {2:1.2e} & {3:1.2e} & {4:6.3f} \\'.format \
            (Enu[i],tempzmax,Volume,Vmax,Volume/Vmax)
            combined = laststrings[i]+extra
            print(combined)
    
    # for estimating calculation time
    #print("Calculation times are ",dt1,dt2)  
    
    if write:
        f.close()
    if laststrings is None:
        return tempstrings
    else:
        return laststrings   


def curate_data(loc,fname,minz,LimZ):
    """
    Edits the data used for the V/Vmax calculation according to various criteria
    
    Returns edited FRB properties
    """
    
    if loc:
        names,DM,freqMHz,SNR,SNRth,DMG,tsamp,width,BeamThetaDeg,Fnu,zDM,Nants \
             = load_loc_frb_data(fname)
        # places a cut on S/N
        # selects only those with SNR greater than the threshold
        SNRth[:] = 14. # artificial threshold
        OK = np.where(SNR >= 14.)
        names = names[OK]
        DM = DM[OK]
        freqMHz = freqMHz[OK]
        SNR = SNR[OK]
        SNRth = SNRth[OK]
        DMG = DMG[OK]
        tsamp = tsamp[OK]
        width = width[OK]
        BeamThetaDeg = BeamThetaDeg[OK]
        Fnu = Fnu[OK]
        zDM = zDM[OK]
        Nants= Nants[OK]
    else:
        names,DM,freqMHz,SNR,SNRth,DMG,tsamp,width,BeamThetaDeg,Fnu,zDM,Nants \
             = load_unloc_frb_data(fname)
    DMhalo = 50. # halo and host
    DMeg = DM-DMG - DMhalo
    DMhost=50
    
    if not loc:
        # estimates z based on Macquart relation
        old_zDM = zDM
        zDM = vvmax.get_macquart_z(DMeg,state,DMhost)
        plot_z_error(zDM,old_zDM)
    
    Enu = cos.F_to_E(Fnu,zDM,alpha)
    Enu /= 1e16 # 1e9 for bandwidth, 1e7 for erg to J
    
    # removes or modifies zDM < 0 FRBs
    if minz is not None:
        # replaces negative z FRBs with a minimum value of z
        low = np.where(zDM < 0.)
        
        # keeps only these FRBs - everything else should be identical
        names = names[low]
        DM = DM[low]
        freqMHz = freqMHz[low]
        SNR = SNR[low]
        SNRth = SNRth[low]
        DMG = DMG[low]
        tsamp = tsamp[low]
        width = width[low]
        BeamThetaDeg = BeamThetaDeg[low]
        Fnu = Fnu[low]
        zDM = zDM[low]
        Nants= Nants[low]
        Enu = Enu[low]
        
        orig_zDM = zDM
        max_zDM = vvmax.get_macquart_z(DM,state,0.) #calculates max z assuming *all* DM is extragalactic
        zDM[:] = minz + zfrac*(max_zDM-minz) # scales linearly from
        print("zDMs calculated as ",zDM," from ",minz," to ",max_zDM)
    else:
        # cuts away anything with z too low
        OK = np.where(zDM > 0.)
        names = names[OK]
        DM = DM[OK]
        freqMHz = freqMHz[OK]
        SNR = SNR[OK]
        SNRth = SNRth[OK]
        DMG = DMG[OK]
        tsamp = tsamp[OK]
        width = width[OK]
        BeamThetaDeg = BeamThetaDeg[OK]
        Fnu = Fnu[OK]
        zDM = zDM[OK]
        Nants= Nants[OK]
        Enu = Enu[OK]
    
    if LimZ is not None:
        # cuts away anything with z too high
        OK = np.where(zDM < LimZ)
        names = names[OK]
        DM = DM[OK]
        freqMHz = freqMHz[OK]
        SNR = SNR[OK]
        SNRth = SNRth[OK]
        DMG = DMG[OK]
        tsamp = tsamp[OK]
        width = width[OK]
        BeamThetaDeg = BeamThetaDeg[OK]
        Fnu = Fnu[OK]
        zDM = zDM[OK]
        Nants= Nants[OK]
        Enu = Enu[OK]
    
    return names,DM,freqMHz,SNR,SNRth,DMG,tsamp,width,BeamThetaDeg,Fnu,\
        zDM,Nants,Enu
         
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

def init_cos():
    """
    
    """
    state = loading.set_state()
    state.cosmo.H0 = 70.0
    state.cosmo.Omega_b = 0.0486
    cos.set_cosmology(state)
    cos.init_dist_measures()
    return state
    



def plot_z_error(myz,waynez):    
    """
    Plots error in z from excluding f_d
    """
    
    plt.figure()
    plt.scatter(waynez,myz)
    plt.xlabel("Wayne")
    plt.ylabel("Me")
    plt.xlim(0.,1.5)
    plt.ylim(0.,1.5)
    plt.savefig("redshift_error.pdf")
    plt.close()
    
    
    
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



global macquartz
global state
global macquartDM

# sets up a constant array of Macquart relation zvalues
state = init_cos()
macquartz = np.linspace(1e-5,9.8,980000) # 1e-5,980000 is better
#macquartz = np.linspace(1e-4,9.8,98000) # 1e-5,980000 is better
macquartDM=pcosmic.get_mean_DM(macquartz, state)
    

########## run testing effect of zDM < 0 FRBs  ###########

# limits maximum redshift to 0.7. This is the unbiased calculation for localised FRBs
# only performs this run for beam-centre calculations
# (it's an example!)
if True:
    loc=True
    LimZ=0.7
    infile="Data/unbiased_loc_data.dat"
    Nsfr=0.
    alpha=0.
    outfile = "BeamCentreUnbiasedLocalisedOutput/localised_vvmax_data_NSFR_"+str(Nsfr)[0:3]+"_alpha_"+str(alpha)+".dat"
    main(infile,outfile,Nsfr,alpha=alpha,loc=loc,minz=None,laststrings=None,LimZ=LimZ)

