"""

This file contains routines generally associated with V/Vmax
calculations.

They were originally developed for the "VVmax" paper
(Arcus et al).

Specific features unique to that paper can be found
in the /papers/Vvmax subdirectory.

"""

from zdm import cosmology as cos
from zdm import pcosmic
import numpy as np
from scipy import interpolate
import time

def get_vvmax(BeamBs,BeamOs,Bfrb,sfrb,z0,DM0,DMcosmic0,w0,freqMHz,\
                tsamp,macquartz,macquartDM,alpha,Vsplines,\
                LimZ):
    """
    Calculates the values of V and Vmax [Mpc^3 dtau] for the given inputs
    
    
    INPUTS:
        BeamBs [np.ndarray (floats)]: list of relative beam values B
        BeamOs [np.ndarray (floats)]: list of solid angles for B
        Bfrb [float]: value of B at which the FRB is detected
        sfrb [float]: value of s=SNR/SNRth at which the FRB is detected
        z0 [float]: measured FRB redshift
        DM0 [float]: measured FRB *total* dispersion measures [pc cm^-3]
        DMcosmic0: estimated FRB *cosmic* dispersion measures [pc cm^-3]
        w0 [float]: measured FRB width [ms]
        freqMHz [float]: frequency of detection [MHz]
        tsamp [float]: integration time of data [ms]
        Macquartz [float]: redshift estimated from the Macquart relation
        MacquartDM [float]: cosmic dispersion measure estimated from the Macquart relation
        alpha [float]: frequency dependence of FRBs - Fnu ~ nu^\alpha
        Vsplines [cubic spline interpolation]: precalcuated cubic spline interpolation
            of volume as a function of redshift (for speed!)
        LimZ [None or float]: maximum redshift for V/Vmax calculations
    """
    # initialises volumes
    Volume = 0.
    Vmax = 0.
    
    for j,logB in enumerate(BeamBs): #,OmegaBs):
            
            B = 10.**logB
            s = sfrb * B/Bfrb
            t0=time.time()
            zmax = calc_zmax(s,z0,DM0,DMcosmic0,w0,freqMHz,
                tsamp,macquartz,macquartDM,alpha)
            
            # limits to maximum volume
            if LimZ is not None and zmax > LimZ:
                zmax = LimZ
            
            t1=time.time()
            #dVdOmega = integrate_volume(zmax,Nsfr=Nsfr) #weighted cubic Mpc per steradian
            
            dVdOmega = Vsplines(zmax)
            
            # calculates times for testing purposes
            t2=time.time()
            dt1 = (t1-t0)
            dt2 = (t2-t1)
            
            dVmax = dVdOmega * BeamOs[j]
            # dVmax is now the volume elements weighted over that solid angle
            
            Vmax += dVmax
            # we keep this tempzmax to be printed later
            tempzmax = zmax
            # calculates the volume by limiting zmax to actual FRB redshift 
            if zmax > z0:
                zmax = z0
            dVdOmega = Vsplines(zmax)
            dV = dVdOmega * BeamOs[j]
            Volume += dV
            
    return Volume,Vmax


def calc_zmax(s0,z0,DM0,DMcosmic0,w0,freq,tsamp,Macquartz,MacquartDM,alpha):
    """
    Routine which calculates the maximum redshift at which
    an FRB could have been detected.
    
    We begin with the ratio s0, which is SNR(det)/SNR(thresh)
    This tells us how much fluence we can lose as a function
    of redshift
    
    We then account for luminosity distance, and changing efficiency
    with distance
    
    '0' properties indicate those at detection
    
    Macquartz and MacquartDM are pre-computed values of Macquart relation
    
    We must have values of z which span the range from the minimum z
    at which the FRB lies, to the maximum z at which it could be detected
    E.g. at S/N of 1000, that's a factor of ~10 for SNRthresh=10
    
    INPUTS:
        s0 [float]: measured SNR/SNRthresh of detected FRB
        z0 [float]: measured FRB redshift
        DM0 [float]: measured FRB *total* dispersion measures [pc cm^-3]
        DMcosmic0: estimated FRB *cosmic* dispersion measures [pc cm^-3]
        w0 [float]: measured FRB width [ms]
        freq [float]: frequency of detection [MHz]
        tsamp [float]: integration time of data [ms]
        Macquartz [float]: redshift estimated from the Macquart relation
        MacquartDM [float]: cosmic dispersion measure estimated from the Macquart relation
        alpha [float]: frequency dependence of FRBs - Fnu ~ nu^\alpha
    
    RETURNS:
        maximum redshift that the FRB could have been detected at
    
    """
    
    # we begin by making a naive guess at zmax based upon
    # Cartesian geometry, and then we interpolate exact
    # values about this to get a precise answer
    zguess = z0 * s0**0.5
    OK=[]
    frac=0.2
    while len(OK)<2:
        OK = np.where(np.abs(Macquartz - zguess)/zguess < frac)[0] # gets 20% error range
        frac *= 5
    ztrials = Macquartz[OK]
    DMcosmicz = MacquartDM[OK]
    
    sz,eff0,effz,modFz = calc_effz(s0,DMcosmic0,z0,w0,DM0,freq,tsamp,alpha,ztrials,DMcosmicz)
    
    # if this fails, try again with a bigger range
    if np.min(sz) > s0 or np.max(sz) < s0:
        OK = np.where(np.abs(Macquartz - zguess)/zguess < 0.5)[0] # gets 50% error range
        ztrials = Macquartz[OK]
        DMcosmicz = MacquartDM[OK]
        sz,eff0,effz,modFz = calc_effz(s0,DMcosmic0,z0,w0,DM0,freq,tsamp,alpha,ztrials,DMcosmicz)
        
        if np.min(sz) > s0 or np.max(sz) < s0:
            # try with all of them
            ztrials=Macquartz
            DMcosmicz = MacquartDM
            sz,eff0,effz,modFz = calc_effz(s0,DMcosmic0,z0,w0,DM0,freq,tsamp,alpha,ztrials,DMcosmicz)
            
            
    # gets zmax via spline interpolation
    splines = interpolate.CubicSpline(sz[::-1],ztrials[::-1])
    zmax = splines(1.)
    
    
    if np.min(sz) > s0 or np.max(sz) < s0: # out of range...
        print("We have found a problem with our z guesses!!!")
        print(z0,s0)
        print(sz/s0)
        print("zmax found to be ",zmax)
        sz = s0 * (effz/eff0) * modFz
        
        #if False:
        plt.figure()
        plt.plot(ztrials,effz,label="efficiency")
        plt.plot(ztrials,effz/eff0,label="Rel efficiency")
        plt.plot(ztrials,modFz,label="Relative lum dist")
        plt.plot(ztrials,sz/s0,label="product")
        plt.plot([zmax,zmax],[0.,1./s0],color="black",linestyle=":")
        plt.plot([ztrials[0],zmax],[1./s0,1./s0],color="black",linestyle=":")
        
        plt.xlabel("z guesses")
        plt.ylabel("Relative efficiency factors")
        plt.yscale('log')
        #plt.ylim(0.1,10)
        plt.xlim(0.0001,0.1)
        plt.xscale('log')
        plt.legend()
        plt.tight_layout()
        plt.savefig("example_zguesses.pdf")
        plt.close()
    
    return zmax

def zefficiency(z0,w0,DM0,freq,tsamp,newz,newDM):
    """
    Calculates width if the burst had been at a higher redshift
    Returns efficiency: propto w**-0.5
    This modifies detectable fluence.
    
    The new DM and z must be input by the user.
    
    This is a similar model to that within the "survey" class.
    
    INPUTS:
        
        z0 [float]: measured FRB redshift
        w0 [float]: measured FRB width [ms]
        DM0 [float]: measured FRB *total* dispersion measures [pc cm^-3]
        freq [float]: frequency of detection [MHz]
        tsamp [float]: integration time of data [ms]
        newz [float]: redshift at which to calculate burst properties
        newDM [float]: DM at which to calculate burst properties
    """
    
    # calculates width components at z0
    dmsmear0 = DM0 * 4.15 * 1e6 * ((freq-0.5)**-2 - (freq+0.5)**-2)
    
    wintrinsic = (w0**2 - dmsmear0**2 - tsamp**2)
    if wintrinsic < 0.:
        wintrinsic = 0.
    else:
        wintrinsic = wintrinsic**0.5
    
    new_dmsmear = newDM * 4.15 * 1e6 * ((freq-0.5)**-2 - (freq+0.5)**-2)
    
    # reduces intrinsic width as 1+z
    new_int = wintrinsic * (1.+z0)/(1.+newz)
    new_w = ((new_int)**2 + new_dmsmear**2 + tsamp**2)**0.5
    eff = new_w **-0.5
    return eff

def calc_effz(s0,DMcosmic0,z0,w0,DM0,freq,tsamp,alpha,ztrials,DMcosmicz):
    """
    calculates s=SNR/SNRth that an FRB would have as a function of redshift
    
    Includes luminosity distance and efficiency factors.
    
    INPUTS:
        s0 [float]: measured SNR/SNRthresh of detected FRB
        DMcosmic0: estimated FRB *cosmic* dispersion measures [pc cm^-3]
        z0 [float]: measured FRB redshift
        w0 [float]: measured FRB width [ms]
        DM0 [float]: measured FRB *total* dispersion measures [pc cm^-3]
        freq [float]: frequency of detection [MHz]
        tsamp [float]: integration time of data [ms]
        ztrials [np.ndarray (floats)]: array of redshifts at which to calculate s
        alpha [float]: frequency dependence of FRBs - Fnu ~ nu^\alpha
    
    RETURNS:
        sz: s as a function of ztrials
        eff0 [float]: efficiency at detected redshift
        effz [np.ndarray (float)]: efficiency as a function of ztrials
        modFz: fluence as a function of redshift
    
    
    """
    # estimate DM at different z
    DMz = DM0 + DMcosmicz - DMcosmic0
    
    # calculates the energy corresponding to 1 Jy ms at z0
    E0 = cos.F_to_E(1.,z0,alpha)
    # calculates the fluence "per Jyms at z0" from an FRB at z
    modFz = cos.E_to_F(E0,ztrials,alpha)
    
    eff0 = zefficiency(z0,w0,DM0,freq,tsamp,z0,DM0)
    effz = zefficiency(z0,w0,DM0,freq,tsamp,ztrials,DMz)
    
    sz = s0 * (effz/eff0) * modFz
    
    return sz,eff0,effz,modFz


def get_macquart_z(DMeg,state,DMhost,zmin=1e-3,zmax=2,NZ=2000):
    """
    gets z(DM) from the Macquart relation
    
    INPUTS:
        DMeg [float or np.ndarray]: extragalactic DM at which
            to calculate z [pc/cm3]
        state [instance of parameter state class]
        DMhost: assumed value of DM host
        zmin [int]: minimum redshift for interpolation
        zmax [int]: maximum redshift for interpolation
        NZ [int]: number of redshifts for interpolation
        
    RETURNS:
        zFRBs (float or np.ndarray): Macquart relation expectation for
            redshift of FRBs with DMeg
    """
    
    # gets z from macquart relation
    zvals=np.linspace(zmin,zmax,NZ)
    
    macquart_relation=pcosmic.get_mean_DM(zvals, state)
    hosts = DMhost/(1+zvals)
    macquart_relation += hosts #adding the host contribution as a function of z 
    splines = interpolate.CubicSpline(macquart_relation,zvals)
    zFRBs = splines(DMeg)
    #print(zFRBs)
    
    return zFRBs


def get_DM_cosmic(z,state):
    """
    gets z(DM) from the Macquart relation
    """
    
    # gets z from macquart relation
    zvals=np.linspace(1e-3,3,3000)
    
    macquart_relation=pcosmic.get_mean_DM(zvals, state)
    splines = interpolate.CubicSpline(zvals,macquart_relation)
    DMcosmic = splines(z)
    
    return DMcosmic
