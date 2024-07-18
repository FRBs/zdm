"""
Script to calculate V and Vmax for CRAFT FRBs
"""

import numpy as np
import os
#from zdm import cosmology as cos
# sets up cosmological grid
#    cos.set_cosmology()
#    cos.init_dist_measures()
from zdm import pcosmic
from scipy import interpolate
from zdm import real_loading as loading
from matplotlib import pyplot as plt
from scipy.special import jv as BessJ
from scipy import constants
from pkg_resources import resource_filename
from zdm import cosmology as cos
from scipy.integrate import quad
import time


# initialises cosmology and distance measures 


def main(fname,outfile,Nsfr,alpha=0.,loc=False,minz=None,write=True,
    laststrings=None,LimZ=None,zfrac = 0.):
    """
    
    """
    global macquartz
    global macquartDM
    global state
    
    # initialises cosmology and distance measures 
    #state = init_cos()
    
    # loads unlocalised data, gets Macquart z
    
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
        zDM = get_macquart_z(DMeg,state,DMhost)
        plot_z_error(zDM,old_zDM)
    
    Enu = F_to_E(Fnu,zDM,alpha)
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
        max_zDM = get_macquart_z(DM,state,0.) #calculates max z assuming *all* DM is extragalactic
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
    
    # gets DM cosmic contributions for data
    DMcosmic = get_DM_cosmic(zDM,state)
    
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
    dt1=0
    dt2=0
    # tries getting zmax for one FRB
    for i,FRB in enumerate(names):
        #if not (FRB == 20180110):
        #    continue
        print("Starting frb ",FRB)
        # calculates zmax at the B of detection
        s = SNR[i]/SNRth[i]
        Bzmax = calc_zmax(s,zDM[i],DM[i],DMcosmic[i],width[i],freqMHz[i],
            tsamp[i],state,macquartz,macquartDM,alpha)
        
        s = SNR[i]/SNRth[i] / Bvals[i]
        Czmax = calc_zmax(s,zDM[i],DM[i],DMcosmic[i],width[i],freqMHz[i],
            tsamp[i],state,macquartz,macquartDM,alpha)
        
        Volume = 0.
        Vmax = 0.
        
        # note: last one is beam centre. This is a speedup
        if write:
            toloop = bvals
        else:
            toloop = bvals[-3:]
        for j,logB in enumerate(toloop):
         
            B = 10.**logB
            s = SNR[i]/SNRth[i] * B/Bvals[i]
            t0=time.time()
            zmax = calc_zmax(s,zDM[i],DM[i],DMcosmic[i],width[i],freqMHz[i],
                tsamp[i],state,macquartz,macquartDM,alpha)
            
            # limits to maximum volume
            if LimZ is not None and zmax > LimZ:
                zmax = LimZ
            
            t1=time.time()
            #dVdOmega = integrate_volume(zmax,Nsfr=Nsfr) #weighted cubic Mpc per steradian
            
            dVdOmega = Vsplines(zmax)
            #print("Compare direct integration to spline solution: ",dVdOmega,dVdOmega2,"   ",(dVdOmega-dVdOmega2)/dVdOmega)
            
            t2=time.time()
            dt1 += (t1-t0)
            dt2 += (t2-t1)
            dVmax = dVdOmega * omegab[j]
            # dVmax is now the volume elements weighted over that solid angle
            
            Vmax += dVmax
            # we keep this tempzmax to be printed later
            tempzmax = zmax
            # calculates the volume by limiting to zmax 
            if zmax > zDM[i]:
                zmax = zDM[i]
            dVdOmega = Vsplines(zmax)
            #dVdOmega = integrate_volume(zmax,Nsfr=Nsfr)
            dV = dVdOmega * omegab[j]
            Volume += dV
            
        # in Gpc
        Bdl = cos.DL(Bzmax)/1e3
        Cdl = cos.DL(Czmax)/1e3
        
        energyErg = F_to_E(Fnu[i],zDM[i],alpha,bandwidth=1.) # bandwidth of unity
        JHz = energyErg / 1e7
        
        
        ### constructs string to write to data file ####
        string = '{0:8} {1:1.3e} {2:6.5f} {3:6.5f} {4:1.3e} {5:1.2e} {6:6.3f}\n'.format(
            int(FRB),JHz,Bzmax,Czmax,Volume,Vmax,Volume/Vmax)
        # printing out Czmax and Cdl would be super useful for the paper! Wait, why Dl? Who cares!
        #string = '{0:8} {1:1.3e} {2:6.4f} {3:6.4f}'.format(
        #    int(FRB),JHz,Bzmax,Bdl)
        
        if write:
            f.write(string),SNR[i],DM[i],DMG[i],freqMHz[i],tsamp[i],width[i],BeamThetaDeg[i]
        
        
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
    print("Calculation times are ",dt1,dt2)  
    
    if write:
        f.close()
    if laststrings is None:
        return tempstrings
    else:
        return laststrings   
            
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
    weighted_dVdtau = cos.dVdtau(z) * SFR(z)**Nsfr[0]
    return weighted_dVdtau
    
def SFR(z):
    """
    Madau & dickenson 2014
    Arguments:
        z (float): redshift
    Leading constant: normalises to unity at z=0
    """
    return 1.0025738*(1+z)**2.7 / (1 + ((1+z)/2.9)**5.6)


    
def calc_effz(s0,DMcosmic0,z0,w0,DM0,freq,tsamp,alpha,ztrials,DMcosmicz):
    """
    calculates the relative SNR/SNRth that an FRB would have
    as a function of redshift
    
    Includes luminosity distance and efficiency factors
    """
    # estimate DM at different z
    DMz = DM0 + DMcosmicz - DMcosmic0
    
    # calculates the energy corresponding to 1 Jy ms at z0
    E0 = F_to_E(1.,z0,alpha)
    # calculates the fluence "per Jyms at z0" from an FRB at z
    modFz = E_to_F(E0,ztrials,alpha)
    
    eff0 = zefficiency(z0,w0,DM0,freq,tsamp,z0,DM0)
    effz = zefficiency(z0,w0,DM0,freq,tsamp,ztrials,DMz)
    
    # using Wayne's method
    # this just scaled effective width as 1+z, hence
    # efficiency went as (1+z)**-0.5
    # i.e. it ignores the fact the intrinsic width behaves differently...
    # eff0 = (1+z0)**-0.5
    # effz = (1+ztrials)**-0.5
    
    sz = s0 * (effz/eff0) * modFz
    
    return sz,eff0,effz,modFz
    

def calc_zmax(s0,z0,DM0,DMcosmic0,w0,freq,tsamp,state,Mz,MDM,alpha):
    """
    Routine which calculates the maximum redshift at which
    an FRB could have been detected
    
    We begin with the ratio s0, which is SNR(det)/SNR(thresh)
    This tells us how much fluence we can lose as a function
    of redshift
    
    We then account for luminosity distance, and changing efficiency
    with distance
    
    '0' properties indicate those at detection
    
    Mz and MDM are pre-computed values of Macquart relation
    
    We must have values of z which span the range from the minimum z
    at which the FRB lies, to the maximum z at which it could be detected
    E.g. at S/N of 1000, that's a factor of ~10 for SNRthresh=10
    """
    
    
    # we begin by making a naive guess at zmax based upon
    # Cartesian geometry, and then we interpolate exact
    # values about this to get a precise answer
    zguess = z0 * s0**0.5
    OK=[]
    frac=0.2
    while len(OK)<2:
        OK = np.where(np.abs(Mz - zguess)/zguess < frac)[0] # gets 20% error range
        frac *= 5
    ztrials = Mz[OK]
    DMcosmicz = MDM[OK]
    
    sz,eff0,effz,modFz = calc_effz(s0,DMcosmic0,z0,w0,DM0,freq,tsamp,alpha,ztrials,DMcosmicz)
    
    # if this fails, try again with a bigger range
    if np.min(sz) > s0 or np.max(sz) < s0:
        OK = np.where(np.abs(Mz - zguess)/zguess < 0.5)[0] # gets 50% error range
        ztrials = Mz[OK]
        DMcosmicz = MDM[OK]
        sz,eff0,effz,modFz = calc_effz(s0,DMcosmic0,z0,w0,DM0,freq,tsamp,alpha,ztrials,DMcosmicz)
        
        if np.min(sz) > s0 or np.max(sz) < s0:
            # try with all of them
            ztrials=Mz
            DMcosmicz = MDM
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
        exit()
    
    return zmax
    
def zefficiency(z0,w0,DM0,freq,tsamp,newz,newDM):
    """
    Calculates width if the burst had been at a higher redshift
    Returns efficiency: propto w**-0.5
    This modifies detectable fluence
    
    The new DM and z must be input by the user
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


def dvdtau(z):
    """ Comoving volume element [Mpc^3 dz sr^-1].
    normalsied by proper time (1+z)^-1 to convert
    rates.
    """ 
    iz=np.array(np.floor(z/DZ)).astype('int')
    kz=z/DZ-iz
    return (dvdtaus[iz]*(1.-kz)+dvdtaus[iz+1]*kz) #removed the 1/(1+z) dependency


def F_to_E(F,z,alpha,bandwidth=1e9):
    """ Converts a fluence to an energy
    Formula from Macquart & Ekers 2018
    Fluence assumed to be in Jy ms
    Energy is returned in ergs (here, erg/Hz)
    """
    E = F * (4*np.pi*(cos.dl(z))**2/(1.+z)**(2.+alpha))
    E *= 9.523396e22*bandwidth 
    return E

def E_to_F(E,z,alpha, bandwidth=1e9):
    """ Converts an energy to a fluence
    Formula from Macquart & Ekers 2018
    Energy is assumed to be in ergs
    Fluence returned in Jy ms
    """
    F=E/(4*np.pi*(cos.dl(z))**2/(1.+z)**(2.+alpha))
    F /= 9.523396e22*bandwidth # see below for constant calculation
    return F
  

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
    
def get_macquart_z(DMeg,state,DMhost):
    """
    gets z(DM) from the Macquart relation
    """
    
    # gets z from macquart relation
    zvals=np.linspace(1e-3,2,2000)
    
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

# setting minz only recalculates those FRBs which would otherwise have z < 0.
# this is used to generate data for the "unbiased" unlocalised FRBs
if True:
    loc=False
    minz = 0.00024
    LimZ=None
    infile="Data/all_unloc_frb_data.dat"
    # if minz is not None, only consider FRBs with negative z
    
    for zfrac in np.linspace(0.,1.,11):
        Nsfr = 0
        strings = None
        alpha=0
        outfile = "MinzOutput/Minzmacquart_vvmax_data_NSFR_"+str(Nsfr)+"_alpha_"+str(alpha)+"_"+str(zfrac)[0:3]+".dat"
        strings = main(infile,outfile,Nsfr,alpha=alpha,loc=loc,minz=minz,write=True,laststrings=strings,LimZ=LimZ,zfrac=zfrac)
        
        exit()
exit()
########## run setting zmax = 0.7 for localised FRBs ###########

# limits maximum redshift to 0.7. This is the unbiased calculation for localised FRBs
if False:
    loc=True
    LimZ=0.7
    infile="Data/unbiased_loc_data.dat"
    for Nsfr in np.linspace(0,2,21):
        strings = None
        for alpha in [0,-1.5]:
            outfile = "UnbiasedLocalisedOutput/localised_vvmax_data_NSFR_"+str(Nsfr)[0:3]+"_alpha_"+str(alpha)+".dat"
            main(infile,outfile,Nsfr,alpha=alpha,loc=loc,minz=None,laststrings=strings,LimZ=LimZ)

######## standard run for initial part of the paper ############

# main data runs over first, unlocalised, than localised

if False:
    loc=False
    infile="Data/all_unloc_frb_data.dat"
    minz = None
    LimZ=None
    opdir="UnlocOutput/"
    if not os.path.isdir(opdir):
        os.mkdir(opdir)
    
    for j,Nsfr in enumerate(np.linspace(0,2,21)):
        
        strings = None
        for alpha in [0,-1.5]:
            
            outfile = opdir+"macquart_vvmax_data_NSFR_"+str(Nsfr)+"_alpha_"+str(alpha)+".dat"
            #    outfile = "Output/macquart_vvmax_data_NSFR_"+str(Nsfr)+"_alpha_"+str(alpha)+".dat"
            strings = main(infile,outfile,Nsfr,alpha=alpha,loc=loc,minz=minz,write=True,laststrings=strings,LimZ=LimZ)

###### initial set of localised FRBs, including 171020 ######

if True:
    loc=True
    opdir="LocOutput/"
    if not os.path.isdir(opdir):
        os.mkdir(opdir)
    LimZ=None
    minz=None
    infile = "data/all_loc_frb_data_w171020.dat"
    for Nsfr in np.linspace(0,2,21):
        strings = None
        for alpha in [0,-1.5]:
            outfile = opdir+"localised_vvmax_data_NSFR_"+str(Nsfr)[0:3]+"_alpha_"+str(alpha)+".dat"
            string=main(infile,outfile,Nsfr,alpha=alpha,loc=loc,minz=minz,write=True,laststrings=None,LimZ=LimZ)
