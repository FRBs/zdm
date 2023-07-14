#!/usr/bin/env python
# coding: utf-8


"""
DESCRIPTION:
    - loops over GW events and calculates P_S *given*
        an FRB occurs in a particular time window
    - The time window is defined in 'main' via drastart
        and drastop. It calculates the LST of CHIME at the
        point of the event, and uses this to define the ra
        range.

CREDIT:
    - Alexandra Moroianu (originally inspired from LIGO codebase)
    - Adapted and extended by Clancy James


REQUIREMENTS:
    - 

INPUTS:
    - GW skymap fits files, in 'Skymaps/'
        Available from GWOSC: https://www.gw-openscience.org/
        

"""

import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy import signal
from scipy import interpolate
from scipy.interpolate import UnivariateSpline
import scipy.integrate as integrate
import pandas as pd

import ligo.skymap
from ligo.skymap import postprocess
from ligo.skymap.postprocess import find_greedy_credible_levels
from ligo.skymap import moc
from ligo.skymap import io
from ligo.skymap import plot
from ligo.skymap.io import fits
import astropy
import healpy as hp

from astropy import visualization
from astropy.visualization import astropy_mpl_style

from astropy.utils.data import get_pkg_data_filename
from astropy.io import fits
from astropy.table import Table
import astropy_healpix as ah
from astropy.coordinates import SkyCoord
from astropy import units as u
import astropy.time
from astropy.time import Time
import astropy.coordinates as coord
from astropy.coordinates import Angle


# some global constants. There are constants of the CHIME
# exposure which need to be passed everywhere.
global C
global ut_spl
global lt_spl
global time_window
global dec_min
global dec_max

dec_min=-7 #minimum declination of CHIME catalogue
dec_max=90 # max declination of interest


def main(verbose=False):
    """
    Uppermost program, that shows the division of the code
    into two parts: calculating P_S for the time window
    of GW190425 (a key part of the final p-value
    calculation), and calculating the probability of
    having an FRB in coincidence with any GW event in the
    catalogue.
    """
    
    # just 190425A for observed time offset (2.5 hr)
    # use four skymaps
    calc_PS_190425A(verbose)
    
    # All GW for 2hr time window. Takes a long time!
    # this data set uses updated superevents
    calc_selection_criteria_all_GW(verbose)
    
    
    

def calc_PS_190425A(verbose=False):
    """
    Generates P_S for 190425A
    """
    # the time window of the search is passed to several routines
    global time_window
    
    # defines the ra range over which we are assumed to search
    # NOTE: we approximate a sidereal day by a calendar day here
    # The below are the ra offsets corresponding to a search time
    # window of -2 hr before the GW event to 24hr after
    drastart=0.
    drastop=37.5 # 2.5hr after the GW event
    time_window=drastop-drastart
    
    # sets the global parameters ut_spl, lt_spl, and constant C
    # Gets CHIME exposure splins
    get_chime_splines(plot=True)
    
    # reads in the available GW event information
    # returns fits file names, and LSTs at CHIME
    # for each event.
    indir='Fits/'
    names,LSTs=read_info()
    iGW = np.where(names[0:8]=='GW190425')
    name=names[iGW]
    LST=LSTs[iGW]
    
    infiles={}
    infiles["GWTC2"]='GW190425_PublicationSamples.multiorder.fits,0' #correct!
    infiles["Bayestar"]='bayestar.fits' # original name lost
    infiles["Superevent"]='gw190425z_skymap.multiorder.fits,0' # correct!
    infiles["LALInference"]='GW190425_LALInference.fits.gz,0' # correct!
    
    
    # will calculate CV at these coordinates
    ra_frb = 255.72
    dec_frb = 21.52
    CVcoords=[ra_frb,dec_frb]
    
    # loops over events, calculating P_S for each.
    # i.e. IF (FRB goes off uniformly in time window
    # THEN what is chance to lie in 90% likelihood region?
    # this routine takes quite some time, since it
    # integrates over the entire sky in 0.25 deg increments
    # The skymaps used are from updated superevents
    indir = "GW/"
    for i,pipeline in enumerate(infiles):
        infile=indir+infiles[pipeline]
        infile='Fits/GW190425.fits'
        rastart=LST.degree+drastart
        rastop=LST.degree+drastop
        if verbose:
            print("Calculating P_S for ",pipeline)
        PS=calc_PS(infile,rastart,rastop,verbose=verbose,CVcoords=CVcoords)
        print("For ",i,"th pipeline, ",pipeline,", PS is ",PS)
        

def calc_selection_criteria_all_GW(verbose=False):
    """
    Outer loop to iterate over events
    Loops over all calculation codes, and generates P_S
    """
    # the time window of the search is passed to several routines
    global time_window
    
    # defines the ra range over which we are assumed to search
    # NOTE: we approximate a sidereal day by a calendar day here
    # The below are the ra offsets corresponding to a search time
    # window of -2 hr before the GW event to 24hr after. This is not
    # used in the analysis, since it is implausible that two merging
    # black holes produce a post-merger event. Instead, we use
    # only the two hours beforehand, for e.g. merging charged
    # black hole models.
    # To calculate chance of entire 26hr window
    # drastart=-2*360./24
    # drastop=360.
    # To calculate chance for events including a BH merger
    drastart = -2*360./24
    drastop = 0.
    
    time_window=drastop-drastart
    
    # sets the global parameters ut_spl, lt_spl, and constant C
    # Gets CHIME exposure splins
    get_chime_splines(plot=True)
    
    # reads in the available GW event information
    # returns fits file names, and LSTs at CHIME
    # for each event.
    indir='Fits/'
    names,LSTs=read_info()
    
    # loops over events, calculating P_S for each.
    # i.e. IF (FRB goes off uniformly in time window
    # THEN what is chance to lie in 90% likelihood region?
    # this routine takes quite some time, since it
    # integrates over the entire sky in 0.25 deg increments
    # The skymaps used are from updated superevents
    for i,name in enumerate(names):
        infile=indir+name
        
        LST=LSTs[i]
        rastart=LST.degree+drastart
        rastop=LST.degree+drastop
        if verbose:
            print("The range for GW ",name," is ",rastart,rastop)
        PS=calc_PS(infile,rastart,rastop,verbose=verbose)
        print("For ",i,"th event, ",name,", PS is ",PS)
    
def read_info(infile="GWTC2_info.csv"):
    """
    Reads GWTC2 info, and calculate LST at CHIME location
    """
    
    from astropy.coordinates import EarthLocation
    from astropy.time import Time
    from astropy import units as u
    
    data=np.loadtxt(infile,delimiter=',',dtype='str')
    
    names=data[:,0]
    gps=data[:,1].astype('float')
    
    
    # converts GPS times to LST at CHIME location
    CHIME_LAT = 49 + 19/60. + 15/3600.
    CHIME_LON = (119 + 37/60. + 25/3600)*-1. #west
    #TIME_MJD = 58598.44989134716 # this is the time of FRB 20190425A, replace this with the time of the GW event
    
    observing_location = EarthLocation(lat=CHIME_LAT*u.deg, lon=CHIME_LON*u.deg)
    observing_time = Time(gps,format='gps',location=observing_location) #, location=observing_location
    #observingtime.format('mjd')
    LSTs = observing_time.sidereal_time('apparent')
    #print("LSTs were ",LST.degree)
    return(names,LSTs)


def get_chime_splines(verbose=False,plot=False):
    """
    Manually extract data points from CHIME Catalog 1 - Figure 5
    https://ui.adsabs.harvard.edu/abs/2021AAS...23832501M/abstract
    """
    
    global ut_spl, lt_spl
    
    # hard-coded data extracted from CHIME catalogue
    ut_decs = [-6, 0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 67.5, 70, 72.5,
        75, 77.5, 80, 82.5, 85, 87, 89] # upper transit declinations (degrees)
    ut_exp = [12.5, 13, 13, 13, 14, 15, 15, 15, 15, 16, 17, 19, 20, 23, 27, 30, 31, 40,
        46, 50, 63, 80, 140, 200, 1000] # upper transit exposure times (hours)
    lt_decs = [72.5, 75, 77.5, 80, 82.5, 85, 87, 89] # lower transit declinations (degrees)
    lt_exp = [30, 38, 50, 70, 90, 120, 200, 900] # lower transit exposure times (hours)

    #NOTE: Lower transit data is only relevant for dec > 70 degrees
    #Fit two separate splines to the data points
    ut_spl = UnivariateSpline(ut_decs, ut_exp) # upper tranist
    lt_spl = UnivariateSpline(lt_decs, lt_exp) # lower transit
    ut_spl.set_smoothing_factor(0.5)
    lt_spl.set_smoothing_factor(0.5)
    if plot:
        ############################
        #    PLOT SPLINE FITS     #
        ############################
        
        fig, ax = plt.subplots(figsize=(15,7))
        plt.scatter(ut_decs, np.log10(ut_exp), c = 'darkblue', s = 50, label = 'Upper Transit: CHIME (2021)')
        plt.scatter(lt_decs, np.log10(lt_exp), s = 50, c = 'red', label = 'Lower Transit: CHIME (2021)', marker = 's')
        plt.plot(ut_decs, np.log10(ut_spl(ut_decs)), c= 'skyblue', lw=4, linestyle = '--', label = 'Upper Transit: Fit')
        plt.plot(lt_decs, np.log10(lt_spl(lt_decs)), c='darkorange', lw=4, linestyle = '--', label = 'Lower Transit: Fit')
        plt.axvline(21.7, c = 'black', label = 'FRB 20190425A') # declination of FRB 20190425A
        plt.xlabel('DEC (degrees)', fontsize=18)
        plt.ylabel('$log_{10}(\mathrm{Exposure\ Time})$ (hours)', fontsize=18)
        #plt.title('CHIME exposure vs dec', fontsize=25)
        plt.legend(fontsize=18)
        plt.savefig('Spline.pdf')
        plt.close()
    
    # sets the normalisation constant C, such that integrating over a sphere gives you unity
    normalise_splines()
    
    # checks everything is in order
    global dec_min,dec_max
    # Checking to see if normalisation works appropriately:
    solution = integrate.quad(normalised_E, dec_min, dec_max)
    if verbose:
        print("Normalised E(delta) SHOULD be unity = 1. It is:", solution[0])
    
def spl(x):
    """
    x: declination (degrees)
    returns: exposure (normalised)
    
    This function chooses whether or not to use one or both transits
    
    """
    if x < 70:
        return ut_spl(x)
    else:
        return ut_spl(x) + lt_spl(x)

def dec_function(delta):
    """
    Returns weighting for declintion interval
    """
    global time_window
    return (np.deg2rad(time_window))*np.cos(np.deg2rad(delta))*spl(delta) # make sure delta is in radians


def normalise_splines(verbose=False):
    """
    Finds normalisation constant C for the splines
    
    NOTE: Below the variable changes from x (general 2D sky coordinate)
    to delta (declination), as there is no right ascension (RA) dependence.
    We account for RA using the 2*pi. Since the CHIME exposure function is
    equivalent to the detection probability per square degree per FRB,
    we account for the square degree dependence with cos(delta).
    """
    global C
    result = integrate.quad(dec_function, dec_min, dec_max)
    C = result[0]
    if verbose:
        print("Normalisation constant C is", C)
    

# Step 5: Define normalised exposure function

def normalised_E(delta):
    """
    This routine returns the exposure function after it has been normalised
    by the constant C, so that integrating it over he entire sky gives 1.
    
    Inuts:
        delta: declination (degrees)
    Returns:
        Normalised exposure p(delta)
    """
    global time_window, C
    return ((np.deg2rad(time_window))*np.cos(np.deg2rad(delta))*spl(delta))/C


def calc_PS(skymapfile,minra,maxra,verbose=False,CV=0.9,CVcoords=None):
    """
    Calculates the probability to find an FRB between rastart and rastop given the skymap
    and CHIME's exposure.
    
    Uses a confidence value (CV) of 0.9, i.e. 90%, as per the search criteria.
    Can be changed, or over-ridden with values at a particular coordinate
    """
    
    
    AngleStep=0.25
    HAngleStep = AngleStep/2.
    
    # time in degrees (15 per hour) in ra to search prior to FRB
    if verbose:
        print("Max and min ra are ",maxra,minra)
    
    #read in the skymap, extract probabilities and their normalisation
    skymap = io.fits.read_sky_map(skymapfile, moc=True) # name of skymap file
    sky_map = np.flipud(np.sort(skymap, order='PROBDENSITY'))
    order, ipix = moc.uniq2nest(sky_map['UNIQ'])
    max_order = np.max(order)
    max_nside = ah.level_to_nside(max_order)
    dA = moc.uniq2pixarea(sky_map['UNIQ'])
    dP = sky_map['PROBDENSITY'] * dA
    prob = np.cumsum(dP)
    
    # Step 8: Compute P_S
    integral_list = [] # define list for probability summation
    error_list = [] # define list for integration errors
    
    decs = np.arange(dec_min, dec_max, AngleStep) # define declination list with d_delta = 0.25 for precision
    ras = np.arange(minra, maxra, AngleStep) # ra range to search
    
    
    if CVcoords is not None:
        # define CV according to location at coordinates
        ra_frb = CVcoords[0]
        dec_frb = CVcoords[1]
        CV, rad, decl = sky_search_single(ra_frb, dec_frb,max_nside,max_order,order,ipix,prob)
        print("CV set to be ",CV," at FRB location ",CVcoords)
    
    allrad=[]
    alldec=[]
    simple=0.
    
    # loops over coordinates, gets total probability
    for i in decs: # upper transit only, over full range
        radcoords = []
        deccoords = []
        
        for j in ras: # min(CHIME RA) to max(CHIME RA)
        
            probab, rad, decl = sky_search_single(j, i,max_nside,max_order,order,ipix,prob)
            
            if probab <= CV: # credible interval FRB 20190425A falls in = 0.667
                'NOTE: if you use another skymap, change the above credible interval accordingly'
                radcoords.append(rad)
                deccoords.append(decl)
        
        # find total length in ra direction in radians where p<CV.
        rad_length = AngleStep*len(radcoords)*(np.pi/180) # find total length of RA(delta)
        
        # integrate exposure function over this length:
        #integral, error = integrate.quad(lambda x: (rad_length)*np.cos(x*np.pi/180)*ut_spl(x)/C, i, i+0.25)
        
        # following formula: length in DEC * length in RA * cosine factor (declination) * CHIME coverage
        simple += AngleStep * (rad_length)*np.cos((i+HAngleStep)*np.pi/180)*ut_spl((i+HAngleStep))/C
        #integral_list.append(integral)
        #error_list.append(error)
        allrad.append(radcoords)
        alldec.append(deccoords)
    
    for i in decs: # lower transit only, i.e. for dec > 70
        if i < 70: 
            continue
        radcoords = []
        deccoords = []
        for j in ras: # min(CHIME RA) to max(CHIME RA)
            j += 180.
            while j>360:
                j -= 360. #lower transit, separated by 180 degrees on sky
            probab, rad, decl = sky_search_single(j, i,max_nside,max_order,order,ipix,prob)
            
            if probab <= CV: # credible interval FRB 20190425A falls in = 0.667
                'NOTE: if you use another skymap, change the above credible interval accordingly'
                radcoords.append(rad)
                deccoords.append(decl)
        
        rad_length = AngleStep*len(radcoords)*(np.pi/180) # find total length of RA(delta) where ci <= 0.667
        
        # originally, exposure calculated via an integral, ala:
        # integrate exposure function over this length:
        #integral, error = integrate.quad(lambda x: (rad_length)*np.cos(x*np.pi/180)*lt_spl(x)/C, i, i+0.25)
        # But taking the exposure to be constant over 0.25 degrees is much quicker! And just as accurate.
        simple += AngleStep * (rad_length)*np.cos((i+HAngleStep)*np.pi/180)*lt_spl((i+HAngleStep))/C
        
        #integral_list.append(integral)
        #error_list.append(error)
        allrad.append(radcoords)
        alldec.append(deccoords)
    
    if verbose:
        print("Result was ",simple)
    return simple #p_spatial
    


def sky_search_single(rad, decl,max_nside,max_order,order,ipix,prob):
    """
    # Step 7: Define function that returns credible interval for each coordinate
    rad,decl are the coordinates, the last four are weird healpix things...
    """
    ra = np.deg2rad(rad)
    dec = np.deg2rad(decl)
    theta = 0.5*np.pi - dec
    phi = ra
    true_pix = hp.ang2pix(max_nside, theta, phi, nest=True)
    max_ipix = ipix << np.int64(2 * (max_order - order))
    idxs = np.argsort(max_ipix)
    true_idx = idxs[np.digitize(true_pix, max_ipix[idxs]) - 1]

    ci = prob[true_idx] # ci = credible interval

    return ci, rad, decl # credible interval, RA, declination

main()
