#!/usr/bin/env python
# coding: utf-8

"""
Contact: clancy.james@curtin.edu.au

This code calculates the probability of detecting a random burst with product pt ps pdm
less than some critical value.

Requirements:
    - Basic Python libraries (numpy, scipy, pandas, astropy, healpy)
    - LIGO libraries

Run with:
    - python3 pvale_calculation.py

Steps (approx outline of logic):
#1: p(ra): 1/nra (i.e. give equal weighting to all ra values)
#2: p(CL < 90): exposure-weighted integrated probability of CL<90 as function of that ra
#3: p(Ps|passed):
    exposure-weighted probability of given value of CL normalised by region with CL<90
    Ps is a function of CL: compare fraction over range ra to ra_GW, what is p(CL'<CL)?
    Calculate p using the *unweighted* exposure map, i.e. do not renormalise
#4: Pt is direct function of ra
# p(PtPs): Value of PtPs has probability equal to p(ra) * p(Ps|passed)
    sum this over all ra
    sum this over all dec points for given ra
    build up histogram
# p(PtPsPdm < R*):
    chance of product less than observed value
    Pdm is uniform 0 to 1 and independent (technically, 1/474, 2/474,..., 1)
    Hence, p(PtPsPdm < R*) = p(Pdm < R*/PtPs)
        = min(R*/PtPs,1)
    Sum this probability weighted by p(PtPs). Get the answer.

"""


import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy import signal
from scipy import interpolate
from scipy.interpolate import UnivariateSpline
import scipy.integrate as integrate
import pandas as pd
import healpy as hp
import os

import ligo.skymap
from ligo.skymap import postprocess
from ligo.skymap.postprocess import find_greedy_credible_levels
from ligo.skymap import moc
from ligo.skymap import io
from ligo.skymap import plot
from ligo.skymap.io import fits

import astropy
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


global outdir
outdir = 'pvalue_outputs/'
if not os.path.exists(outdir):
    print("mkdir ",outdir)
    os.mkdir(outdir)

def main(verbose=False):
    ##################################
    ###### Important constants #######
    
    # NOTE: below, the terms "ra" are used rather loosely
    # The skymap is fixed in time, so if we let time evolve,
    # we deal with that by changing position
    
    # FRB coordinates, these are fixed
    ra_frb = 255.72
    dec_frb = 21.52
    global ra_GW
    ra_GW = ra_frb-37.5 # ra of coordinate when GW is discovered
    
    # this is the distribution of times in the ra range, relative to the time of the FRB
    # (relative to FRB time for historical reasons)
    maxra=ra_frb+21.5*15 # 21.5 hr after the FRB, i.e. until 24hr after GW
    minra=ra_frb-4.5*15 # 4.5hr before the FRB, i.e. 2 hr before GW, 2.5 after
    global time_window
    time_window=maxra-minra # in degrees
    if verbose:
        print("Max and min ra are ",maxra,minra)
    
    
    ###### FINAL CORRECT SKYMAPS ######
    skymapfile = 'GW/GW190425_PublicationSamples.multiorder.fits,0' #GWTC Probability of spatial coincidence is 0.26476568419591673
    init_skymap(skymapfile)
    
    #############################
    # Step 1: Load data points for CHIME exposure
    # these were manusally extracted from CHIME Catalog 1 - Figure 5
    # make these splines global functions
    global ut_spl
    global lt_spl
    ut_spl,lt_spl=get_splines()
    
    # maximum ranges of declination over which to apply splines and search
    # below dec_min, exposure is zero
    dec_min = -7
    dec_max = 90
    
    # Integrating from min(CHIME dec) ~ -7 degrees to max(CHIME dec) ~ 90 degrees to find C:
    # this normalises the spline such that the chance to find an FRB somewhere is unity
    global C
    C = integrate.quad(dec_function, dec_min, dec_max)
    print("Normalisation constant C is", C[0])
    
    # Checking to see if normalisation works appropriately:
    sol = integrate.quad(normalised_E, dec_min, dec_max)
    print("Normalised E(delta) SHOULD be unity = 1. It is:", sol[0])
    
    # Step 8: Compute P_S
    delta_angle = 0.25
    decs = np.arange(dec_min, dec_max, delta_angle) # define declination list with d_delta = 0.25 for precision
    ras = np.arange(minra, maxra, delta_angle) # ra range to search
    
    # defines length for one radial increment
    rad_length = delta_angle*(np.pi/180) # find total length of RA(delta) where ci <= 0.667
    
    # critical value at which the FRB is found, for interest sake
    CV_frb, rad, decl = sky_search_single(ra_frb, dec_frb)
    print("FRB CV is ",CV_frb)
    
    # the below CV is used for determining if something passes the selection criteria
    CV=0.9 #this is the threshold
    print("Fixing CV at the critical value ",CV)
    
    # this routine pts retrieves a distribution in time offsets, pts
    pts=get_pts(ras,ra_frb,plot=False)
    
    uexp,lexp = get_int_values(decs,ut_spl,lt_spl,rad_length)
    
    # cvrahist is the distribution of confidence levels for a given ra
    # cvs are the bin centres
    cvs,cvrahist,plist=make_cv_hist(ras,decs,uexp,lexp,verbose=False,plot=True)
    
    # normalises such that the integral over dec = 1 for any given ra, i.e. "given there is an FRB at that ra"
    cvrahist *= ras.size
    
    # makes this a cumulative distribution from 0 to CV in the ra direction
    cum_cvrahist=make_cumulative_list(cvrahist,ras,plot=True)
    
    # now each column is the sum between ra_GW and ra
    tint_cum_hist=make_tint_hist(cum_cvrahist,ras,ra_GW)
    
    # does some plotting
    make_plots(plist,cvrahist,cum_cvrahist,tint_cum_hist)
    
    # final calculation, huzzay!
    calc_p_pass(ras,cvs,tint_cum_hist,cvrahist,CV_THRESH=0.9,Rstar=1.9e-4)
    
    # checks p_pass using real FRBs, not exposure

def calc_real_pass(ras,cvs,tint_cum_cvrahist,cvrahist,CV_THRESH=0.9,Rstar=1.9e-4):
    """
    ras: ra values
    cvs: confidence levels from GW sky map, between 0 and 1
    tint_cum_cvrahist: 2D numpy array
        [i,j]: ith ra, jth cv
        cumulative probability of observing value of likelihood < CV*
        in ra range ra to ra_GW, i.e. pspatial.
        We need to cut this off at value of CV=0.9
    cvrahist: 2D numpy array
        [i,j]: ith ra, jth cv
        probability of observing a value of cv for a given ra
        integrates over ra to unity, i.e. np.sum*cvrahist[i,:]=1.
        
    CV_THRESH: at 90%, events are no longer considered
        Hence, the probability that an event with CV>CV_THRESH
        has a product < R* is 0
    Rstar: observed critical value of PtPsPdm
    
    """
    # here, need to compare Rstar to chance of an event that definitely does pass cuts
    hours=26
    day_rate=1.93
    expected=hours*day_rate/24.
    Ppass=1.-np.exp(-expected)
    print("Ppass is ",Ppass)
    
    # makes a histogram for recording PsPt probability values
    Nspt=10000
    dNspt=1./Nspt # from zero to unity
    pspthist=np.zeros([Nspt])
    ctc=0.
    
    # this is the probability of observing that pt
    # it is constant - no correlation in time
    t_chance=1./(ras.size) # chance of being in this ra given it past cuts
    
    for i,ra in enumerate(ras):
        # this is the value of Pt that would be calculated for this ra value
        Ptime=get_ptime(ra)
        #print("Looping ra ",ra," with Ptime ",Ptime)
        # loop over values of CV, and convert them to P_spatials
        for icv,cv in enumerate(cvs):
            # if cv>CV_THRESH, we never count the observation
            # the weihted chance of seeing any given value of Pspatial
            # should only be proportional to the non-time-integrated histogram
            if cv >= CV_THRESH:
                #print(icv,cv,CV_THRESH)
                continue
            
            P_spatial = tint_cum_cvrahist[i,icv] #it's the chance of getting as low a value or lower given the raw bin
            thresh=Rstar/(P_spatial * Ptime)
            #print("For ra ",ra," and cl ",cv,
            #    " P_spatial is ",P_spatial,
            #    " Ptime is ",Ptime," so for Rstar ",Rstar," need ",thresh)
            
            # chance of this dm being less than the threshold
            dm_chance=min(thresh,1.)
            
            # chance of observing this value of cv
            s_chance = cvrahist[i,icv]
            
            total_chance=dm_chance * t_chance * s_chance # cvrahist is local chance of getting that value
            ctc += total_chance
    print("Cumulative total chance is ",ctc)
    print("Including chance of no FRB, we find ",ctc*Ppass)

   
def get_splines(plot=False):
    'NOTE: There is a (very) slight and brief drop in CHIME exposure time ~ 20 degrees, but fitting a spline requires the exposure'
    'time does not decrease with declination. We keep the exposure time as ~ 15 hours around this period. This overestimates'
    'P_S, but the effect will be very minimal.'
    
    global outdir
    
    ut_decs = [-6, 0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 67.5, 70, 72.5,
           75, 77.5, 80, 82.5, 85, 87, 89] # upper transit declinations (degrees)
    ut_exp = [12.5, 13, 13, 13, 14, 15, 15, 15, 15, 16, 17, 19, 20, 23, 27, 30, 31, 40,
                46, 50, 63, 80, 140, 200, 1000] # upper transit exposure times (hours)
    lt_decs = [72.5, 75, 77.5, 80, 82.5, 85, 87, 89] # lower transit declinations (degrees)
    lt_exp = [30, 38, 50, 70, 90, 120, 200, 900] # lower transit exposure times (hours)

    'NOTE: Lower transit data is only relevant for dec > 70 degrees'
    
    # Step 2: Fit two separate splines to the data points
    
    ut_spl = UnivariateSpline(ut_decs, ut_exp) # upper tranist
    lt_spl = UnivariateSpline(lt_decs, lt_exp) # lower transit
    
    
    #    PLOT SPLINE FITS     #
    if plot:
        ut_spl.set_smoothing_factor(0.5)
        lt_spl.set_smoothing_factor(0.5)
        fig, ax = plt.subplots(figsize=(15,7))
        plt.scatter(ut_decs, np.log10(ut_exp), c = 'darkblue', s = 50, label = 'Upper Transit: CHIME (2021)')
        plt.scatter(lt_decs, np.log10(lt_exp), s = 50, c = 'red', label = 'Lower Transit: CHIME (2021)', marker = 's')
        plt.plot(ut_decs, np.log10(ut_spl(ut_decs)), c= 'skyblue', lw=4, linestyle = '--', label = 'Upper Transit: Fit')
        plt.plot(lt_decs, np.log10(lt_spl(lt_decs)), c='darkorange', lw=4, linestyle = '--', label = 'Lower Transit: Fit')
        plt.axvline(21.7, c = 'black', label = 'FRB 20190425A') # declination of FRB 20190425A
        plt.xlabel('DEC (degrees)', fontsize=18)
        plt.ylabel('$log_{10}(\mathrm{Exposure\ Time})$ (hours)', fontsize=18)
        plt.legend(fontsize=18)
        plt.savefig(outdir+'Spline.pdf')
    
    return ut_spl,lt_spl

# Step 3: Define function that combines the upper and lower transit data for dec > 70 degrees

def spl(x):
    
    if x < 70:
        return ut_spl(x)
    else:
        return ut_spl(x) + lt_spl(x)

# Step 4: Find normalisation constant C

'NOTE: Below the variable changes from x (general 2D sky coordinate) to delta (declination), as there is no right'
'ascension (RA) dependence. We account for RA using the 2*pi. Since the CHIME exposure function is equivalent to'
'the detection probability per square degree per FRB, we account for the square degree dependence with cos(delta)'
    
def dec_function(delta):
    global time_window
    return (np.deg2rad(time_window))*np.cos(np.deg2rad(delta))*spl(delta) # make sure delta is in radians

# Step 5: Define normalised exposure function

def normalised_E(delta):
    global time_window
    global C
    return ((np.deg2rad(time_window))*np.cos(np.deg2rad(delta))*spl(delta))/C[0]


# Step 6: Read in GW190425 GWTC-2 skymap data (or any relevant skymap)
def init_skymap(skymapfile):
    global max_nside
    global ipix
    global order
    global max_order
    global prob
    skymap = io.fits.read_sky_map(skymapfile, moc=True) # name of skymap file
    sky_map = np.flipud(np.sort(skymap, order='PROBDENSITY'))
    order, ipix = moc.uniq2nest(sky_map['UNIQ'])
    max_order = np.max(order)
    max_nside = ah.level_to_nside(max_order)
    dA = moc.uniq2pixarea(sky_map['UNIQ'])
    dP = sky_map['PROBDENSITY'] * dA
    prob = np.cumsum(dP)
    

# Step 7: Define function that returns credible interval for each coordinate
def sky_search_single(rad, decl):
    global max_nside
    global ipix
    global order
    global max_order
    global prob
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



def get_pts(ras,ra_frb,plot=False):
    """
    Converts actual ra values into time offsets
    Args:
    ras: numpy array of ra values, lowest to highest
    ra_frb: ra of the FRB
    returns: argument for offset in time
    """
    #norm pt
    norm_pt=0.
    pts=[]
    for ra in ras:
        #accounts for wrapping
        pt1=np.abs(ra-ra_frb)
        if ra < ra_frb:
            pt2=np.abs(ra-ra_frb+2.*np.pi)
        else:
            pt2=np.abs(ra-ra_frb-2.*np.pi)
        pt=min(pt1,pt2)
        pts.append(pt)
        norm_pt += pt
    pts=np.array(pts)
    pts /= norm_pt
    
    if plot:
        plt.figure()
        plt.plot(ras,pts)
        plt.xlabel('ra')
        plt.ylabel('$\\Delta T$')
        plt.tight_layout()
        plt.savefig(outdir+'pt.pdf')
        plt.close()
    return pts


def get_int_values(decs,ut_spl,lt_spl,rad_length,plot=False):
    """
    Calculates the exposure to each declination bin, from dec to dec+delta_angle
    
    here, decs are the LOWEST values in the bin
    """
    delta_angle=decs[1]-decs[0]
    ### pre-calculates declination exposures
    uexp=np.zeros([decs.size])
    lexp=np.zeros([decs.size])
    for i,dec in enumerate(decs):
        integral, error = integrate.quad(lambda x: rad_length*np.cos(x*np.pi/180)*ut_spl(x)/C[0], dec, dec+delta_angle)
        uexp[i]=integral
        if dec < 70.:
            lexp[i]=0.
        else:
            integral, error = integrate.quad(lambda x: rad_length*np.cos(x*np.pi/180)*lt_spl(x)/C[0], dec, dec+delta_angle)
            lexp[i]=integral
    return uexp,lexp
    
def get_ptime(ra1):
    " ras must be in degrees here"
    global ra_GW
    dt=np.abs(ra1-ra_GW)
    hours=dt/15. # from degrees to hours
    day_rate=1.93
    expected=hours*day_rate/24.
    PT=1.-np.exp(-expected)
    #print(hours,PT)
    return PT

def make_cv_hist(ras,decs,uexp,lexp,verbose=False,plot=True):
    """
    For each ra, we make a histogram of the confidence levels as a function of dec
    """
    global outdir
    
    # we now make a histogram of p(CV) for each ra. Only care about cumulative values, so it can be a fine grid
    ncv=1000
    nra=ras.size
    cvs=np.linspace(1./ncv,1.0,ncv)
    dcv=1./ncv
    cvrahist=np.zeros([nra,ncv])
    plist=[]
    cvrafile = 'cvrahist.npy'
    plistfile = 'plist.npy'
    if os.path.exists(cvrafile) and os.path.exists(plistfile):
        cvrahist=np.load(outdir+cvrafile)
        plist=np.load(outdir+plistfile)
    else:
        for i,ra in enumerate(ras): # min(CHIME RA) to max(CHIME RA)
            radcoords=[]
            deccoords=[]
            ps=0. # sums p_spatial for this ra range
            # finds all declinations passing selection criteria
            
            minp=100.
            minicv=1000
            for j,dec in enumerate(decs): # upper transit only, over full range
                
                probab, rad, decl = sky_search_single(ra, dec)
                plist.append(probab)
                # get probability bin
                icv=int(probab/dcv)
                
                if (probab < minp):
                    minp=probab
                    contrib=uexp[j]
                if icv >= ncv:
                    icv=ncv-1
                if icv < minicv:
                    minicv=icv
                # get probability increment for that bin
                # i.e. increments probability bin icv with
                # exposure uexp(j)
                cvrahist[i,icv] += uexp[j]
                
                # upper transit
                if dec >= 70.:
                    ra2 = ra + 180.
                    if ra2 > 360:
                        ra2 -= 360.
                    probab2, rad2, decl2 = sky_search_single(ra2, dec)
                    
                    # get probability bin
                    icv=int(probab2/dcv)
                    if icv >= ncv:
                        icv=ncv-1
                    
                    # get probability increment for that bin
                    cvrahist[i,icv] += lexp[j]
            print("For ra ",ra," min prob is ",minp,minicv,contrib)
        np.save(outdir+cvrafile,cvrahist)
        plist=np.array(plist)
        np.save(outdir+plistfile,plist)
    
    # checks the normalisation of the cvrahist when summing over declination
    if verbose:
        temp=np.sum(cvrahist,axis=1)
        print("Sum of histogram is ",temp)
    
    # use bin centres
    cvs -= 0.5/ncv
    
    if plot:
        plt.figure()
        
        aspect=550
        
        plt.xlabel('RA')
        plt.ylabel('Confidence level thingy')
        plt.imshow(np.log10(cvrahist.T),origin='lower',extent=[ras[0],ras[-1],0,1],aspect=aspect)
        cb=plt.colorbar()
        cb.set_label('P(CL)')
        plt.tight_layout()
        plt.savefig(outdir+'cvrahist.pdf')
        plt.close()
        
        # now makes down-scaled version
        nx,ny=cvrahist.shape
        red1=cvrahist.reshape(nx,int(ny/10),10)
        red2=np.sum(red1,axis=2)
        red3=red2.T
        red4=red3.reshape(int(ny/10),int(nx/10),10)
        red5=np.sum(red4,axis=2)
        red6=red5.T
        
        plt.figure()
        aspect=550
        plt.xlabel('RA')
        plt.ylabel('Confidence level thingy')
        plt.imshow(np.log10(red6.T),origin='lower',extent=[ras[0],ras[-1],0,1],aspect=aspect)
        cb=plt.colorbar()
        cb.set_label('P(CL)')
        plt.tight_layout()
        plt.savefig(outdir+'reduced_cvrahist.pdf')
        plt.close()
        
        filled=np.count_nonzero(cvrahist)
    return cvs,cvrahist,plist
    


# we now know the probability distribution of each critical value for each RA
# expect each histogram to sum vertically to a constant
# to get the probability of any given value of CV is to integrate everything below it
# makes cumulative histogram of probabilities of any given CV
def make_cumulative_list(cvrahist,ras,plot=False):
    cum_cvrahist = np.copy(cvrahist)
    nra,ncv=cvrahist.shape
    for icv in np.arange(ncv):
        cum_cvrahist[:,icv] += np.sum(cvrahist[:,:icv],axis=1)
    
    
    if plot:
        # now makes down-scaled version
        nx,ny=cum_cvrahist.shape
        red1=cum_cvrahist.reshape(nx,int(ny/10),10)
        red2=np.sum(red1,axis=2)
        red3=red2.T
        red4=red3.reshape(int(ny/10),int(nx/10),10)
        red5=np.sum(red4,axis=2)
        red6=red5.T
        
        plt.figure()
        aspect=550
        plt.xlabel('RA')
        plt.ylabel('Confidence level thingy')
        plt.imshow(red6.T,origin='lower',extent=[ras[0],ras[-1],0,1],aspect=aspect)
        cb=plt.colorbar()
        cb.set_label('cumulative P(CL < CL*)')
        plt.tight_layout()
        plt.savefig(outdir+'reduced_cum_cvrahist.pdf')
        plt.close()
    
    return cum_cvrahist

def make_plots(plist,cvrahist,cum_cvrahist,tint_cum_hist):
    """
    Makes some diagnostic figures to understand
    the arcane process of p-value calculation
    """
    global outdir
    
    plt.figure()
    plt.hist(plist)
    plt.savefig(outdir+'histogram_of_p.pdf')
    plt.close()
    
    plt.figure()
    plt.imshow(cvrahist,origin='lower')
    plt.xlabel('likelihood')
    plt.ylabel('ra')
    
    cb=plt.colorbar()
    cb.set_label('p(likelihood,ra)')
    plt.clim(0,1e-2)
    plt.tight_layout()
    plt.savefig(outdir+'cv_ra_hist.pdf')
    plt.close()
    
    plt.figure()
    plt.imshow(cum_cvrahist,origin='lower')
    plt.xlabel('likelihood')
    plt.ylabel('ra')
    
    cb=plt.colorbar()
    cb.set_label('p(<likelihood,ra)')
    plt.clim(0,0.1)
    plt.tight_layout()
    plt.savefig(outdir+'cum_cv_ra_hist.pdf')
    plt.close()
    
    plt.figure()
    plt.imshow(tint_cum_hist,origin='lower')
    plt.xlabel('likelihood')
    plt.ylabel('ra')
    cb=plt.colorbar()
    cb.set_label('p(<likelihood,ra)')
    plt.clim(0,0.1)
    plt.tight_layout()
    plt.savefig(outdir+'tint_cum_cv_ra_hist.pdf')
    plt.close()
    
# Pspatial is calculated relative to all regions (ras) between that ra and the FRB

def make_tint_hist(cum_cvrahist,ras,ra_GW):
    """
    Simulates the distribution of P(spatial) as a function
    of FRB time by integrating the P(spatial|ra) distribution
    of cum_cvrahist between the GW event time and the FRB ras
    value. This creates the same distribution as done for the P_S
    calculation used to evaluate the P_S|ra calculation of FRB 190425A.
    
    Input:
    cum_cvrahist: 2D np array, dimentions ra x confidence level.
        Contains cumulative value of p(CL <= CL*|ra) for each bin.
    ras: 1D np array of right ascension values (degrees)
    ra_GW: ra of CHIME at the time when the GW event occured, i.e.
        where an FRB would have been detected if temporally coincident
    
    Return: p(s|ra)
    
    """
    
    
    nra=ras.size
    diffs=np.abs(ras-ra_GW)
    ira_GW=np.argmin(diffs)
    # need to evaluate the chance of landing in a certain spot
    #from ra 0 to mean. Function of ra and GW confidence level
    tint_cum_hist=np.zeros(cum_cvrahist.shape)
    print("Making tint hist")
    
    # identifies the time bin that coincides with the GW event
    # this is the one that has lowest P(time) - others relative to that
    print("CHIME at GW t=0 at ",ra_GW,", so the i of the GW is ",ira_GW)
    
    # we sum the probabilities of getting a particular GW confiendence limit
    # for all times from the event time i *before* the GW event, to the
    # GW event. This is the same procedure used for the actual P_chance
    # calculation
    for i in np.arange(ira_GW):
        tint_cum_hist[i,:]=np.sum(cum_cvrahist[i:ira_GW,:],axis=0)
        # normalise by number of rad bins
        # because we have equal probability
        inorm=ira_GW-i
        # we normalise the below over the number of time bins,
        # i.e. so the sum above is actually a mean
        tint_cum_hist[i,:] /= inorm
    
    # now sums in opposite direction, i.e. for all events *after* the GW event
    for i in np.arange(nra-ira_GW):
        effi=ira_GW+i+1
        tint_cum_hist[effi-1,:]=np.sum(cum_cvrahist[ira_GW:effi,:],axis=0)
        # normalise by number of rad bins
        # because we have equal probability
        inorm=i+1
        tint_cum_hist[effi-1,:] /= inorm
    
    # now normalises according to total in sum, i.e. sum for all ra is unity
    # clearly however, some values useless. Need to include p(>90% given ra).
    # this is done at a later stage. Not here - i.e. we do not renormalise 
    # the probability at this point.
    #for i in np.arange(nra):
    #    tint_cum_hist[i,:]/=np.sum(tint_cum_hist[i,:])
    
    return tint_cum_hist

def calc_p_pass(ras,cvs,tint_cum_cvrahist,cvrahist,CV_THRESH=0.9,Rstar=1.9e-4):
    """
    ras: ra values
    cvs: confidence levels from GW sky map, between 0 and 1
    tint_cum_cvrahist: 2D numpy array
        [i,j]: ith ra, jth cv
        cumulative probability of observing value of likelihood < CV*
        in ra range ra to ra_GW, i.e. pspatial.
        We need to cut this off at value of CV=0.9
    cvrahist: 2D numpy array
        [i,j]: ith ra, jth cv
        probability of observing a value of cv for a given ra
        integrates over ra to unity, i.e. np.sum*cvrahist[i,:]=1.
        
    CV_THRESH: at 90%, events are no longer considered
        Hence, the probability that an event with CV>CV_THRESH
        has a product < R* is 0
    Rstar: observed critical value of PtPsPdm
    
    """
    # here, need to compare Rstar to chance of an event that definitely does pass cuts
    hours=26
    day_rate=1.93
    expected=hours*day_rate/24.
    Ppass=1.-np.exp(-expected)
    # perhaps this should instead be the expected rate of events passing?
    # in any case, it should account for the fact that subsequent calculations
    # are *given* an FRB is observed in the assigned time range
    print("Probability of passing the temporal cuts is ",Ppass)
    
     #observed value of Pt Ps Pdm
    #Rstar_prime=1.9e-4 / Ppass
    
    # makes a histogram for recording PsPt probability values
    Nspt=10000
    dNspt=1./Nspt # from zero to unity
    pspthist=np.zeros([Nspt])
    ctc=0.
    
    # this is the probability of observing that pt
    # it is constant - no correlation in time
    t_chance=1./(ras.size) # chance of being in this ra given it past cuts
    
    ptime=np.zeros([ras.size])
    
    for i,ra in enumerate(ras):
        # this is the value of Pt that would be calculated for this ra value
        Ptime=get_ptime(ra)
        ptime[i]=Ptime
        #print("Looping ra ",ra," with Ptime ",Ptime)
        # loop over values of CV, and convert them to P_spatials
        for icv,cv in enumerate(cvs):
            # if cv>CV_THRESH, we never count the observation
            # the weihted chance of seeing any given value of Pspatial
            # should only be proportional to the non-time-integrated histogram
            if cv >= CV_THRESH:
                #print(icv,cv,CV_THRESH)
                continue
            
            P_spatial = tint_cum_cvrahist[i,icv] #it's the chance of getting as low a value or lower given the raw bin
            thresh=Rstar/(P_spatial * Ptime)
            #print("For ra ",ra," and cl ",cv,
            #    " P_spatial is ",P_spatial,
            #    " Ptime is ",Ptime," so for Rstar ",Rstar," need ",thresh)
            
            # chance of this dm being less than the threshold
            dm_chance=min(thresh,1.)
            
            # chance of observing this value of cv
            s_chance = cvrahist[i,icv]
            
            total_chance=dm_chance * t_chance * s_chance # cvrahist is local chance of getting that value
            ctc += total_chance
    print("Cumulative total chance is ",ctc)
    print("Including chance of no FRB, we find ",ctc*Ppass)
    
    
    if plot:
        plt.figure()
        plt.plot(ras,ptime)
        plt.xlabel('ra [deg]')
        plt.ylabel
main()
