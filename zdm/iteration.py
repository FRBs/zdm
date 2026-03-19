"""
Likelihood calculation routines for z-DM grids.

This module provides functions for computing log-likelihoods of FRB survey
data given model predictions from a Grid object. These likelihoods are used
for parameter estimation via MCMC or maximum likelihood methods.

The main likelihood components include:
- p(DM, z): Probability of observed DM and redshift (if localized)
- p(N): Poisson probability of total number of FRBs detected
- p(SNR): Signal-to-noise ratio distribution
- p(w, tau): Width and scattering time distributions

Main Functions
--------------
- `get_log_likelihood`: Compute total log-likelihood for a grid/survey pair
- `calc_likelihoods_1D`: Likelihood for 1D (DM only) FRB data
- `calc_likelihoods_2D`: Likelihood for 2D (DM + z) localized FRBs

Example
-------
>>> from zdm import iteration
>>> ll = iteration.get_log_likelihood(grid, survey)
>>> print(f"Log-likelihood: {ll}")

Author: C.W. James
"""

import os
import time
from IPython.terminal.embed import embed
import matplotlib.pyplot as plt
import numpy as np
import pandas
from scipy.optimize import minimize
# to hold one of these parameters constant, just remove it from the arg set here
from zdm import cosmology as cos
from scipy.stats import poisson
import scipy.stats as st
from zdm import repeat_grid as zdm_repeat_grid


def get_log_likelihood(grid, s, norm=True, psnr=True, Pn=False, pNreps=True, ptauw=False, pwb=False):
    """Compute total log-likelihood for a grid given survey data.

    This is the main likelihood function used for parameter estimation.
    It combines multiple likelihood components depending on what data
    is available (DM only, localized z, SNR, etc.).

    Parameters
    ----------
    grid : Grid or repeat_Grid
        Grid object containing model predictions.
    s : Survey
        Survey object containing FRB observations.
    norm : bool, optional
        If True, include normalization terms. Default True.
    psnr : bool, optional
        If True, include SNR distribution likelihood. Default True.
    Pn : bool, optional
        If True, include Poisson likelihood for total number. Default False.
    pNreps : bool, optional
        If True, include number of repeaters likelihood. Default True.
    ptauw : bool, optional
        If True, include p(tau, width) likelihood. Default False.
    pwb : bool, optional
        If True, include width/beam likelihood. Default False.

    Returns
    -------
    float
        Total log-likelihood value.
    """

    if isinstance(grid, zdm_repeat_grid.repeat_Grid):
        # Repeaters
        if s.nDr==1:
            llsum = calc_likelihoods_1D(grid, s, norm=norm, psnr=psnr,
                            dolist=0, grid_type=1, Pn=Pn, pNreps=pNreps, ptauw=ptauw, pwb=pwb)
        elif s.nDr==2:
            llsum = calc_likelihoods_2D(grid, s, norm=norm, psnr=psnr,
                            dolist=0, grid_type=1, Pn=Pn, pNreps=pNreps, ptauw=ptauw, pwb=pwb)
        elif s.nDr==3:
            llsum11 = calc_likelihoods_1D(grid, s, norm=norm, psnr=psnr,
                                        dolist=0, grid_type=1, Pn=Pn, pNreps=pNreps, ptauw=ptauw,
                                        pwb=pwb)
            llsum2 = calc_likelihoods_2D(grid, s, norm=norm, psnr=psnr,
                                        dolist=0, grid_type=1, Pn=False, pNreps=False, ptauw=ptauw,
                                        pwb=pwb)
            llsum = llsum1 + llsum2
        else:
            print("Implementation is only completed for nD 1-3.")
            exit()

        # Singles
        if s.nDs==1:
            llsum1 = calc_likelihoods_1D(grid, s, norm=norm, psnr=psnr,
                                                dolist=0, grid_type=2, Pn=Pn, ptauw=ptauw,
                                                pwb=pwb)
            llsum += llsum1
        elif s.nDs==2:
            llsum1 = calc_likelihoods_2D(grid, s, norm=norm, psnr=psnr,
                                                dolist=0, grid_type=2, Pn=Pn, ptauw=ptauw,
                                                pwb=pwb)
            llsum += llsum1
        elif s.nDs==3:
            llsum1 = calc_likelihoods_1D(grid, s, norm=norm, psnr=psnr,
                                                    dolist=0, grid_type=2, Pn=Pn, ptauw=ptauw,
                                                    pwb=pwb)
            llsum2 = calc_likelihoods_2D(grid, s, norm=norm, psnr=psnr,
                                                    dolist=0, grid_type=2, Pn=False, ptauw=ptauw,
                                                    pwb=pwb)
            llsum = llsum + llsum1 + llsum2
        else:
            print("Implementation is only completed for nD 1-3.")
            exit()
    else:
        if s.nD==1:
            llsum = calc_likelihoods_1D(grid, s, norm=norm, psnr=psnr,
                                                            dolist=0, Pn=Pn, ptauw=ptauw,pwb=pwb)
        elif s.nD==2:
            llsum = calc_likelihoods_2D(grid, s, norm=norm, psnr=psnr,
                                                            dolist=0, Pn=Pn, ptauw=ptauw,pwb=pwb)
        elif s.nD==3:
            llsum1 = calc_likelihoods_1D(grid, s, norm=norm, psnr=psnr,
                                                                    dolist=0, Pn=Pn, ptauw=ptauw,
                                                                    pwb=pwb)
            llsum2 = calc_likelihoods_2D(grid, s, norm=norm, psnr=psnr,
                                                                    dolist=0, Pn=False, ptauw=ptauw,
                                                                    pwb=pwb)
            llsum = llsum1 + llsum2
        else:
            print("Implementation is only completed for nD 1-3.")
            exit()
    return llsum

def calc_likelihoods_1D(grid,survey,doplot=False,norm=True,psnr=True,
                    Pn=False,pNreps=True,ptauw=False,pwb=False,dolist=0,grid_type=0,PATH=False):
    """ Calculates 1D likelihoods using only observedDM values
    Here, Zfrbs is a dummy variable allowing it to be treated like a 2D function
    for purposes of calling.
    
    grid: the grid object calculated from survey
    
    survey: survey object containing the observed z,DM values
    
    doplot: will generate a plot of z,DM values
    
    psnr:
        True: calculate probability of observing each FRB at the observed SNR
        False: do not calculate this

    Pn:
        True: calculate probability of observing N FRBs
        False: do not calculate this
    
    pNreps:
        True: calculate probability of the number of repetitions for each repeater
        False: do not calculate this
    
    ptauw:
        True: calculate probability of intrinsic width and scattering *given* total width
        False: do not calculate this
    
    pwb:
        True: calculate probability of specific width and beam values, and psnr | bw
        False: do not calculate this; simply sum psnr over all possible p(b,w)
    
    PATH (bool):
        True: returns p(z|[properties]) for these FRBs, as well as p(properties).
            If set, dolist is returned as the third parameter
        False [default]: does not return these values.
    
    dolist:
        0: returns total log10 likelihood llsum only [float]
        1: also returns a dict of individual statistical contributions to log-likelihood
        2: also returns the above for every FRB individually
        Structure of llsum and longlist:
            ["pzDM"]["pDM"]
            ["pN"]
            ["Nexpected"] (only for 
            ["ptauw"]
                ["pbar"]
                ["piw"]
                ["ptau"]
                ["w_indices"]
            ["pbw"]
                ["pb"]
                ["pw"]
                ["pbgw"]
                ["pwgb"]
                ["pbw"]
                ["psnr_gbw"]
                ["psnrbw"]
            lllist: <Nfrbs>
            longlist [list of lists]:
                Pn: float
                zDM: PzDM
                zDM extras: P(z) P(DM), P(DM|z), P(z|DM)
                ptauw: p(tau|wtot), p(wtot), p(snr|wtot)
                psnr: p_snr (over all beams/widths)
                pwb: p(snr|b,w,z,dm), p(snr,b,w|z,DM), p(b|zDM), p(w|zDM), p(b|w,zDM), p(w|b,zDM), p(wb|zDM)
        else: returns nothing (actually quite useful behaviour!)
    
    grid_type:
        0: normal zdm grid
        1: assumes the grid passed is a repeat_grid.zdm_repeat_grid object and calculates likelihood for repeaters
        2: assumes the grid passed is a repeat_grid.zdm_repeat_grid object and calculates likelihood for single bursts

    """
    
    if ptauw:
        if not survey.backproject:
            print("WARNING: cannot calculate ptauw for this survey, please initialised backproject")
    
    # Determine which array to perform operations on and initialise
    if grid_type == 1: 
        rates = grid.exact_reps 
        if survey.nozreps is not None:
            if PATH: # we are doing this for every FRB regardless
                nozlist = np.concatenate((survey.nozreps,survey.zreps))
            else:
                nozlist=survey.nozreps
        else:
            raise ValueError("No non-localised singles in this survey, cannot calculate 1D likelihoods")
    elif grid_type == 2: 
        rates = grid.exact_singles 
        if survey.nozsingles is not None:
            if PATH:
                nozlist=np.concatenate((survey.nozsingles,survey.zsingles))
            else:
                nozlist=survey.nozsingles
        else:
            raise ValueError("No non-localised repeaters in this survey, cannot calculate 1D likelihoods")
    else: 
        rates=grid.rates 
        if survey.nozlist is not None:
            if PATH: # we are doing this for every FRB regardless
                nozlist = np.concatenate((survey.nozlist,survey.zlist))
            else:
                nozlist = survey.nozlist
        else:
            raise ValueError("No non-localised FRBs in this survey, cannot calculate 1D likelihoods")
    
    # extract extragalactic DMEGs
    DMobs=survey.DMEGs[nozlist]
    
    dmvals=grid.dmvals
    zvals=grid.zvals
    
    # start by collapsing over z
    pdm=np.sum(rates,axis=0)
    
    if np.sum(pdm) == 0:
        if dolist==0:
            return -np.inf
        elif dolist==1:
            return -np.inf, None
        elif dolist==2:
            return -np.inf, None, None
    
    if norm:
        global_norm=np.sum(pdm)
        log_global_norm=np.log10(global_norm)
    else:
        global_norm = 1.
        log_global_norm=0
    
    idms1,idms2,dkdms1,dkdms2 = grid.get_dm_coeffs(DMobs)
    
    ################ Calculation of p(DM) #################
    if grid.state.MW.sigmaDMG == 0.0 and grid.state.MW.sigmaHalo == 0.0:
        if np.any(DMobs < 0):
            raise ValueError("Negative DMobs with no uncertainty")

        # Linear interpolation between DMs
        pvals=pdm[idms1]*dkdms1 + pdm[idms2]*dkdms2
    else:
        dm_weights, iweights = calc_DMG_weights(DMobs, survey.DMhalos[nozlist], survey.DMGs[nozlist], dmvals, grid.state.MW.sigmaDMG, 
                                                 grid.state.MW.sigmaHalo, grid.state.MW.logu)
        pvals = np.zeros(len(idms1))
        # For each FRB
        for i in range(len(idms1)):
            pvals[i]=np.sum(pdm[iweights[i]]*dm_weights[i])
    
    # sums over all FRBs for total likelihood
    llsum=np.sum(np.log10(pvals))-log_global_norm*DMobs.size
    
    # initialise dicts to return detailed log-likelihood information
    longlist={}
    longlist["pzDM"]={}
    longlist["pzDM"]["pdm"]=np.log10(pvals)-log_global_norm
    
    
    if PATH:
        # create PATH dict and add list of p(DM) values
        PATH_OP={}
        PATH_OP["pdm"] = pvals/global_norm
        
    lllist={}
    lllist["pzDM"]={} # pz,DM
    lllist["pzDM"]["pdm"]=llsum
    
    ########################################################
    # calculates a p(z) distribution for each FRB, allowing other
    # distributions to be weighted by p(z).
    # ensures a normalised p(z) distribution for each FRB (shape: nz,nDM)
    noztau_in_noz=[]
    
    if grid.state.MW.sigmaDMG == 0.0 and grid.state.MW.sigmaHalo == 0.0:
        # here, each FRB only has two DM weightings (linear interolation)
        zidms1,zidms2,zdkdms1,zdkdms2 = grid.get_dm_coeffs(DMobs)
        tomult = rates[:,zidms1]*zdkdms1 + rates[:,zidms2]*zdkdms2
        # normalise to a p(z) distribution for each FRB
        tomult /= np.sum(tomult,axis=0)
        
    else:
        dm_weights, iweights = calc_DMG_weights(DMobs, survey.DMhalos[nozlist],
                                        survey.DMGs[nozlist], dmvals, grid.state.MW.sigmaDMG, 
                                         grid.state.MW.sigmaHalo, grid.state.MW.logu)
        # here, each FRB has many DM weightings
        tomult = np.zeros([grid.zvals.size,len(iweights)])
        # construct a p(z) distribution.
        for iFRB,indices in enumerate(iweights):
            # we construct a p(z) vector for each FRB
            indices = indices[0]
            tomult[:,iFRB] = np.sum(rates[:,indices] * dm_weights[iFRB],axis=1)
        # normalise to a p(z) distribution for each FRB
        tomult /= np.sum(tomult,axis=0)
    
    if PATH:
        PATH_OP["pzgdm"] = tomult # normalised p(z|DM) distribution
    
    ########### Calculation of p((Tau,w)) ##############
    if ptauw:
        # checks which have OK tau values - in general, this is a subset
        # ALSO: note that this only checks p(tau,iw | w)! It does NOT
        # evaluate p(w)!!! Which is a pretty key thing...
        noztaulist = []
        inoztaulist = []
        for i,iz in enumerate(nozlist):
            if iz in survey.OKTAU:
                noztaulist.append(iz) # for direct indexing of survey
                inoztaulist.append(i) # for getting a subset of zlist
        Wobs = survey.WIDTHs[noztaulist]
        Tauobs = survey.TAUs[noztaulist]
        Iwobs = survey.IWIDTHs[noztaulist]
        ztDMobs=survey.DMEGs[noztaulist]
        
        # gets indices of noztaulist within nozlist
        tz_tomult = tomult[:,:inoztaulist]
    
        # This could all be precalculated within the survey.
        iws1,iws2,dkws1,dkws2 = survey.get_w_coeffs(Wobs) # total width in survey width bins
        itaus1,itaus2,dktaus1,dktaus2 = survey.get_internal_coeffs(Tauobs) # scattering time tau
        iis1,iis2,dkis1,dkis2 = survey.get_internal_coeffs(Iwobs) # intrinsic width
        
        # vectors below are [nz,NFRB] in length
        piws = survey.pws[:,iis1,iws1]*dkis1*dkws1 \
            + survey.pws[:,iis1,iws2]*dkis1*dkws2 \
            + survey.pws[:,iis2,iws1]*dkis1*dkws1 \
            + survey.pws[:,iis2,iws2]*dkis1*dkws2
        
        ptaus = survey.ptaus[:,itaus1,iws1]*dktaus1*dkws1\
            + survey.ptaus[:,itaus1,iws2]*dktaus1*dkws2 \
            + survey.ptaus[:,itaus2,iws1]*dktaus1*dkws1 \
            + survey.ptaus[:,itaus2,iws2]*dktaus1*dkws2
        
        # we now multiply by the z-dependencies
        ptaus *= zt_tomult
        piws *= zt_tomult
        
        # sum down the redshift axis to get sum p(tau,w|z)*p(z)
        ptaus = np.sum(ptaus,axis=0)
        piws = np.sum(piws,axis=0)
        
        bad1 = np.where(piws==0)
        bad2 = np.where(ptaus==0)
        piws[bad1] = 1e-10
        ptaus[bad2] = 1e-10
        pbars = 0.5*ptaus + 0.5*piws # take the mean of these two
        
        llptw = np.sum(np.log10(ptaus))
        llpiw = np.sum(np.log10(piws))
        
        # while we calculate llpiw, we don't add it to the sum
        # this is because w and tau are not independent!
        # p(iw|tau,w) = \delta(iw-(w**2 - tau**2)**0.5)
        # However, numerical differences will affect this
        # hence, we add half of eavh value here
        #llsum += 0.5*llpiw
        #llsum += 0.5*llptw
        llpbar = np.sum(np.log10(pbars))
        llsum += llpbar
        
        lllist["ptauw"]={}
        # appending total of each to log0-likelihood list
        lllist["ptauw"]["piw"]=llpiw
        lllist["ptauw"]["ptw"]=llptw
        lllist["ptauw"]["pbar"]=llpbar
        
        # appending individual FRB data to long long list
        longlist["ptauw"]={}
        longlist["ptauw"]["pbar"]=np.log10(pbars)
        longlist["ptauw"]["piw"]=np.log10(piws)
        longlist["ptauw"]["ptau"]=np.log10(ptaus)
        longlist["ptauw"]["w_indices"]=inoztaulist
        
    ############# Assesses total number of FRBs, P(N) #########
    # TODO: make the grid tell you the correct normalisation
    if Pn and (survey.TOBS is not None):
        if grid_type==1:
            observed=survey.NORM_REPS
            C = grid.Rc
            reps=True
        elif grid_type==2:
            observed=survey.NORM_SINGLES
            C = grid.Rc
            reps=True
        else:
            observed=survey.NORM_FRB
            C = 10**grid.state.FRBdemo.lC
            reps=False
        expected=CalculateIntegral(rates,survey,reps)
        expected *= C

        Pn=Poisson_p(observed,expected)
        
        if Pn==0:
            Nll=-1e10
        else:
            Nll=np.log10(Pn)
        
        lllist["pN"]=Nll
        lllist["Nexpected"]=expected
        llsum += Nll
    else:
        lllist["Nexpected"]=-1
        lllist["pN"]=0
    
    if PATH:
        if not Pn:
            PATH_OP["pn"] = 1.
        else:
            PATH_OP["pn"] = Pn
    
    # this is updated version, and probably should overwrite the previous calculations
    if psnr:
        # We now evaluate p(snr) at every point in b,w,and z space
        # This is p(snr) = p(Eobs) dE / \int_Eth^inf p(E) dE
        # We then sum p(snr) over the three above dimensions,
        # normalising in each case.
        
        # calculate vector of grid thresholds
        Emax=10**grid.state.energy.lEmax
        Emin=10**grid.state.energy.lEmin
        gamma=grid.state.energy.gamma
        psnr=np.zeros([DMobs.size]) # has already been cut to non-localised number
        
        # Evaluate thresholds at the exact DMobs
        DMEGmeans = survey.DMs[nozlist] - np.median(survey.DMGs + survey.DMhalos)
        idmobs1,idmobs2,dkdmobs1,dkdmobs2 = grid.get_dm_coeffs(DMEGmeans)
        
        # Linear interpolation
        Eths = grid.thresholds[:,:,idmobs1]*dkdmobs1 + grid.thresholds[:,:,idmobs2]*dkdmobs2
        
        # get the correct p(z) distributions to weight the likelihoods by
        if grid.state.MW.sigmaDMG == 0.0:
            # Linear interpolation
            rs = rates[:,idms1]*dkdms1+ rates[:,idms2]*dkdms2
        else:
            rs = np.zeros([grid.zvals.size, len(idms1)])
            # For each FRB
            for i in range(len(idms1)):
                # For each redshift
                for j in range(grid.zvals.size):
                    rs[j,i] = np.sum(grid.rates[j,iweights[i]] * dm_weights[i]) / np.sum(dm_weights[i])
                    
        # this has shape nz,nFRB - FRBs could come from any z-value
        nb = survey.beam_b.size
        nw,nz,nfrb = Eths.shape
        zpsnr=np.zeros([nz,nfrb])
        # numpy flattens this to the order of [z0frb0,z0f1,z0f2,...,z1f0,...]
        # zpsnr = zpsnr.flatten()
        
        # determine if the weights have redshift dependence or not
        if grid.eff_weights.ndim ==2:
            zwidths = True
        else:
            zwidths = False
        
        # this variable keeps the normalisation of sums over p(b,w) as a function of z
        pbw_norm = 0
        
        if ptauw and not pwb:
            # hold array representing p(w)
            dpbws = np.zeros([nw,nz,nfrb])
        
        if pwb:
            psnrbws = np.zeros([nb,nw,nz*nfrb]) # holds psnr_gbw * p(b,w,) for each b,w bin
            psnr_gbws = np.zeros([nb,nw,nz*nfrb]) # holds psnr_gbw * p(b,w,) for each b,w bin
            pbws = np.zeros([nb,nw,nz*nfrb]) # holds p(bw given z,dm) for each b,w, bin
            
        for i,b in enumerate(survey.beam_b):
            #iterate over the grid of weights
            bEths=Eths/b #this is the only bit that depends on j, but OK also!
            #now wbEths is the same 2D grid
            # bEobs has dimensions Nwidths * Nz * NFRB
            bEobs=bEths*survey.Ss[nozlist] #should correctly multiply the last dimensions
            for j,w in enumerate(grid.eff_weights):
                # p(SNR | b,w,DM,z) is given by differential/cumulative
                # however, p(b,w|DM,z) is given by cumulative*w*Omegab / \sum_w,b cumulative*w*Omegab
                # hence, the factor of cumulative cancels when calculating p(SNR,w,b), which is what we do here
                differential = grid.array_diff_lf(bEobs[j,:,:],Emin,Emax,gamma) * bEths[j,:,:]
                cumulative=grid.array_cum_lf(bEobs[j,:,:],Emin,Emax,gamma)
                
                if zwidths:
                    usew = np.repeat(w,nfrb).reshape([nz,nfrb]) # old
                else:
                    usew = w
                
                # this keeps track of the \sum_w,b cumulative*w*Omegab
                dpbw = survey.beam_o[i]*usew*cumulative
                pbw_norm += dpbw
                zpsnr += differential*survey.beam_o[i]*usew
                
                if ptauw and not pwb:
                    # record probability of this w summed over all beams for each FRB
                    dpbws[j,:,:] += dpbw
                
                if pwb:
                    # psnr given beam, width, z,dm
                    cumulative = cumulative.flatten() # first index is [0,0], next is [0,1]
                    differential = differential.flatten()
                    
                    
                    OK = np.where(cumulative > 0)[0]
                    
                    if zwidths:
                        usew = usew.flatten()[OK]
                    
                    psnr_gbws[i,j,OK] = differential[OK]/cumulative[OK]
                    
                    # psnr given beam, width, z,dm
                    psnrbws[i,j,OK] = differential[OK]*survey.beam_o[i]*usew
                    
                    # total probability of that p(w,b)
                    pbws[i,j,OK] = survey.beam_o[i]*usew*cumulative[OK]
                    
                
        # calculate p(w)
        if ptauw and not pwb:
            # we would like to calculate \int p(w|z) p(z) dz
            # we begin by calculating p(w|z), below, by normalising for each z
            # normalise over all w values for each z
            # Q: should we calculate p(w|b,z) then multiply by p(b,w)?
            dpbws /= np.sum(dpbws,axis=0)
            temp = dpbws[iws1,:,inoztaulist]
            # tomult is p(z) distribution
            temp *= tomult.T
            pws = np.sum(temp,axis=1) # sum in the linear domain - it's summing over p(z)
            bad = np.where(pws == 0.)[0]
            pws[bad] = 1.e-10 # prevents nans, but penalty is a bit arbitrary.
            llpws = np.sum(np.log10(pws))
            llsum += llpws
            
            # adds these to list of likelihood outputs
            lllist["ptauw"]["pws"]=llpws
            longlist["ptauw"]["pws"]=np.log10(pws)
        
        # calculates all metrics: (psnr|b,w,z,DM), p(b,w | z,DM), p(w|z,DM), p(b|z,dM), p(w|b,z,DM), p(b|w,z,DM)
        if pwb:
            # each of these is probability calculated at each particular value of z
            # to properly interpret, we should first calculate the probability within each z
            # and at the last, multiply by p(z)
            
            # we have previously go "tomult": which is calculated *only* for ptauw
            # this should be done for all cases, not just ptauw
            
            
            psnrbws = psnrbws.reshape([nb,nw,nz,nfrb]) # holds psnr_gbw * p(b,w,) for each b,w bin
            psnr_gbws = psnr_gbws.reshape([nb,nw,nz,nfrb]) # holds psnr_gbw * p(b,w,) for each b,w bin
            pbws = pbws.reshape([nb,nw,nz,nfrb]) # holds p(bw given z,dm) for each b,w, bin
            
            pw_norm = np.sum(pbws,axis=0) # sums along b axis, giving p(w)
            pb_norm = np.sum(pbws,axis=1) # sums along w axis, giving p(b)
            pwb_norm = np.sum(pw_norm,axis=0) # sums along w axis after b axis, giving pbw norm for all FRBs
            
            psnrbw = np.zeros([nz,nfrb])
            psnr_gbw = np.zeros([nz,nfrb])
            pbw = np.zeros([nz,nfrb])
            pw = np.zeros([nz,nfrb])
            pb = np.zeros([nz,nfrb])
            
            for i,b in enumerate(survey.beam_b):
                for j,w in enumerate(grid.eff_weights):
                    # multiplies by the width and beam weights for that FRB. These are pre-calculated in the survey
                    # each component below is a vector over nfrb
                    psnrbw += psnrbws[i,j,:,:]*survey.frb_bweights[nozlist,i]*survey.frb_wweights[nozlist,j] # multiply last axis
                    psnr_gbw += psnr_gbws[i,j,:,:]*survey.frb_bweights[nozlist,i]*survey.frb_wweights[nozlist,j] # multiply last axis
                    pbw += pbws[i,j,:,:]*survey.frb_bweights[nozlist,i]*survey.frb_wweights[nozlist,j] # multiply last axis for all z
            
            # normalises pbw by normalised sum over all b,w. This gives dual p(b,w) for each FRB
            # pwb_norm is 2D. pbw is 2D. Should work!
            pbw = pbw / pwb_norm
            psnrbw = psnrbw/pwb_norm
            
            # psnr_gbws needs no normalisation, provided weights in each dimension sum to unity. But we check here just to be sure
            # the division is along the last (FRB) axis
            psnr_gbw = psnr_gbw / (np.sum(survey.frb_bweights[nozlist,:],axis=1) * np.sum(survey.frb_wweights[nozlist,:],axis=1))
            psnrbw = psnrbw / (np.sum(survey.frb_bweights[nozlist,:],axis=1) * np.sum(survey.frb_wweights[nozlist,:],axis=1))
            
            
            # calculates p(w) values
            # then normalises probability over all pbw
            for j,w in enumerate(grid.eff_weights):
                pw[:,:] += pw_norm[j,:,:]*survey.frb_wweights[nozlist,j]
            pw = pw/pwb_norm
            
            
            # calculates p(b) values.
            # then normalised probability over all pbw
            for i,b in enumerate(survey.beam_b):
                pb[:,:] += pb_norm[i,:,:]*survey.frb_bweights[nozlist,i]
            pb = pb/pwb_norm
            
            # calculates p(b|w,z,dM), using p(b|w) p(w) = p(b,w)
            pb_gw = pbw / pw
            
            # calculates p(w|b,z,DM), using p(w|b) p(b) = p(b,w)
            pw_gb = pbw / pb
            
            # this is where we sum along the z-axis! (after multiplying by p(z) of course)
            
            pb = np.sum(pb*tomult,axis=0)
            pw = np.sum(pw*tomult,axis=0)
            pb_gw = np.sum(pb_gw*tomult,axis=0)
            pw_gb = np.sum(pw_gb*tomult,axis=0)
            pbw = np.sum(pbw*tomult,axis=0)
            psnr_gbw = np.sum(psnr_gbw*tomult,axis=0)
            
            if PATH:
                # store p(SNR,b,w(z) | DM)
                PATH_OP["psnrbwgzdm"] = psnrbw #tomult is returned separately
            psnrbw = np.sum(psnrbw*tomult,axis=0)
            
            # calcs p(width, beam)
            bad = np.where(pbw == 0.)
            pbw[bad] = 1.e-10
            llpbw = np.sum(np.log10(pbw))
            #llsum += llpbw
            
            # cals psnr values
            bad = np.where(psnr_gbw == 0.)
            psnr_gbw[bad] = 1.e-10
            llpsnr_gbw = np.sum(np.log10(psnr_gbw))
            
            # adds psnrbw values to the list
            bad = np.where(psnrbw == 0.)
            psnrbw[bad] = 1.e-10
            llpsnrbw = np.log10(psnrbw)
            llsum += np.sum(llpsnrbw)
            
            longlist["pbw"]={}
            longlist["pbw"]["pb"]=np.log10(pb)
            longlist["pbw"]["pw"]=np.log10(pw)
            longlist["pbw"]["pbgw"]=np.log10(pb_gw)
            longlist["pbw"]["pwgb"]=np.log10(pw_gb)
            longlist["pbw"]["pbw"]=np.log10(pbw)
            longlist["pbw"]["psnr_gbw"]=np.log10(psnr_gbw)
            longlist["pbw"]["psnrbw"]=np.log10(psnrbw)
            
            lllist["pbw"]={}
            lllist["pbw"]["pb"]=np.sum(np.log10(pb))
            lllist["pbw"]["pw"]=np.sum(np.log10(pw))
            lllist["pbw"]["pbgw"]=np.sum(np.log10(pb_gw))
            lllist["pbw"]["pwgb"]=np.sum(np.log10(pw_gb))
            lllist["pbw"]["pbw"]=np.sum(np.log10(pbw))
            lllist["pbw"]["psnr_gbw"]=np.sum(np.log10(psnr_gbw))
            lllist["pbw"]["psnrbw"]=np.sum(np.log10(psnrbw))
        
        
        # normalise by the beam and FRB width values
        #This ensures that regions with zero probability don't produce nans due to 0/0
        OK = np.where(pbw_norm.flatten() > 0.)
        zpsnr = zpsnr.flatten()
        zpsnr[OK] /= pbw_norm.flatten()[OK]
        zpsnr = zpsnr.reshape([nz,nfrb])
        
        # add simply psnr. Do this before normalisation over z - we want p(snr|z,DM), not p(snr|DM)
        if not pwb and PATH:
            PATH_OP["psnrbwgzdm"] = zpsnr
        
        # perform the weighting over the redshift axis, i.e. to multiply by p(z|DM) and normalise \int p(z|DM) dz = 1
        rnorms = np.sum(rs,axis=0)
        
        psnr = np.sum(zpsnr*rs,axis=0) / rnorms
        
        
        # normalises for total probability of DM occurring in the first place.
        # We need to do this. This effectively cancels however the Emin-Emax factor.
        # sums down the z-axis
        
        # keeps individual FRB values
        longlist["psnr"] = np.log10(psnr)
        
        
        # checks to ensure all frbs have a chance of being detected
        bad=np.array(np.where(psnr == 0.))
        if bad.size > 0:
            snrll = -1e10 # none of this is possible! [somehow...]
        else:
            snrll = np.sum(np.log10(psnr))
        
        # add to likelihood list
        lllist["psnr"] = snrll
        
        if not pwb:
            # only do this if we are not already calculating psnr given p(w,b)
            llsum += snrll
        
    if grid_type==1 and pNreps:
        repll = 0
        if len(survey.replist) != 0:
            for irep in survey.replist:
                pReps = grid.calc_exact_repeater_probability(Nreps=survey.frbs["NREP"][irep],DM=survey.DMs[irep],z=None)
                if pReps == 0:
                    repll += -1e10
                else:
                    repll += np.log10(float(pReps))
        lllist["pReps"]=repll
        longlist["pReps"] = np.log10(np.array(allpReps))
        llsum += repll
    
    # if we're in PATH mode, do this first
    if PATH:
        return PATH_OP
    
    # determines which list of things to return
    if dolist==0:
        return llsum
    elif dolist==1:
        return llsum,lllist
    elif dolist==2:
        return llsum,lllist,longlist
    

def calc_likelihoods_2D(grid,survey,doplot=False,norm=True,psnr=True,printit=False,
                Pn=False,pNreps=True,ptauw=False,pwb=False,dolist=0,verbose=False,grid_type=0):
    """ Calculates 2D likelihoods using observed DM,z values
    
    grid: the grid object calculated from survey
    
    survey: survey object containing the observed z,DM values
    
    doplot: will generate a plot of z,DM values
    
    psnr:
        True: calculate probability of observing each FRB at the observed SNR
        False: do not calculate this

    Pn:
        True: calculate probability of observing N FRBs
        False: do not calculate this

    pNreps:
        True: calculate probability that each repeater detects the given number of bursts
        False: do not calculate this
    
    ptauw:
        True: calculate probability of intrinsic width and scattering *given* total width
        False: do not calculate this
    
    pwb:
        True: calculate probability of specific width and beam values, and psnr | bw
        False: do not calculate this; simply sum psnr over all possible p(b,w)
    
    dolist:
        0: returns total log10 likelihood llsum only [float]
        1: also returns a dict of individual statistical contributions to log-likelihood
        2: also returns the above for every FRB individually
        Structure of llsum and longlist:
            ["pzDM"]["pDM"]
            ["pN"]
            ["Nexpected"] (only for 
            ["ptauw"]
                ["pbar"]
                ["piw"]
                ["ptau"]
                ["w_indices"]
            ["pbw"]
                ["pb"]
                ["pw"]
                ["pbgw"]
                ["pwgb"]
                ["pbw"]
                ["psnr_gbw"]
                ["psnrbw"]
            lllist: <Nfrbs>
            longlist [list of lists]:
                Pn: float
                zDM: PzDM
                zDM extras: P(z) P(DM), P(DM|z), P(z|DM)
                ptauw: p(tau|wtot), p(wtot), p(snr|wtot)
                psnr: p_snr (over all beams/widths)
                pwb: p(snr|b,w,z,dm), p(snr,b,w|z,DM), p(b|zDM), p(w|zDM), p(b|w,zDM), p(w|b,zDM), p(wb|zDM)
        else: returns nothing (actually quite useful behaviour!)
    
    norm:
        True: calculates p(z,DM | FRB detected)
        False: calculates p(detecting an FRB with z,DM). Meaningless unless
            some sensible normalisation has already been applied to the grid.
    
    grid_type:
        0: normal zdm grid
        1: assumes the grid passed is a repeat_grid.zdm_repeat_grid object and calculates likelihood for repeaters
        2: assumes the grid passed is a repeat_grid.zdm_repeat_grid object and calculates likelihood for single bursts
    """

    ######## Calculates p(DM,z | FRB) ########
    # i.e. the probability of a given z,DM assuming
    # an FRB has been observed. The normalisation
    # below is proportional to the total rate (ish)
    
    if ptauw:
        if not survey.backproject:
            print("WARNING: cannot calculate ptauw for this survey, please initialised backproject")
    
    # Determine which array to perform operations on and initialise
    if grid_type == 1:
        rates = grid.exact_reps 
        if survey.zreps is not None:
            DMobs=survey.DMEGs[survey.zreps]
            Zobs=survey.Zs[survey.zreps]
            zlist=survey.zreps
        else:
            raise ValueError("No localised singles in this survey, cannot calculate 1D likelihoods")
    elif grid_type == 2: 
        rates = grid.exact_singles 
        if survey.zsingles is not None:
            DMobs=survey.DMEGs[survey.zsingles]
            Zobs=survey.Zs[survey.zsingles]
            zlist=survey.zsingles
        else:
            raise ValueError("No localised repeaters in this survey, cannot calculate 1D likelihoods")
    else: 
        rates=grid.rates 
        if survey.zlist is not None:
            DMobs=survey.DMEGs[survey.zlist]
            Zobs=survey.Zs[survey.zlist]
            zlist=survey.zlist
        else:
            raise ValueError("No nlocalised FRBs in this survey, cannot calculate 1D likelihoods")
    zvals=grid.zvals
    dmvals=grid.dmvals
    
    # normalise to total probability of 1
    if norm:
        norm=np.sum(rates) # gets multiplied by event size later
    else:
        norm=1.
    
    # in the grid, each z and dm value represents the centre of a bin, with p(z,DM)
    # giving the probability of finding the FRB in the range z +- dz/2, dm +- dm/2.
    # threshold for when we shift from lower to upper is if z < zcentral,
    # weight slowly shifts from lower to upper bin
    
    idms1,idms2,dkdms1,dkdms2 = grid.get_dm_coeffs(DMobs)
    izs1,izs2,dkzs1,dkzs2 = grid.get_z_coeffs(Zobs)
    
    ############## Calculate probability p(z,DM) ################
    if grid.state.MW.sigmaDMG == 0.0 and grid.state.MW.sigmaHalo == 0.0:
        if np.any(DMobs < 0):
            raise ValueError("Negative DMobs with no uncertainty")

        # Linear interpolation
        pvals = rates[izs1,idms1]*dkdms1*dkzs1
        pvals += rates[izs2,idms1]*dkdms1*dkzs2
        pvals += rates[izs1,idms2]*dkdms2*dkzs1
        pvals += rates[izs2,idms2]*dkdms2*dkzs2
    else:
        dm_weights, iweights = calc_DMG_weights(DMobs, survey.DMhalos[zlist], survey.DMGs[zlist], dmvals, grid.state.MW.sigmaDMG, 
                                                grid.state.MW.sigmaHalo, grid.state.MW.logu)
        pvals = np.zeros(len(izs1))
        for i in range(len(izs1)):
            pvals[i] = np.sum(rates[izs1[i],iweights[i]] * dm_weights[i] * dkzs1[i] 
                              + rates[izs2[i],iweights[i]] * dm_weights[i] * dkzs2[i])
    
    bad = pvals <= 0.
    flg_bad = False
    if np.any(bad):
        # This avoids a divide by 0 but we are in a NAN regime
        pvals[bad]=1e-50 # hopefully small but not infinitely so
        flg_bad = True
    
    # initialise dicts to return detailed log-likelihood information
    longlist={}
    lllist={}
    
    # holds individual FRB data
    # records p(z,DM) for each FRB. Analagous to lllist, but for each FRB
    longlist["pzDM"]={}
    longlist["pzDM"]["pzDM"]=np.log10(pvals/norm)
    
    # llsum is the total log-likelihood for the entire set
    llsum=np.sum(np.log10(pvals))
    if flg_bad:
        llsum = -1e10
    llsum -= np.log10(norm)*Zobs.size # once per event
    
    # creates a list of all z,DM results
    lllist["pzDM"]={} # pz,DM
    lllist["pzDM"]["pzDM"]=llsum
    
    #### calculates zdm components p(DM),p(z|DM),p(z),p(DM|z)
    # does this by using previous results for p(z,DM) and
    # calculating p(DM) and p(z)
    if dolist > 0:
        # calculates p(dm)
        pdmvals = np.sum(rates[:,idms1],axis=0)*dkdms1
        pdmvals += np.sum(rates[:,idms2],axis=0)*dkdms2
        
        # implicit calculation of p(z|DM) from p(z,DM)/p(DM)
        #neither on the RHS is normalised so this is OK!
        pzgdmvals = pvals/pdmvals
        
        #calculates p(z)
        pzvals = np.sum(rates[izs1,:],axis=1)*dkzs1
        pzvals += np.sum(rates[izs2,:],axis=1)*dkzs2
        
        # implicit calculation of p(z|DM) from p(z,DM)/p(DM)
        pdmgzvals = pvals/pzvals
        
        for array in pdmvals,pzgdmvals,pzvals,pdmgzvals:
            bad=np.array(np.where(array <= 0.))
            if bad.size > 0:
                array[bad]=1e-20 # hopefully small but not infinitely so
        
        llnorm = np.log10(norm)
        
        # logspace and normalisation
        llpzgdm = np.sum(np.log10(pzgdmvals))
        llpdmgz = np.sum(np.log10(pdmgzvals))
        llpdm = np.sum(np.log10(pdmvals)) - llnorm*Zobs.size
        llpz = np.sum(np.log10(pzvals)) - llnorm*Zobs.size
        
        # adds survey totals to log-likelihood list
        lllist["pzDM"]["pzgdm"] = llpzgdm
        lllist["pzDM"]["pdmgz"] = llpdmgz
        lllist["pzDM"]["pdm"] = llpdm
        lllist["pzDM"]["pz"] = llpz
        
        # adds individual FRB data to long list
        longlist["pzDM"]["pzgdm"] = np.log10(pzgdmvals)
        longlist["pzDM"]["pdmgz"] = np.log10(pdmgzvals)
        longlist["pzDM"]["pdm"] = np.log10(pdmvals) - llnorm
        longlist["pzDM"]["pz"] = np.log10(pzvals) - llnorm
    
    
    ############### Calculate p(N) ###############3
    if Pn and (survey.TOBS is not None):
        if grid_type == 1:
            observed=survey.NORM_REPS
            C = grid.Rc
            reps=True
        elif grid_type == 2:
            observed=survey.NORM_SINGLES
            C = grid.Rc
            reps=True
        else:
            observed=survey.NORM_FRB
            C = 10**grid.state.FRBdemo.lC
            reps=False
        expected=CalculateIntegral(rates,survey,reps)
        expected *= C
        
        Pn=Poisson_p(observed,expected)
        if Pn==0:
            Pll=-1e10
        else:
            Pll=np.log10(Pn)
        lllist["pN"]=Pll
        lllist["Nexpected"]=expected
        if verbose:
            print(f'Pll term = {Pll}')
        llsum += Pll
    else:
        # dummy values
        lllist["Nexpected"]=-1
        lllist["pN"]=0
    
    ################ Calculates p(tau,w| total width) ###############
    if ptauw:
        if not survey.backproject:
            raise ValueError("Cannot estimate p(tau,w) without survey.backproject being True!")
        
        # checks which have OK tau values - in general, this is a subset
        # ALSO: note that this only checks p(tau,iw | w)! It does NOT
        # evaluate p(w)!!! Which is a pretty key thing...
        ztaulist = []
        iztaulist = []
        for i,iz in enumerate(zlist):
            if iz in survey.OKTAU:
                ztaulist.append(iz) # for direct indexing of survey
                iztaulist.append(i) # for getting a subset of zlist
        Wobs = survey.WIDTHs[ztaulist]
        Tauobs = survey.TAUs[ztaulist]
        Iwobs = survey.IWIDTHs[ztaulist]
        ztDMobs=survey.DMEGs[ztaulist]
        ztZobs=survey.Zs[ztaulist]
        
        # This could all be precalculated within the survey.
        iws1,iws2,dkws1,dkws2 = survey.get_w_coeffs(Wobs) # total width in survey width bins
        itaus1,itaus2,dktaus1,dktaus2 = survey.get_internal_coeffs(Tauobs) # scattering time tau
        iis1,iis2,dkis1,dkis2 = survey.get_internal_coeffs(Iwobs) # intrinsic width
        
        #ztidms1,ztidms2,ztdkdms1,ztdkdms2 = grid.get_dm_coeffs(ztDMobs)
        ztizs1,ztizs2,ztdkzs1,ztdkzs2 = grid.get_z_coeffs(ztZobs)
        
        piws = survey.pws[ztizs1,iis1,iws1]*ztdkzs1*dkis1*dkws1 \
            + survey.pws[ztizs1,iis1,iws2]*ztdkzs1*dkis1*dkws2 \
            + survey.pws[ztizs1,iis2,iws1]*ztdkzs1*dkis1*dkws1 \
            + survey.pws[ztizs1,iis2,iws2]*ztdkzs1*dkis1*dkws2 \
            + survey.pws[ztizs2,iis1,iws1]*ztdkzs2*dkis1*dkws1 \
            + survey.pws[ztizs2,iis1,iws2]*ztdkzs2*dkis1*dkws2 \
            + survey.pws[ztizs2,iis2,iws1]*ztdkzs2*dkis2*dkws1 \
            + survey.pws[ztizs2,iis2,iws2]*ztdkzs2*dkis2*dkws2
        
        ptaus = survey.ptaus[ztizs1,itaus1,iws1]*ztdkzs1*dktaus1*dkws1 \
            + survey.ptaus[ztizs1,itaus1,iws2]*ztdkzs1*dktaus1*dkws2 \
            + survey.ptaus[ztizs1,itaus2,iws1]*ztdkzs1*dktaus1*dkws1 \
            + survey.ptaus[ztizs1,itaus2,iws2]*ztdkzs1*dktaus1*dkws2 \
            + survey.ptaus[ztizs2,itaus1,iws1]*ztdkzs2*dktaus1*dkws1 \
            + survey.ptaus[ztizs2,itaus1,iws2]*ztdkzs2*dktaus1*dkws2 \
            + survey.ptaus[ztizs2,itaus2,iws1]*ztdkzs2*dktaus2*dkws1 \
            + survey.ptaus[ztizs2,itaus2,iws2]*ztdkzs2*dktaus2*dkws2
        
        # safegaudr zero probabilities
        bad1 = np.where(piws==0)[0]
        bad2 = np.where(ptaus==0)[0]
        piws[bad1] = 1e-10
        ptaus[bad2] = 1e-10
        pbars = 0.5*piws + 0.5*ptaus
        
        llpbar = np.sum(np.log10(pbars))
        llpiw = np.sum(np.log10(piws))
        llptw = np.sum(np.log10(ptaus))
        # while we calculate llpiw, we don't add it to the sum
        # this is because w and tau are not independent!
        
        llsum += llpbar # now summing in linear space
        
        lllist["ptauw"]={}
        # appending total of each to log0-likelihood list
        lllist["ptauw"]["piw"]=llpiw
        lllist["ptauw"]["ptw"]=llptw
        lllist["ptauw"]["pbar"]=llpbar
        
        # appending individual FRB data to long long list
        longlist["ptauw"]={}
        longlist["ptauw"]["pbar"]=np.log10(pbars)
        longlist["ptauw"]["piw"]=np.log10(piws)
        longlist["ptauw"]["ptau"]=np.log10(ptaus)
        longlist["ptauw"]["w_indices"]=iztaulist
        
    
    ############ Calculates p(s | z,DM) #############
    # i.e. the probability of observing an FRB
    # with energy E given redshift and DM
    # this calculation ignores beam values
    # this is the derivative of the cumulative distribution
    # function from Eth to Emax
    # this does NOT account for the probability of
    # observing something at a relative sensitivty of b, i.e. assumes you do NOT know localisation in your beam...
    # to do that, one would calculate this for the exact value of b for that event. The detection
    # probability has already been integrated over the full beam pattern, so it would be trivial to
    # calculate this in one go. Or in other words, one could simple add in survey.Bs, representing
    # the local sensitivity to the event [keeping in mind that Eths has already been calculated
    # taking into account the burst width and DM, albeit for a mean FRB]
    # Note this would be even simpler than the procedure described here - we just
    # use b! Huzzah! (for the beam)
    # IF:
    # - we want to make FRB width analogous to beam, THEN
    # - we need an analogous 'beam' (i.e. width) distribution to integrate over,
    #     which gives the normalisation
    
    if psnr:
        # NOTE: to break this into a p(SNR|b) p(b) term, we first take
        # the relative likelihood of the threshold b value compare
        # to the entire lot, and then we calculate the local
        # psnr for that beam only. But this requires a much more
        # refined view of 'b', rather than the crude standard 
        # parameterisation

        # calculate vector of grid thresholds
        Emax=10**grid.state.energy.lEmax
        Emin=10**grid.state.energy.lEmin
        gamma=grid.state.energy.gamma

        # Evaluate thresholds at the exact DMobs
        # The thresholds have already been calculated at mean values
        # of the below quantities. Hence, we use the DM relative to
        # those means, not the actual DMEG for that FRB
        DMEGmeans = survey.DMs[zlist] - np.median(survey.DMGs + survey.DMhalos)
        idmobs1,idmobs2,dkdmobs1,dkdmobs2 = grid.get_dm_coeffs(DMEGmeans)
        
        # Linear interpolation
        Eths = grid.thresholds[:,izs1,idmobs1]*dkdmobs1*dkzs1
        Eths += grid.thresholds[:,izs2,idmobs1]*dkdmobs1*dkzs2
        Eths += grid.thresholds[:,izs1,idmobs2]*dkdmobs2*dkzs1
        Eths += grid.thresholds[:,izs2,idmobs2]*dkdmobs2*dkzs2
        
        FtoE = grid.FtoE[izs1]*dkzs1
        FtoE += grid.FtoE[izs2]*dkzs2
        
        # now do this in one go
        # We integrate p(snr|b,w) p(b,w) db dw.
        # Eths.shape[i] is the number of FRBs: length of izs1
        # this has shape nz,nFRB - FRBs could come from any z-value
        # Note: given that this includes p(b,w), we can use this loop
        # to simultaneously calculate p(b,w)
        nb = survey.beam_b.size
        nw,nfrb = Eths.shape
        psnr=np.zeros([nfrb])
        
        if grid.eff_weights.ndim ==2:
            zwidths = True
            usews = np.zeros([nfrb])
        else:
            zwidths = False
        
        # initialised to hold w-b normalisations
        pbw_norm = 0.
        
        if ptauw and not pwb:
            # hold array representing p(w) and p(b)
            dpbws = np.zeros([nw,nfrb]) # holds pw over the width only, i.e. summing over the beam
            
        if pwb:
            psnrbws = np.zeros([nb,nw,nfrb]) # holds psnr_gbw * p(b,w,) for each b,w bin
            psnr_gbws = np.zeros([nb,nw,nfrb]) # holds psnr_gbw * p(b,w,) for each b,w bin
            pbws = np.zeros([nb,nw,nfrb]) # holds p(bw given z,dm) for each b,w, bin
            
        for i,b in enumerate(survey.beam_b):
            bEths=Eths/b # array of shape NFRB, 1/b
            bEobs=bEths*survey.Ss[zlist]
            
            for j,w in enumerate(grid.eff_weights):
                # probability of observing an FRB at this z,DM with given b,w at this particular snr dsnr
                temp=grid.array_diff_lf(bEobs[j,:],Emin,Emax,gamma) # * FtoE #one dim in beamshape, one dim in FRB
                differential = temp.T*bEths[j,:] #multiplies by beam factors and weight
                
                # probability of observing an FRB at this z,DM with given b,w at *any* snr
                temp2=grid.array_cum_lf(bEths[j,:],Emin,Emax,gamma) # * FtoE #one dim in beamshape, one dim in FRB
                cumulative = temp2.T #*bEths[j,:] #multiplies by beam factors and weight
                
                
                if zwidths:
                    # a function of redshift
                    usew = w[izs1]*dkzs1 + w[izs2]*dkzs2
                    usews += usew
                    usew = usew
                else:
                    usew = w # just a scalar quantity
                
                # the product here is p(SNR|DM,z) = p(SNR|b,w,DM,z) * p(b,w|DM,z)
                # p(SNR|b,w,DM,z) = differential/cumulative
                # p(b,w|DM,z) = survey.beam_o[i]*usew * cumulative / sum(survey.beam_o[i]*usew * cumulative)
                # hence, the "cumulative" part cancels
                
                # this value normalises the pbw_gdmz value
                dpbw = survey.beam_o[i]*usew*cumulative
                
                if ptauw and not pwb:
                    # record probability of this w summed over all beams for each FRB
                    dpbws[i,j,:] += dpbw
                
                pbw_norm += dpbw
                
                # this is the psnr_gbw * pbw_gdmz contribution for this particular b,w. The "cumulative" value cancels
                psnr += differential*survey.beam_o[i]*usew
                
                ###### Breaks p(snr,b,w) into three components, and saves them #####
                # this allows computations of psnr given b and w values, collapsing these over the dimensions of b and w
                
                if pwb:
                    # psnr given beam, width, z,dm
                    OK = np.where(cumulative > 0)[0]
                    if zwidths:
                        usew = usew[OK]
                    
                    psnr_gbws[i,j,OK] = differential[OK]/cumulative[OK]
                    
                    # psnr given beam, width, z,dm. if differential is OK, cool!
                    psnrbws[i,j,OK] = differential[OK]*survey.beam_o[i]*usew
                    
                    # total probability of that p(w,b)
                    pbws[i,j,OK] = survey.beam_o[i]*usew*cumulative[OK]
                    
        
        # calculate p(w)
        # Note that iws1 and iws2 is only defined for ztaulist
        # this leaves info on the table for FRBs with no tau but known total width
        if ptauw and not pwb:
            # normalise over all w values
            dpbws /= np.sum(dpbws,axis=0)
            # calculate pws
            pws = dpbws[iws1,iztaulist]*dkws1 + dpbws[iws2,iztaulist]*dkws2
            bad = np.where(pws == 0.)[0]
            pws[bad] = 1.e-10 # prevents nans, but 
            
            llpws = np.sum(np.log10(pws))
            llsum += llpws
            
            # adds these to list of likelihood outputs
            lllist["ptauw"]["pws"]=llpws
            longlist["ptauw"]["pws"]=np.log10(pws)
        
        # calculates all metrics: (psnr|b,w,z,DM), p(b,w | z,DM), p(w|z,DM), p(b|z,dM), p(w|b,z,DM), p(b|w,z,DM)
        if pwb:
            pw_norm = np.sum(pbws,axis=0) # sums along b axis, giving p(w)
            pb_norm = np.sum(pbws,axis=1) # sums along w axis, giving p(b)
            pwb_norm = np.sum(pw_norm,axis=0) # sums along w axis after b axis, giving pbw norm for all FRBs
            
            psnrbw = np.zeros([nfrb])
            psnr_gbw = np.zeros([nfrb])
            pbw = np.zeros([nfrb])
            pw = np.zeros([nfrb])
            pb = np.zeros([nfrb])
            
            for i,b in enumerate(survey.beam_b):
                for j,w in enumerate(grid.eff_weights):
                    # multiplies by the width and beam weights for that FRB. These are pre-calculated in the survey
                    # each component below is a vector over nfrb
                    
                    psnrbw += psnrbws[i,j,:]*survey.frb_zbweights[:,i]*survey.frb_zwweights[:,j]
                    psnr_gbw += psnr_gbws[i,j,:] *survey.frb_zbweights[:,i]*survey.frb_zwweights[:,j]
                    pbw += pbws[i,j,:]*survey.frb_zbweights[:,i]*survey.frb_zwweights[:,j]
                    
            
            # normalises pbw by normalised sum over all b,w. This gives dual p(b,w) for each FRB
            pbw = pbw / pwb_norm
            psnrbw = psnrbw / pwb_norm
            
            # psnr_gbws needs no normalisation, provided weights in each dimension sum to unity. But we check here just to be sure
            psnr_gbw = psnr_gbw / (np.sum(survey.frb_zbweights,axis=1) * np.sum(survey.frb_zwweights,axis=1))
            psnrbw = psnrbw / (np.sum(survey.frb_zbweights,axis=1) * np.sum(survey.frb_zwweights,axis=1))
            
            # calculates p(w) values
            # then normalises probability over all pbw
            for j,w in enumerate(grid.eff_weights):
                pw[:] += pw_norm[j,:]*survey.frb_zwweights[:,j]
            pw = pw/pwb_norm
            
            # calculates p(b) values.
            # then normalised probability over all pbw
            for i,b in enumerate(survey.beam_b):
                pb[:] += pb_norm[i,:]*survey.frb_zbweights[:,i]
            pb = pb/pwb_norm
            
            # calculates p(b|w,z,dM), using p(b|w) p(w) = p(b,w)
            pb_gw = pbw / pw
            
            # calculates p(w|b,z,DM), using p(w|b) p(b) = p(b,w)
            pw_gb = pbw / pb
            
            # adds p(widht, beam) to the list
            bad = np.where(pbw == 0.)
            pbw[bad] = 1.e-10
            llpbw = np.sum(np.log10(pbw))
            llsum += llpbw
            
            # adds psnr values to the list
            bad = np.where(psnr_gbw == 0.)
            psnr_gbw[bad] = 1.e-10
            llpsnr_gbw = np.sum(np.log10(psnr_gbw))
            llsum += llpsnr_gbw
            
            longlist["pbw"]={}
            longlist["pbw"]["pb"]=np.log10(pb)
            longlist["pbw"]["pw"]=np.log10(pw)
            longlist["pbw"]["pbgw"]=np.log10(pb_gw)
            longlist["pbw"]["pwgb"]=np.log10(pw_gb)
            longlist["pbw"]["pbw"]=np.log10(pbw)
            longlist["pbw"]["psnr_gbw"]=np.log10(psnr_gbw)
            longlist["pbw"]["psnrbw"]=np.log10(psnrbw)
            
            lllist["pbw"]={}
            lllist["pbw"]["pb"]=np.sum(np.log10(pb))
            lllist["pbw"]["pw"]=np.sum(np.log10(pw))
            lllist["pbw"]["pbgw"]=np.sum(np.log10(pb_gw))
            lllist["pbw"]["pwgb"]=np.sum(np.log10(pw_gb))
            lllist["pbw"]["pbw"]=np.sum(np.log10(pbw))
            lllist["pbw"]["psnr_gbw"]=np.sum(np.log10(psnr_gbw))
            lllist["pbw"]["psnrbw"]=np.sum(np.log10(psnrbw))
            
        OK = np.where(pbw_norm > 0.)[0]
        psnr[OK] /= pbw_norm[OK]
        
        # checks to ensure all frbs have a chance of being detected
        bad=np.array(np.where(psnr == 0.))
        if bad.size > 0:
            snrll = -1e10 # none of this is possible! [somehow...]
        else:
            snrll = np.sum(np.log10(psnr))
        
        
        # keeps individual FRB values
        longlist["psnr"] = np.log10(psnr)
        
        # add to likelihood list
        lllist["psnr"] = snrll
        
        if not pwb:
            # only do this if we are not already calculating psnr given p(w,b)
            llsum += snrll
    
    if grid_type==1 and pNreps:
        repll = 0
        allpReps=[]
        if len(survey.replist) != 0:
            for irep in survey.replist:
                pReps = grid.calc_exact_repeater_probability(Nreps=survey.frbs["NREP"][irep],DM=survey.DMs[irep],z=survey.Zs[irep])
                allpReps.append(float(pReps))
                repll += np.log10(float(pReps))
        lllist["pReps"]=repll
        llsum += repll
        longlist["pReps"] = np.log10(np.array(allpReps))
    
    if verbose:
        print(f"rates={np.sum(rates):0.5f}," \
            f"nterm={-np.log10(norm)*Zobs.size:0.2f}," \
            f"pvterm={np.sum(np.log10(pvals)):0.2f}," \
            f"wzterm={np.sum(np.log10(psnr)):0.2f}," \
            f"comb={np.sum(np.log10(psnr*pvals)):0.2f}")
    
    # determines which list of things to return
    if dolist==0:
        return llsum
    elif dolist==1:
        return llsum,lllist
    elif dolist==2:
        return llsum,lllist,longlist

def calc_DMG_weights(DMEGs, DMhalos, DM_ISMs, dmvals, sigma_ISM=0.5, sigma_halo_abs=15.0, log=False):
    """
    Given an uncertainty on the DMG value, calculate the weights of DM values to integrate over

    Inputs:
        DMEGs       =   Extragalactic DMs
        DMhalo      =   Assumed constant (average) DMhalo
        DM_ISMs     =   Array of each DM_ISM value
        dmvals      =   Vector of DM values used
        sigma_ISM   =   Fractional uncertainty in DMG values
        sigma_halo  =   Uncertainty in DMhalo value (in pc/cm3)

    Returns:
        weights     =   Relative weights for each of the DM grid points
        iweights    =   Indices of the corresponding weights
    """
    weights = []
    iweights = []

    # Loop through the DMG of each FRB in the survey and determine the weights
    for i,DM_ISM in enumerate(DM_ISMs):
        # Determine lower and upper DM values used
        # From 0 to DM_total
        DM_total = DMEGs[i] + DM_ISM + DMhalos[i]

        idxs = np.where(dmvals < DM_total)

        # Get weights
        DMGvals = DM_total - dmvals[idxs] # Descending order because dmvals are ascending order
        ddm = dmvals[1] - dmvals[0]

        # Get absolute uncertainty in DM_ISM
        sigma_ISM_abs = DM_ISM * sigma_ISM

        # pISM
        if sigma_ISM_abs == 0.0:
            pISM = None
        elif log:
            pISM = st.lognorm.pdf(DMGvals, scale=DM_ISM, s=sigma_ISM) * ddm
        else:
            pISM = st.norm.pdf(DMGvals, loc=DM_ISM, scale=sigma_ISM_abs) * ddm
    
        # pHalo
        if sigma_halo_abs == 0.0:
            pDMG = None
        elif log:
            sigma_halo = sigma_halo_abs / DMhalos[i]
            pHalo = st.lognorm.pdf(DMGvals, scale=DMhalos[i], s=sigma_halo) * ddm
        else:
            pHalo = st.norm.pdf(DMGvals, loc=DMhalos[i], scale=sigma_halo_abs) * ddm
        
        if pISM is None:
            pDMG = pHalo 
        elif pHalo is None:
            pDMG = pISM
        else:
            pDMG = np.convolve(pISM, pHalo, mode='full')

        # Set upper limit of DMG = DM_total 
        # Reversed because DMGvals are descending order which corresponds to DMEGvals (dmvals) in ascending order
        pDMG = pDMG[-len(DMGvals):] 

        weights.append(pDMG)
        iweights.append(idxs)

    return weights, iweights
 
def CalculateMeaningfulConstant(pset,grid,survey,newC=False):
    """ Gets the flux constant, and quotes it above some energy minimum Emin """
    
    # Units: IF TOBS were in yr, it would be smaller, and raw const greater.
    # also converts per Mpcs into per Gpc3
    units=1e9*365.25
    if newC:
        rawconst=CalculateConstant(grid,survey) #required to convert the grid norm to Nobs
    else:
        rawconst=10**pset[7]
    const = rawconst*units # to cubic Gpc and days to year
    Eref=1e40 #erg per Hz
    Emin=10**pset[0]
    gamma=pset[3]
    factor=(Eref/Emin)**gamma
    const *= factor
    return const

def ConvertToMeaningfulConstant(state,Eref=1e39):
    """ Gets the flux constant, and quotes it above some energy minimum Emin """
    
    # Units: IF TOBS were in yr, it would be smaller, and raw const greater.
    # also converts per Mpcs into per Gpc3
    units=1e9*365.25
    
    const = (10**state.FRBdemo.lC)*units # to cubic Gpc and days to year
    #Eref=1e39 #erg per Hz
    Emin=10**state.energy.lEmin
    Emax=10**state.energy.lEmax
    gamma=state.energy.gamma
    if state.energy.luminosity_function == 0:
        factor=(Eref/Emin)**gamma - (Emax/Emin)**gamma
    else:
        from zdm import energetics
        factor = energetics.vector_cum_gamma(np.array([Eref]),Emin,Emax,gamma)
    const *= factor
    return const

def Poisson_p(observed, expected):
    """ returns the Poisson likelihood """
    p=poisson.pmf(observed,expected)
    return p

def CalculateConstant(grid,survey):
    """ Calculates the best-fitting constant for the total
    number of FRBs. Units are:
        - grid volume units of 'per Mpc^3',
        - survey TOBS of 'days',
        - beam units of 'steradians'
        - flux for FRBs with E > Emin
    Hence the constant is 'Rate (FRB > Emin) Mpc^-3 day^-1 sr^-1'
    This should be scaled to be above some sensible value of Emin
    or otherwise made relevant.
    
    """
    
    expected=CalculateIntegral(grid.rates,survey,reps=False)
    observed=survey.NORM_FRB
    constant=observed/expected
    return constant

def CalculateIntegral(rates,survey,reps=False):
    """
    Calculates the total expected number of FRBs for that rate array and survey
    
    This does NOT include the aboslute number of FRBs (through the log-constant)
    """
    
    # check that the survey has a defined observation time
    if survey.TOBS is not None:
        if reps:
            TOBS=1 # already taken into account
        else:
            TOBS=survey.TOBS
    else:
        return 0
    
    if survey.max_dm is not None:
        idxs = np.where(survey.dmvals < survey.max_dm)
    else:
        idxs = None

    total=np.sum(rates[:,idxs])
    return total*TOBS
    
def GetFirstConstantEstimate(grids,surveys,pset):
    ''' simple 1D minimisation of the constant '''
    # ensure the grids are uo-to-date
    for i,g in enumerate(grids):
        update_grid(g,pset,surveys[i])
    
    NPARAMS=8
    # use my minimise in a single parameter
    disable=np.arange(NPARAMS-1)
    C_ll,C_p=my_minimise(pset,grids,surveys,disable=disable,psnr=False,PenTypes=None,PenParams=None)
    newC=C_p[-1]
    print("Calculating C_ll as ",C_ll,C_p)
    return newC


def minus_poisson_ps(log10C,data):
    rs=data[0,:]
    os=data[1,:]
    rsp = rs*10**log10C
    lp=0
    for i,r in enumerate(rsp):
        Pn=Poisson_p(os[i],r)
        if (Pn == 0):
            lp = -1e10
        else:
            lp += np.log10(Pn)
    return -lp
    

def minimise_const_only(vparams:dict,grids:list,surveys:list,
                        Verbose=False, use_prev_grid:bool=True, update=False):
    """
    Only minimises for the constant, but returns the full likelihood
    It treats the rest as constants
    the grids must be initialised at the currect values for pset already

    Args:
        vparams (dict): Parameter dict. Can be None if nothing has varied.
        grids (list): List of grids
        surveys (list): List of surveys
            A bit superfluous as these are in the grids..
        Verbose (bool, optional): [description]. Defaults to True.
        use_prev_grid (bool, optional): 
            If True, make use of the previous grid when 
            looping over them. Defaults to True.

    Raises:
        ValueError: [description]
        ValueError: [description]

    Returns:
        tuple: newC,llC,lltot
    """

    '''
    '''
    
    # specifies which set of parameters to pass to the dmx function
    
    if isinstance(grids,list):
        if not isinstance(surveys,list):
            raise ValueError("Grid is a list, survey is not...")
        ng=len(grids)
        ns=len(surveys)
        if ng != ns:
            raise ValueError("Number of grids and surveys not equal.")
    else:
        ng=1
        ns=1
    
    # calculates likelihoods while ignoring the constant term
    rs=[] #expected
    os=[] #observed
    lls=np.zeros([ng])
    dC=0
    for j,s in enumerate(surveys):
        # Update - but only if there is something to update!
        if vparams is not None:
            grids[j].update(vparams, 
                        prev_grid=grids[j-1] if (
                            j > 0 and use_prev_grid) else None)
        ### Assesses total number of FRBs ###
        if s.TOBS is not None:
            # If we include repeaters, then total number of FRB progenitors = number of repeater progenitors + number of single burst progenitors
            if isinstance(grids[j], zdm_repeat_grid.repeat_Grid):
                r1= CalculateIntegral(grids[j].exact_singles, s,reps=True)
                r2= CalculateIntegral(grids[j].exact_reps, s,reps=True)
                r= r1 + r2
                r*=grids[j].Rc
            # If we do not include repeaters, then we just integrate rates
            else:
                r=CalculateIntegral(grids[j].rates, s, reps=False)
                r*=10**grids[j].state.FRBdemo.lC #vparams['lC']

            o=s.NORM_FRB
            rs.append(r)
            os.append(o)

    # Check it is not an empty survey. We allow empty surveys as a 
    # non-detection still gives information on the FRB event rate.
    if len(rs) != 0:
        data=np.array([rs,os])
        ratios=np.log10(data[1,:]/data[0,:])
        bounds=(np.min(ratios),np.max(ratios))
        startlog10C=(bounds[0]+bounds[1])/2.
        bounds=[bounds]
        t0=time.process_time()
        # If only 1 survey, the answer is trivial
        if len(surveys) == 1:
            dC = startlog10C
        else:
            result=minimize(minus_poisson_ps,startlog10C,
                        args=data,bounds=bounds)
            dC=result.x
        t1=time.process_time()
        
        # constant needs to include the starting value of .lC
        newC = grids[j].state.FRBdemo.lC + float(dC)
        # likelihood is calculated  *relative* to the starting value
        llC=-minus_poisson_ps(dC,data)
    else:
        newC = grids[j].state.FRBdemo.lC
        llC = 0.0

    if update:
        for g in grids:
            g.state.FRBdemo.lC = newC

            if isinstance(g, zdm_repeat_grid.repeat_Grid):
                g.state.rep.RC *= 10**float(dC)
                g.Rc = g.state.rep.RC

    return newC,llC

def parse_input_dict(input_dict:dict):
    """ Method to parse the input dict for generating a cube
    It is split up into its various pieces

    Args:
        input_dict (dict): [description]

    Returns:
        tuple: dicts (can be empty):  state, cube, input
        
    This is almost deprecated, but not quite!
    """
    state_dict, cube_dict = {}, {}
    # 
    if 'state' in input_dict.keys():
        state_dict = input_dict.pop('state')
    if 'cube' in input_dict.keys():
        cube_dict = input_dict.pop('cube')
    # Return 
    return state_dict, cube_dict, input_dict
