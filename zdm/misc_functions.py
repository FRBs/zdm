import os
import sys
import numpy as np
import argparse
import pickle
import json
import copy
from numpy import mean

import scipy as sp

import matplotlib.pyplot as plt
import matplotlib
import cmasher as cmr

from frb import dlas
from frb.dm import igm
from frb.dm import cosmic

import time
from zdm import iteration as it
from zdm import beams
from zdm import cosmology as cos
from zdm import survey
from zdm import grid as zdm_grid
from zdm import repeat_grid as zdm_repeat_grid
from zdm import pcosmic
from zdm import parameters


def j2000_to_galactic(ra_deg, dec_deg):
    """
    Convert Galactic coordinates to Equatorial J2000 coordinates.

    Parameters:
    l_deg (float): Galactic longitude in degrees
    b_deg (float): Galactic latitude in degrees

    Returns:
    tuple: Right Ascension and Declination in degrees (RA, Dec)
    
    # this code written by ChatGPT
    """
    
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    
    # Create a SkyCoord object in ICRS coordinates
    icrs_coord = SkyCoord(ra = ra_deg * u.degree, dec = dec_deg * u.degree, frame='icrs')
    
    # Convert to ICRS frame (J2000 equatorial coordinates)
    galactic_coord = icrs_coord.galactic

    # Return RA and Dec in degrees
    return galactic_coord.b.degree, galactic_coord.l.degree


def galactic_to_j2000(l_deg, b_deg):
    """
    Convert J2000 Equatorial coordinates to Galactic coordinates.

    Parameters:
    ra_deg (float): Right Ascension in degrees
    dec_deg (float): Declination in degrees

    Returns:
    tuple: Galactic longitude and latitude in degrees (l, b)
    
    # this code written by ChatGPT
    """
    
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    
    # Create a SkyCoord object in Galactic coordinates
    galactic_coord = SkyCoord(l=l_deg * u.degree, b=b_deg * u.degree, frame='galactic')
    
    # Convert to ICRS frame (J2000 equatorial coordinates)
    equatorial_coord = galactic_coord.icrs

    # Return RA and Dec in degrees
    return equatorial_coord.ra.degree, equatorial_coord.dec.degree


def get_source_counts(grid, plot=None, Slabel=None):
    """
    Calculates the source-counts function for a given grid
    It does this in terms of p(SNR)dSNR
    
    WARNING: this function may not currently work, but
    is kept here as an example "how-to"
    """
    # this is closely related to the likelihood for observing a given psnr!

    # calculate vector of grid thresholds
    Emax = grid.Emax
    Emin = grid.Emin
    gamma = grid.gamma

    nsnr = 71
    snrmin = 0.001
    snrmax = 1000.0
    ndm = grid.dmvals.size
    snrs = np.logspace(0, 2, nsnr)  # histogram array of values for s=SNR/SNR_th

    # holds cumulative and differential source counts
    cpsnrs = np.zeros([nsnr])
    psnrs = np.zeros([nsnr - 1])

    # holds DM-dependent source counts
    dmcpsnrs = np.zeros([nsnr, ndm])
    dmpsnrs = np.zeros([nsnr - 1, ndm])

    backup1 = np.copy(grid.thresholds)
    Emin = grid.Emin
    Emax = grid.Emax
    gamma = grid.gamma

    # modifies grid to simplify beamshape
    grid.beam_b = np.array([grid.beam_b[-1]])
    grid.beam_o = np.array([grid.beam_o[-1]])
    grid.b_fractions = None

    for i, s in enumerate(snrs):

        grid.thresholds = backup1 * s
        grid.calc_pdv(Emin, Emax, gamma)
        grid.calc_rates()
        rates = grid.rates
        dmcpsnrs[i, :] = np.sum(rates, axis=0)
        cpsnrs[i] = np.sum(dmcpsnrs[i, :])

    # the last one contains cumulative values
    for i, s in enumerate(snrs):
        if i == 0:
            continue
        psnrs[i - 1] = cpsnrs[i - 1] - cpsnrs[i]
        dmpsnrs[i - 1, :] = dmcpsnrs[i - 1, :] - dmcpsnrs[i, :]

    mod = 1.5
    snrs = snrs[:-1]
    imid = int((nsnr + 1) / 2)
    xmid = snrs[imid]
    ymid = psnrs[imid]
    slopes = np.linspace(1.3, 1.7, 5)
    ys = []
    for i, s in enumerate(slopes):
        ys.append(ymid * xmid ** s * snrs ** -s)

    if plot is not None:
        fixpoint = ys[0][0] * snrs[0] ** mod
        plt.figure()
        plt.xscale("log")
        plt.yscale("log")
        plt.ylim(1, 3)
        plt.xlabel("$s=\\frac{\\rm SNR}{\\rm SNR_{\\rm th}}$")
        plt.ylabel("$p(s) s^{1.5} d\\,\\log(s)$ [a.u.]")
        plt.plot(
            snrs,
            psnrs * snrs ** mod / fixpoint,
            label="Prediction (" + Slabel + ")",
            color="black",
            linewidth=2,
        )  # this is in relative units
        for i, s in enumerate(slopes):
            plt.plot(snrs, ys[i] * snrs ** mod / fixpoint, label="slope=" + str(s)[0:3])
        ax = plt.gca()
        # labels = [item.get_text() for item in ax.get_yticklabels()]
        # print("Labels are ",labels)
        # labels[0] = '1'
        # labels[1] = '2'
        # labels[2] = '3'
        # ax.set_yticklabels(labels)
        ax.set_yticks([1, 2, 3])
        ax.set_yticklabels(["1", "2", "3"])
        plt.legend(fontsize=12)  # ,loc=[6,8])
        plt.tight_layout()
        plt.savefig(plot)
        plt.close()
    return snrs, psnrs, dmpsnrs


def make_dm_redshift(
    grid,
    savename="",
    DMmax=1000,
    zmax=1,
    loc="upper left",
    Macquart=None,
    H0=None,
    showplot=False,
):
    """ generates full dm-redhsift (Macquart) relation """
    if H0 is None:
        H0 = cos.cosmo.H0
    ndm = 1000
    cvs = [0.025, 0.16, 0.5, 0.84, 0.975]
    nc = len(cvs)
    names = ["$2\\sigma$", "$1\\sigma$", "Median", "", ""]
    styles = [":", "--", "-", "--", ":"]
    colours = ["white", "white", "black", "white", "white"]
    DMs = np.linspace(DMmax / ndm, DMmax, ndm, endpoint=True)
    priors = grid.get_p_zgdm(DMs)
    zvals = grid.zvals
    means = np.mean(priors, axis=1)
    csums = np.cumsum(priors, axis=1)

    crits = np.zeros([nc, ndm])

    for i in np.arange(ndm):
        for j, c in enumerate(cvs):
            ic = np.where(csums[i] > c)[0][0]
            if ic > 0:
                kc = (csums[i, ic] - c) / (csums[i, ic] - csums[i, ic - 1])
                crits[j, i] = zvals[ic] * (1 - kc) + zvals[ic - 1] * kc
            else:
                crits[j, i] = zvals[ic]

    # now we convert this between real values and integer units
    dz = zvals[1] - zvals[0]
    crits /= dz

    ### concatenate for plotting ###
    delete = np.where(zvals > zmax)[0][0]
    plotpriors = priors[:, 0:delete]
    plotz = zvals[0:delete]

    plt.figure()

    ############# sets the x and y tics ################3
    ytvals = np.arange(plotz.size)
    every = int(plotz.size / 5)
    ytickpos = np.insert(ytvals[every - 1 :: every], [0], [0])
    yticks = np.insert(plotz[every - 1 :: every], [0], [0])

    # plt.yticks(ytvals[every-1::every],plotz[every-1::every])
    plt.yticks(ytickpos, yticks)
    xtvals = np.arange(ndm)
    everx = int(ndm / 5)
    xtickpos = np.insert(xtvals[everx - 1 :: everx], [0], [0])
    xticks = np.insert(DMs[everx - 1 :: everx], [0], [0])
    plt.xticks(xtickpos, xticks)
    # plt.xticks(xtvals[everx-1::everx],DMs[everx-1::everx])

    ax = plt.gca()
    labels = [item.get_text() for item in ax.get_xticklabels()]
    for i in np.arange(len(labels)):
        thisl = len(labels[i])
        labels[i] = labels[i][0 : thisl - 1]
    ax.set_xticklabels(labels)

    #### rescales priors to max value for visibility's sake ####
    dm_max = np.max(plotpriors, axis=1)
    for i in np.arange(ndm):
        plotpriors[i, :] /= np.max(plotpriors[i, :])

    cmx = plt.get_cmap("cubehelix")
    plt.xlabel("${\\rm DM}_{\\rm EG}$")
    plt.ylabel("z")

    aspect = float(ndm) / plotz.size
    plt.imshow(plotpriors.T, origin="lower", cmap=cmx, aspect=aspect)
    cbar = plt.colorbar()
    cbar.set_label("$p(z|{\\rm DM})/p_{\\rm max}(z|{\\rm DM})$")
    ###### now we plot the specific thingamies #######
    for i, c in enumerate(cvs):
        plt.plot(
            np.arange(ndm),
            crits[i, :],
            linestyle=styles[i],
            label=names[i],
            color=colours[i],
        )

    # Macquart=None
    if Macquart is not None:
        plt.ylim(0, ytvals.size)
        nz = zvals.size

        plt.xlim(0, xtvals.size)
        zmax = zvals[-1]
        DMbar, zeval = igm.average_DM(zmax, cumul=True, neval=nz + 1)
        DMbar = DMbar * H0 / (cos.DEF_H0)  # NOT SURE THIS IS RIGHT
        DMbar = np.array(DMbar)
        DMbar += Macquart  # should be interpreted as muDM

        # idea is that 1 point is 1, hence...
        zeval /= zvals[1] - zvals[0]
        DMbar /= DMs[1] - DMs[0]

        plt.plot(DMbar, zeval, linewidth=2, label="Macquart", color="blue")
        # l=plt.legend(loc='lower right',fontsize=12)
        # l=plt.legend(bbox_to_anchor=(0.2, 0.8),fontsize=8)
        # for text in l.get_texts():
        # 	text.set_color("white")

    # plt.plot([30,40],[0.5,10],linewidth=10)

    plt.legend(loc=loc)
    plt.savefig(savename)
    if H0 is not None:
        plt.title("H0 " + str(H0))
    if showplot:
        plt.show()
    plt.close()


def basic_width_test(pset, surveys, grids, logmean=2, logsigma=1):
    """
    Tests the effects of intrinsic widths on FRB properties
    WARNING: outdated, but kept here for future width analysis
    """

    IGNORE = 0.0  # a parameter that gets ignored

    ############ set default parameters for width distribution ###########
    # 'real' version
    wmin = 0.1
    wmax = 20
    NW = 200
    # short version
    # wmin=0.1
    # wmax=50
    # NW=100
    dw = (wmax - wmin) / 2.0
    widths = np.linspace(wmin, wmax, NW)

    probs = pcosmic.linlognormal_dlin(
        widths, [logmean, logsigma]
    )  # not integrating, just amplitudes
    # normalise probabilities
    probs /= np.sum(probs)

    MAX = 1
    norm = MAX / np.max(probs)
    probs *= norm

    Emin = 10 ** pset[0]
    Emax = 10 ** pset[1]
    alpha = pset[2]
    gamma = pset[3]
    NS = len(surveys)
    rates = np.zeros([NS, NW])
    rp = np.zeros([NS, NW])

    # calculating total rate compared to what is expected for ~width 0
    sumw = 0.0
    DMvals = grids[0].dmvals
    zvals = grids[0].zvals

    dmplots = np.zeros([NS, NW, DMvals.size])
    zplots = np.zeros([NS, NW, zvals.size])

    #### does this for width distributions #######

    for i, s in enumerate(surveys):
        g = grids[i]
        fbar = s.meta["FBAR"]
        tres = s.meta["TRES"]
        fres = s.meta["FRES"]
        thresh = s.meta["THRESH"]
        ###### "Full" version ######
        for j, w in enumerate(widths):
            # artificially set response function
            sens = survey.calc_relative_sensitivity(
                IGNORE, DMvals, w, fbar, tres, fres, model="Quadrature", dsmear=False
            )

            g.calc_thresholds(thresh, sens, alpha=alpha)
            g.calc_pdv(Emin, Emax, gamma)

            g.calc_rates()
            rates[i, j] = np.sum(g.rates)
            dmplots[i, j, :] = np.sum(g.rates, axis=0)
            zplots[i, j, :] = np.sum(g.rates, axis=1)

        rates[i, :] /= rates[i, 0]

        # norm_orates[i,:] = orates[i,0]/rates[i,0]
        rp[i, :] = rates[i, :] * probs

        rp[i, :] *= MAX / np.max(rp[i, :])

    # shows the distribution of widths using Wayne's method
    WAlogmean = np.log(2.67)
    WAlogsigma = np.log(2.07)

    waprobs = pcosmic.linlognormal_dlin(widths, [WAlogmean, WAlogsigma])
    wasum = np.sum(waprobs)

    # rename
    WAprobs = waprobs * MAX / np.max(waprobs)

    return WAprobs, rp[0, :]  # for lat50


def width_test(
    pset,
    surveys,
    grids,
    names,
    logmean=2,
    logsigma=1,
    plot=True,
    outdir="Plots/",
    NP=5,
    scale=3.5,
):
    """ Tests the effects of intrinsic widths on FRB properties 
    Considers three cases:
        - width distribution of Wayne Arcus et al (2020)
        - width distribution specified by user (logmean, logsigma)
        - "practical" width distribution with a few width parameters
        - no width distribution (i.e. for width = 0)
    
    WARNING: outdated, but kept here for future width analysis
    """

    if plot:
        print("Performing test of intrinsic width effects")
        t0 = time.process_time()

    IGNORE = 0.0  # a parameter that gets ignored

    ############ set default parameters for width distribution ###########
    # 'real' version
    wmin = 0.1
    wmax = 30
    NW = 300
    # short version
    # wmin=0.1
    # wmax=50
    # NW=100
    dw = (wmax - wmin) / 2.0
    widths = np.linspace(wmin, wmax, NW)

    probs = pcosmic.linlognormal_dlin(
        widths, logmean, logsigma
    )  # not integrating, just amplitudes
    # normalise probabilities
    probs /= np.sum(probs)
    pextra, err = sp.integrate.quad(
        pcosmic.loglognormal_dlog,
        np.log(wmax + dw / 2.0),
        np.log(wmax * 2),
        args=(logmean, logsigma),
    )
    probs *= 1.0 - pextra  # now sums to 1.-pextra
    probs[-1] += pextra  # now sums back to 1

    styles = ["--", "-.", ":"]

    MAX = 1
    norm = MAX / np.max(probs[:-1])
    probs *= norm
    wsum = np.sum(probs)
    if plot:
        plt.figure()
        plt.xlabel("w [ms]")
        plt.ylabel("p(w)")
        plt.xlim(0, wmax)
        plt.plot(
            widths[:-1],
            probs[:-1],
            label="This work: $\\mu_w=5.49, \\sigma_w=2.46$",
            linewidth=2,
        )

    Emin = 10 ** pset[0]
    Emax = 10 ** pset[1]
    alpha = pset[2]
    gamma = pset[3]
    NS = len(surveys)
    rates = np.zeros([NS, NW])
    rp = np.zeros([NS, NW])
    warp = np.zeros([NS, NW])
    # loop over surveys
    # colours=['blue','orange','

    names = ["ASKAP/FE", "ASKAP/ICS", "Parkes/MB"]
    colours = ["blue", "red", "green", "orange", "black"]

    # calculating total rate compared to what is expected for ~width 0
    sumw = 0.0
    DMvals = grids[0].dmvals
    zvals = grids[0].zvals

    dmplots = np.zeros([NS, NW, DMvals.size])
    zplots = np.zeros([NS, NW, zvals.size])

    ##### values for 'practical' arrays #####

    # NP=5 #NP 10 at scale 2 good
    # scale=3.5

    pdmplots = np.zeros([NS, NP, DMvals.size])
    pzplots = np.zeros([NS, NP, zvals.size])
    prates = np.zeros([NS, NP])

    # collapsed over width dimension with appropriate weights
    spdmplots = np.zeros([NS, DMvals.size])
    spzplots = np.zeros([NS, zvals.size])
    sprates = np.zeros([NS])

    ######## gets original rates for DM and z distributions #########
    # norm_orates=([NS,zvals.size,DMvals.size) # normed to width=0!
    # wait - does this include beamshape and the others not?
    orates = np.zeros([NS])
    norates = np.zeros([NS])  # for normed version
    odms = np.zeros([NS, DMvals.size])
    ozs = np.zeros([NS, zvals.size])
    for i, g in enumerate(grids):
        odms[i, :] = np.sum(g.rates, axis=0)
        ozs[i, :] = np.sum(g.rates, axis=1)
        orates[i] = np.sum(g.rates)  # total rate for grid - 'original' rates

    ############ Wayne Arcus's fits ##########3
    # calculates probabilities and uses this later; WAprobs
    WAlogmean = np.log(2.67)
    WAlogsigma = np.log(2.07)
    waprobs = pcosmic.linlognormal_dlin(widths, WAlogmean, WAlogsigma)
    waprobs /= np.sum(waprobs)
    pextra, err = sp.integrate.quad(
        pcosmic.loglognormal_dlog,
        np.log(wmax + dw / 2.0),
        np.log(wmax * 2),
        args=(WAlogmean, WAlogsigma),
    )
    waprobs *= 1.0 - pextra  # now sums to 1.-pextra
    waprobs[-1] += pextra  # now sums back to 1
    wasum = np.sum(waprobs)

    # rename
    WAprobs = waprobs * MAX / np.max(waprobs)
    WAsum = np.sum(WAprobs)
    # print(np.max(rates[0,:]),np.max(WAprobs))
    ls = ["-", "--", ":", "-.", "-."]

    #### does this for width distributions #######

    for i, s in enumerate(surveys):
        g = grids[i]
        # DMvals=grids[i].dmvals

        # gets the 'practical' widths for this survey
        pwidths, pprobs = survey.make_widths(s, g.state)

        pnorm_probs = pprobs / np.max(pprobs)

        # if plot:
        #    plt.plot(pwidths,pnorm_probs,color=colours[i],marker='o',linestyle='',label='Approx.')
        # gets the survey parameters
        fbar = s.meta["FBAR"]
        tres = s.meta["TRES"]
        fres = s.meta["FRES"]
        thresh = s.meta["THRESH"]

        ######## "practical" version ### (note: not using default behaviour) ########
        for j, w in enumerate(pwidths):
            # artificially set response function
            sens = survey.calc_relative_sensitivity(
                IGNORE, DMvals, w, fbar, tres, fres, model="Quadrature", dsmear=False
            )
            g.calc_thresholds(thresh, sens, alpha=alpha)
            g.calc_pdv(Emin, Emax, gamma)

            g.calc_rates()
            prates[i, j] = np.sum(g.rates) * pprobs[j]
            pdmplots[i, j, :] = np.sum(g.rates, axis=0) * pprobs[j]
            pzplots[i, j, :] = np.sum(g.rates, axis=1) * pprobs[j]
        # sum over weights - could just do all this later, but whatever
        sprates[i] = np.sum(prates[i], axis=0)
        spdmplots[i] = np.sum(pdmplots[i], axis=0)
        spzplots[i] = np.sum(pzplots[i], axis=0)

        ######### "Full" (correct) version #########
        for j, w in enumerate(widths):
            # artificially set response function
            sens = survey.calc_relative_sensitivity(
                IGNORE, DMvals, w, fbar, tres, fres, model="Quadrature", dsmear=False
            )

            g.calc_thresholds(thresh, sens, alpha=alpha)
            g.calc_pdv(Emin, Emax, gamma)

            g.calc_rates()
            rates[i, j] = np.sum(g.rates)

            dmplots[i, j, :] = np.sum(g.rates, axis=0)
            zplots[i, j, :] = np.sum(g.rates, axis=1)

        # this step divides by the full rates for zero width
        norates[i] = (
            orates[i] / rates[i, 0]
        )  # normalises original weight by rate if no width
        sprates[i] /= rates[i, 0]
        rates[i, :] /= rates[i, 0]

        # norm_orates[i,:] = orates[i,0]/rates[i,0]
        rp[i, :] = rates[i, :] * probs
        warp[i, :] = rates[i, :] * WAprobs

        if plot:
            plt.plot(widths[:-1], rp[i, :-1], linestyle=styles[i], linewidth=1)

        norm = MAX / np.max(rp[i, :-1])
        if plot:
            plt.plot(
                widths[:-1],
                rp[i, :-1] * norm,
                label=names[i],
                linestyle=styles[i],
                color=plt.gca().lines[-1].get_color(),
                linewidth=2,
            )

    print("The total fraction of events detected as a function of experiment are")
    print("Survey  name   [input_grid]   WA   lognormal     practical")
    for i, s in enumerate(surveys):
        print(
            i,
            names[i],
            norates[i],
            np.sum(warp[i, :]) / WAsum,
            np.sum(rp[i, :]) / wsum,
            sprates[i],
        )
        # print(i,rates[i,:])
        # print(i,names[i],np.sum(rates[i,:]),np.sum(rp[i,:]),wsum,np.sum(rp[i,:])/wsum)

    if plot:
        plt.plot(
            widths[:-1],
            WAprobs[:-1],
            label="Arcus et al: $\\mu_w=2.67, \\sigma_w=2.07$",
            color="black",
            linestyle="-",
            linewidth=2,
        )

        plt.legend(loc="upper right")
        plt.tight_layout()
        plt.xlim(0, 30)
        plt.savefig(outdir + "/width_effect.pdf")
        plt.close()
        t1 = time.process_time()
        print("Done. Took ", t1 - t0, " seconds.")

        #### we now do DM plots ###
        plt.figure()
        plt.xlabel("DM [pc cm$^{-3}$]")
        plt.ylabel("p(DM) [a.u.]")
        plt.xlim(0, 3000)
        # dmplots[i,j,:]=np.sum(g.rates,axis=0)

        twdm = np.zeros([NS, DMvals.size])
        wadm = np.zeros([NS, DMvals.size])
        w0dm = np.zeros([NS, DMvals.size])
        for i, s in enumerate(surveys):

            w0dm[i] = dmplots[i, 0, :]
            wadm[i] = np.sum((waprobs.T * dmplots[i, :, :].T).T, axis=0) / wasum
            twdm[i] = np.sum((probs.T * dmplots[i, :, :].T).T, axis=0) / wsum

            print(
                "Mean DM for survey ",
                i,
                " is (0) ",
                np.sum(DMvals * w0dm[i, :]) / np.sum(w0dm[i, :]),
            )
            print(
                "                  (full verson) ",
                np.sum(DMvals * twdm[i, :]) / np.sum(twdm[i, :]),
            )
            print(
                "                (wayne arcus a) ",
                np.sum(DMvals * wadm[i, :]) / np.sum(wadm[i, :]),
            )
            print(
                "                    (practical) ",
                np.sum(DMvals * spdmplots[i, :]) / np.sum(spdmplots[i, :]),
            )
            
            if i == 0:
                plt.plot(
                    DMvals,
                    w0dm[i] / np.max(w0dm[i]),
                    linestyle=ls[0],
                    label="$w_{\\rm inc}=0$",
                    color=colours[0],
                )
                
                plt.plot(
                    DMvals,
                    wadm[i] / np.max(wadm[i]),
                    linestyle=ls[2],
                    label="Arcus et al.",
                    color=colours[2],
                )
                plt.plot(
                    DMvals,
                    twdm[i] / np.max(twdm[i]),
                    linestyle=ls[1],
                    label="This work",
                    color=colours[1],
                )

                # plt.plot(DMvals,odms[i]/np.max(odms[i]),linestyle=ls[3],label='old',color=colours[3])
                plt.plot(
                    DMvals,
                    spdmplots[i] / np.max(spdmplots[i]),
                    linestyle=ls[4],
                    label="This work",
                    color=colours[4],
                )
            else:
                plt.plot(
                    DMvals, w0dm[i] / np.max(w0dm[i]), linestyle=ls[0], color=colours[0]
                )
                plt.plot(
                    DMvals, twdm[i] / np.max(twdm[i]), linestyle=ls[1], color=colours[1]
                )
                plt.plot(
                    DMvals, wadm[i] / np.max(wadm[i]), linestyle=ls[2], color=colours[2]
                )
                # plt.plot(DMvals,odms[i]/np.max(odms[i]),linestyle=ls[3],color=colours[3])
                plt.plot(
                    DMvals,
                    spdmplots[i] / np.max(spdmplots[i]),
                    linestyle=ls[4],
                    color=colours[4],
                )
        plt.legend(loc="upper right")
        plt.tight_layout()
        plt.savefig(outdir + "/width_dm_effect.pdf")
        plt.close()

        ##### z plots ####
        plt.figure()
        plt.xlabel("z")
        plt.ylabel("p(z) [a.u.]")
        plt.xlim(0, 3)
        # zplots[i,j,:]=np.sum(g.rates,axis=1)

        twz = np.zeros([NS, zvals.size])
        waz = np.zeros([NS, zvals.size])
        w0z = np.zeros([NS, zvals.size])
        for i, s in enumerate(surveys):

            w0z[i] = zplots[i, 0, :]
            waz[i] = np.sum((waprobs.T * zplots[i, :, :].T).T, axis=0) / wasum
            twz[i] = np.sum((probs.T * zplots[i, :, :].T).T, axis=0) / wsum

            print(
                "Mean z for survey ",
                i,
                " is (0) ",
                np.sum(zvals * w0z[i]) / np.sum(w0z[i]),
            )
            print(
                "                           (tw) ",
                np.sum(zvals * twz[i]) / np.sum(twz[i]),
            )
            print(
                "                           (wa) ",
                np.sum(zvals * waz[i]) / np.sum(waz[i]),
            )
            print(
                "                            (p) ",
                np.sum(zvals * spzplots[i, :]) / np.sum(spzplots[i, :]),
            )

            
            if i == 0:
                plt.plot(
                    zvals,
                    w0z[i] / np.max(w0z[i]),
                    label="$w_{\\rm inc}=0$",
                    linestyle=ls[0],
                    color=colours[0],
                )
                
                plt.plot(
                    zvals,
                    waz[i] / np.max(waz[i]),
                    linestyle=ls[2],
                    label="Arcus et al.",
                    color=colours[2],
                )
                plt.plot(
                    zvals,
                    twz[i] / np.max(twz[i]),
                    label="This work",
                    linestyle=ls[1],
                    color=colours[1],
                )
                
                plt.plot(
                    zvals,
                    spzplots[i] / np.max(spzplots[i]),
                    linestyle=ls[4],
                    label="This work",
                    color=colours[4],
                )
            else:
                plt.plot(
                    zvals, w0z[i] / np.max(w0z[i]), linestyle=ls[0], color=colours[0]
                )
                plt.plot(
                    zvals, twz[i] / np.max(twz[i]), linestyle=ls[1], color=colours[1]
                )
                plt.plot(
                    zvals, waz[i] / np.max(waz[i]), linestyle=ls[2], color=colours[2]
                )
                # plt.plot(zvals,ozs[i]/np.max(ozs[i]),linestyle=ls[3],color=colours[3])
                plt.plot(
                    zvals,
                    spzplots[i] / np.max(spzplots[i]),
                    linestyle=ls[4],
                    color=colours[4],
                )
        plt.legend(loc="upper right")
        plt.tight_layout()
        plt.savefig(outdir + "/width_z_effect.pdf")
        plt.close()

    return WAprobs, rp[0, :]  # for lat50


def test_pks_beam(
    surveys, zDMgrid, zvals, dmvals, pset, outdir="Plots/BeamTest/", zmax=1, DMmax=1000
):
    """
    WARNING: likely outdated, kept here for potential future adaptation
    """

    if not os.path.isdir(outdir):
        os.mkdir(outdir)
    
    from zdm import figures
    # get parameter values
    lEmin, lEmax, alpha, gamma, sfr_n, logmean, logsigma = pset
    Emin = 10 ** lEmin
    Emax = 10 ** lEmax

    # generates a DM mask
    # creates a mask of values in DM space to convolve with the DM grid
    mask = pcosmic.get_dm_mask(dmvals, (logmean, logsigma), zvals, plot=True)

    # get an initial grid with no beam values
    grids = []
    bbs = []
    bos = []

    print("Just got into test parkes beam")

    # norms=np.zeros([len(surveys)])
    # numbins=np.zeros([len(surveys)])
    rates = []
    New = False
    for i, s in enumerate(surveys):
        print("Starting", i)
        # s.beam_b
        # s.beam_o
        print("Sum of i is ", np.sum(s.beam_o))
        print(s.beam_o)
        print(s.beam_b)
        if New == True:

            grid = zdm_grid.Grid()
            grid.pass_grid(zDMgrid, zvals, dmvals)
            grid.smear_dm(mask, logmean, logsigma)
            efficiencies = s.get_efficiency(dmvals)
            grid.calc_thresholds(s.meta["THRESH"], s.mean_efficiencies, alpha=alpha)
            grid.calc_dV()
            grid.set_evolution(
                sfr_n
            )  # sets star-formation rate scaling with z - here, no evoltion...
            grid.calc_pdv(
                Emin, Emax, gamma, s.beam_b, s.beam_o
            )  # calculates volumetric-weighted probabilities
            grid.calc_rates()  # calculates rates by multiplying above with pdm plot
            name = outdir + "rates_" + s.meta["BEAM"] + ".pdf"
            figures.plot_grid(
                grid.rates,
                grid.zvals,
                grid.dmvals,
                zmax=zmax,
                DMmax=DMmax,
                name=name,
                norm=2,
                log=True,
                label="$f(DM,z)p(DM,z)dV$ [Mpc$^3$]",
                project=True,
            )
            grids.append(grid)
            np.save(outdir + s.meta["BEAM"] + "_rates.npy", grid.rates)
            rate = grid.rates
        else:
            rate = np.load(outdir + s.meta["BEAM"] + "_rates.npy")
        print("Sum of rates: ", np.sum(rate), s.meta["BEAM"])
        rates.append(rate)
    fig1 = plt.figure()
    plt.xlabel("z")
    plt.xlim(0, zmax)
    fig2 = plt.figure()
    plt.xlabel("DM")
    plt.xlim(0, DMmax)

    fig3 = plt.figure()
    plt.xlabel("z")
    plt.xlim(0, zmax)
    fig4 = plt.figure()
    plt.xlabel("DM")
    plt.xlim(0, DMmax)

    # plt.yscale('log')
    # now does z-only and dm-only projection plots for Parkes
    for i, s in enumerate(surveys):
        r = rates[i]
        z = np.sum(r, axis=1)
        dm = np.sum(r, axis=0)
        plt.figure(fig1.number)
        plt.plot(zvals, z, label=s.meta["BEAM"])
        plt.figure(fig2.number)
        plt.plot(dmvals, dm, label=s.meta["BEAM"])

        z /= np.sum(z)
        dm /= np.sum(dm)
        plt.figure(fig3.number)
        plt.plot(zvals, z, label=s.meta["BEAM"])
        plt.figure(fig4.number)
        plt.plot(dmvals, dm, label=s.meta["BEAM"])

    plt.figure(fig1.number)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(outdir + "z_projections.pdf")
    plt.close()

    plt.figure(fig2.number)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(outdir + "dm_projections.pdf")
    plt.close()

    plt.figure(fig3.number)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(outdir + "normed_z_projections.pdf")
    plt.close()

    plt.figure(fig4.number)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(outdir + "normed_dm_projections.pdf")
    plt.close()

    ###### makes a 1d set of plots in dm and redshift ########
    font = {"family": "normal", "weight": "normal", "size": 6}

    matplotlib.rc("font", **font)
    fig, ax = plt.subplots(
        3, 2, sharey="row", sharex="col"
    )  # ,sharey='row',sharex='col')

    ax[1, 0].set_xlabel("z")
    ax[1, 1].set_xlabel("DM")
    ax[2, 0].set_xlabel("z")
    ax[2, 1].set_xlabel("DM")
    ax[0, 0].set_ylabel("Abs")
    ax[0, 1].set_ylabel("Abs")
    ax[1, 0].set_ylabel("Dabs")
    ax[1, 1].set_ylabel("Dabs")
    ax[2, 0].set_ylabel("Rel diff")
    ax[2, 1].set_ylabel("Rel diff")

    # force relative range only
    ax[2, 0].set_ylim(-1, 1)
    ax[2, 1].set_ylim(-1, 1)

    ax[0, 0].set_xlim(0, zmax)
    ax[0, 1].set_xlim(0, DMmax)
    ax[1, 0].set_xlim(0, zmax)
    ax[2, 0].set_xlim(0, zmax)
    ax[1, 1].set_xlim(0, DMmax)
    ax[2, 1].set_xlim(0, DMmax)

    # gets Keith's normalised rates
    kr = rates[2]
    kz = np.sum(kr, axis=1)
    kdm = np.sum(kr, axis=0)
    kz /= np.sum(kz)
    kdm /= np.sum(kdm)

    ax[0, 0].plot(zvals, kz, label=surveys[2].meta["BEAM"], color="black")
    ax[0, 1].plot(dmvals, kdm, label=surveys[2].meta["BEAM"], color="black")

    for i, s in enumerate(surveys):
        if i == 2:
            continue

        # calculates relative and absolute errors in dm and z space
        z = np.sum(rates[i], axis=1)
        dm = np.sum(rates[i], axis=0)
        z /= np.sum(z)
        dm /= np.sum(dm)

        dz = z - kz
        ddm = dm - kdm
        rdz = dz / kz
        rdm = ddm / kdm

        ax[0, 0].plot(zvals, z, label=s.meta["BEAM"])
        ax[0, 1].plot(dmvals, dm, label=s.meta["BEAM"])
        ax[1, 0].plot(zvals, dz, label=s.meta["BEAM"])
        ax[1, 1].plot(dmvals, ddm, label=s.meta["BEAM"])
        ax[2, 0].plot(zvals, rdz, label=s.meta["BEAM"])
        ax[2, 1].plot(dmvals, rdm, label=s.meta["BEAM"])
    ax[0, 0].legend(fontsize=6)
    plt.figure(fig.number)
    plt.tight_layout()
    plt.savefig(outdir + "montage.pdf")
    plt.close()


def test_beam_rates(
    survey,
    zDMgrid,
    zvals,
    dmvals,
    pset,
    binset,
    method=2,
    outdir="Plots/BeamTest/",
    thresh=1e-3,
    zmax=1,
    DMmax=1000,
):
    """ For a list of surveys, construct a zDMgrid object
    binset is the set of bins which we use to simplify the
    beamset
    We conclude that method=2, nbeams=5, acc=0 is the best here
    
    WARNING: likely outdated, to be updated
    """

    # zmax=4
    # DMmax=4000
    from zdm import figures
    if not os.path.isdir(outdir):
        os.mkdir(outdir)

    # get parameter values
    lEmin, lEmax, alpha, gamma, sfr_n, logmean, logsigma, C = pset
    Emin = 10 ** lEmin
    Emax = 10 ** lEmax

    # generates a DM mask
    # creates a mask of values in DM space to convolve with the DM grid
    mask = pcosmic.get_dm_mask(dmvals, (logmean, logsigma), zvals, plot=True)
    efficiencies = survey.get_efficiency(dmvals)

    # get an initial grid with no beam values
    grids = []
    bbs = []
    bos = []

    norms = np.zeros([len(binset)])
    numbins = np.zeros([len(binset)])

    for k, nbins in enumerate(binset):
        grid = zdm_grid.Grid()
        grid.pass_grid(zDMgrid, zvals, dmvals)
        grid.smear_dm(mask, logmean, logsigma)
        grid.calc_thresholds(
            survey.meta["THRESH"], survey.mean_efficiencies, alpha=alpha
        )
        grid.calc_dV()
        grid.set_evolution(
            sfr_n
        )  # sets star-formation rate scaling with z - here, no evoltion...

        if nbins != 0 and nbins != "all":
            survey.init_beam(nbins=nbins, method=method, thresh=thresh)
            bbs.append(np.copy(survey.beam_b))
            bos.append(np.copy(survey.beam_o))
            grid.calc_pdv(
                Emin, Emax, gamma, survey.beam_b, survey.beam_o
            )  # calculates volumetric-weighted probabilities
            numbins[k] = nbins
        elif nbins == 0:
            grid.calc_pdv(
                Emin, Emax, gamma
            )  # calculates volumetric-weighted probabilities
            bbs.append(np.array([1]))
            bos.append(np.array([1]))
            numbins[k] = nbins
        else:
            survey.init_beam(nbins=nbins, method=3, thresh=thresh)
            bbs.append(np.copy(survey.beam_b))
            bos.append(np.copy(survey.beam_o))
            numbins[k] = survey.beam_o.size
            grid.calc_pdv(
                Emin, Emax, gamma, survey.beam_b, survey.beam_o
            )  # calculates volumetric-weighted probabilities

        grid.calc_rates()  # calculates rates by multiplying above with pdm plot
        name = (
            outdir
            + "beam_test_"
            + survey.meta["BEAM"]
            + "_nbins_"
            + str(nbins)
            + ".pdf"
        )
        figures.plot_grid(
            grid.rates,
            grid.zvals,
            grid.dmvals,
            zmax=zmax,
            DMmax=DMmax,
            name=name,
            norm=2,
            log=True,
            label="$f(DM_{\\rm EG},z)p(DM_{\\rm EG},z)dV$ [Mpc$^3$]",
            project=True,
        )
        grids.append(grid)

    # OK, we now have a list of grids with various interpolating factors
    # we produce plots of the rate for each, and also difference plots with the best
    # Does a linear plot relative to the best case

    bestgrid = grids[-1]  # we always have the worst grid at 0
    # bestgrid.rates=bestgrid.rates / np.sum(bestgrid.rates)

    # normalises

    for i, grid in enumerate(grids):
        norms[i] = np.sum(grid.rates)
        grid.rates = grid.rates / norms[i]

    np.save(outdir + survey.meta["BEAM"] + "_total_rates.npy", norms)
    np.save(outdir + survey.meta["BEAM"] + "_nbins.npy", numbins)

    bestz = np.sum(grid.rates, axis=1)
    bestdm = np.sum(grid.rates, axis=0)

    ###### makes a 1d set of plots in dm and redshift ########
    font = {"family": "normal", "weight": "normal", "size": 6}

    matplotlib.rc("font", **font)
    fig, ax = plt.subplots(
        3, 2, sharey="row", sharex="col"
    )  # ,sharey='row',sharex='col')

    # ax[0,0]=fig.add_subplot(221)
    # ax[0,1]=fig.add_subplot(222)
    # ax[1,0]=fig.add_subplot(223)
    # ax[1,1]=fig.add_subplot(224)

    # ax[0,0].plot(grid.zvals,bestz,color='black',label='All')
    ax[1, 0].set_xlabel("z")
    ax[1, 1].set_xlabel("DM_{\\rm EG}")
    ax[2, 0].set_xlabel("z")
    ax[2, 1].set_xlabel("DM_{\\rm EG}")
    ax[0, 0].set_ylabel("Abs")
    ax[0, 1].set_ylabel("Abs")
    ax[1, 0].set_ylabel("Dabs")
    ax[1, 1].set_ylabel("Dabs")
    ax[2, 0].set_ylabel("Rel diff")
    ax[2, 1].set_ylabel("Rel diff")

    # force relative range only
    ax[2, 0].set_ylim(-1, 1)
    ax[2, 1].set_ylim(-1, 1)

    ax[0, 0].set_xlim(0, zmax)
    ax[0, 1].set_xlim(0, DMmax)
    ax[1, 0].set_xlim(0, zmax)
    ax[2, 0].set_xlim(0, zmax)
    ax[1, 1].set_xlim(0, DMmax)
    ax[2, 1].set_xlim(0, DMmax)

    ax[0, 0].plot(
        grid.zvals, np.sum(bestgrid.rates, axis=1), label="All", color="black"
    )
    ax[0, 1].plot(
        grid.dmvals, np.sum(bestgrid.rates, axis=0), label="All", color="black"
    )

    for i, grid in enumerate(grids[:-1]):

        diff = grid.rates - bestgrid.rates

        # calculates relative and absolute errors in dm and z space
        dz = np.sum(diff, axis=1)
        ddm = np.sum(diff, axis=0)
        rdz = dz / bestz
        rdm = ddm / bestdm

        thisz = np.sum(grid.rates, axis=1)
        thisdm = np.sum(grid.rates, axis=0)

        ax[0, 0].plot(grid.zvals, thisz, label=str(binset[i]))
        ax[0, 1].plot(grid.dmvals, thisdm, label=str(binset[i]))
        ax[1, 0].plot(grid.zvals, dz, label=str(binset[i]))
        ax[1, 1].plot(grid.dmvals, ddm, label=str(binset[i]))
        ax[2, 0].plot(grid.zvals, rdz, label=str(binset[i]))
        ax[2, 1].plot(grid.dmvals, rdm, label=str(binset[i]))
    ax[0, 0].legend(fontsize=6)
    plt.figure(fig.number)
    plt.tight_layout()
    plt.savefig(
        outdir + "d_dm_z_" + survey.meta["BEAM"] + "_nbins_" + str(binset[i]) + ".pdf"
    )
    plt.close()

    acc = open(outdir + "accuracy.dat", "w")
    mean = np.mean(bestgrid.rates)
    size = bestgrid.rates.size
    string = "#Nbins   Acc     StdDev    StdDev/mean; mean={:.2E}\n".format(mean)
    acc.write("#Nbins   Acc     StdDev    StdDev/mean; mean=" + str(mean) + "\n")

    for i, grid in enumerate(grids[:-1]):

        diff = grid.rates - bestgrid.rates

        inaccuracy = np.sum(diff ** 2)
        std_dev = (inaccuracy / size) ** 0.5
        rel_std_dev = std_dev / mean
        # print("Beam with bins ",binset[i]," has total inaccuracy ",inaccuracy)
        string = "{:.0f} {:.2E} {:.2E} {:.2E}".format(
            binset[i], inaccuracy, std_dev, rel_std_dev
        )
        acc.write(string + "\n")
        name = (
            outdir
            + "diff_beam_test_"
            + survey.meta["BEAM"]
            + "_nbins_"
            + str(binset[i])
            + ".pdf"
        )

        figures.plot_grid(
            diff,
            grid.zvals,
            grid.dmvals,
            zmax=zmax,
            DMmax=DMmax,
            name=name,
            norm=0,
            log=False,
            label="$f(DM_{\\rm EG},z)p(DM_{\\rm EG},z)dV$ [Mpc$^3$]",
            project=True,
        )
        diff = diff / bestgrid.rates
        nans = np.isnan(diff)
        diff[nans] = 0.0
        name = (
            outdir
            + "rel_diff_beam_test_"
            + survey.meta["BEAM"]
            + "_nbins_"
            + str(binset[i])
            + ".pdf"
        )
        figures.plot_grid(
            diff,
            grid.zvals,
            grid.dmvals,
            zmax=zmax,
            DMmax=DMmax,
            name=name,
            norm=0,
            log=False,
            label="$f(DM_{\\rm EG},z)p(DM_{\\rm EG},z)dV$ [Mpc$^3$]",
            project=True,
        )

    acc.close()


def initialise_grids(
    surveys: list,
    zDMgrid: np.ndarray,
    zvals: np.ndarray,
    dmvals: np.ndarray,
    state: parameters.State,
    wdist=True,
    repeaters=False,
):
    """ For a list of surveys, construct a zDMgrid object
    wdist indicates a distribution of widths in the survey,
    i.e. do not use the mean efficiency
    Assumes that survey efficiencies ARE initialised

    Args:
        surveys (list): [description]
        zDMgrid (np.ndarray): [description]
        zvals (np.ndarray): [description]
        dmvals (np.ndarray): [description]
        state (parameters.State): parameters guiding the analysis
            Each grid gets its *own* copy
        wdist (bool, optional): [description]. Defaults to False.

    Returns:
        list: list of Grid objects
    """
    if not isinstance(surveys, list):
        surveys = [surveys]

    # generates a DM mask
    # creates a mask of values in DM space to convolve with the DM grid
    mask = pcosmic.get_dm_mask(
        dmvals, (state.host.lmean, state.host.lsigma), zvals, plot=True
    )
    grids = []
    for survey in surveys:
        prev_grid = None
        # print(f"Working on {survey.name}")

        if repeaters:
            grid = zdm_repeat_grid.repeat_Grid(
                survey, copy.deepcopy(state), zDMgrid, zvals, dmvals, mask, wdist, prev_grid=prev_grid
            )
        else:
            grid = zdm_grid.Grid(
                survey, copy.deepcopy(state), zDMgrid, zvals, dmvals, mask, wdist, prev_grid=prev_grid
            )

        grids.append(grid)
        prev_grid = grid

    return grids

def get_filenames(datdir,state,tag,method):
    """
    Initialises filenames for saving grid data.
    
    Args:
        datdir [string]: directory for saving
        state [zdm.state]: state construct
        tag [string]: unique tag to identify save files
        method [string]: MC or analytic
    """
    try:
        os.mkdir(datdir)
    except:
        pass
    if method == "MC":
        savefile = datdir + "/" + tag + "zdm_MC_grid_" + str(state.IGM.logF) + ".npy"
        datfile = datdir + "/" + tag + "zdm_MC_data_" + str(state.IGM.logF) + ".npy"
        zfile = datdir + "/" + tag + "zdm_MC_z_" + str(state.IGM.logF) + ".npy"
        dmfile = datdir + "/" + tag + "zdm_MC_dm_" + str(state.IGM.logF) + ".npy"
    elif method == "analytic":
        savefile = (
            datdir
            + "/"
            + tag
            + "zdm_A_grid_"
            + str(state.IGM.logF)
            + "H0_"
            + str(state.cosmo.H0)
            + ".npy"
        )
        datfile = (
            datdir
            + "/"
            + tag
            + "zdm_A_data_"
            + str(state.IGM.logF)
            + "H0_"
            + str(state.cosmo.H0)
            + ".npy"
        )
        zfile = (
            datdir
            + "/"
            + tag
            + "zdm_A_z_"
            + str(state.IGM.logF)
            + "H0_"
            + str(state.cosmo.H0)
            + ".npy"
        )
        dmfile = (
            datdir
            + "/"
            + tag
            + "zdm_A_dm_"
            + str(state.IGM.logF)
            + "H0_"
            + str(state.cosmo.H0)
            + ".npy"
        )
        C0file = (
            datdir
            + "/"
            + tag
            + "zdm_A_C0_"
            + str(state.IGM.logF)
            + "H0_"
            + str(state.cosmo.H0)
            + ".npy"
        )
    return savefile,datfile,zfile,dmfile,C0file

# generates grid based on Monte Carlo model
def get_zdm_grid(
    state: parameters.State,
    new=True,
    plot=False,
    method="analytic",
    nz=500,
    zmin=0.01,
    zmax=5,
    ndm=1400,
    dmmax=7000.0,
    datdir="GridData",
    tag="",
    orig=False,
    verbose=False,
    save=False,
    zlog=False,
):
    """Generate a grid of z vs. DM for an assumed F value
    for a specified z range and DM range.

    Args:
        state (parameters.State): Object holding all the key parameters for the analysis
        new (bool, optional):
            True (default): generate a new grid
            False: load from file.
        plot (bool, optional):
            True: Make a2D plot of the zdm distribution.
            False (default): do nothing.
        method (str, optional): Method of generating p(DM|z).
            Analytic (default): use pcosic make_c0_grid
            MC: generate via Monte Carlo using dlas.monte_dm
        nz (int, optional): Size of grid in redshift. Defaults to 500.
        zmin (float,optional): Minimum z. Used only for log-spaced grids.
        zmax (float, optional): Maximum z. Defaults to 5. Represents the
                upper edge of the maximum zbin.
        ndm (int, optional): Size of grid in DM.  Defaults to 1400.
        dmmax ([type], optional): Maximum DM of grid. Defaults to 7000.
                Represents the upper edge of the max bin in the DM grid.
        datdir (str, optional): Directory to load/save grid data. Defaults to 'GridData'.
        tag (str, optional): Label for grids (unique identifier). Defaults to "".
        orig (bool, optional): Use original calculations for 
            things like C0. Defaults to False.
        save (bool, optional): Save the grid to disk?
        zlog (bool, optional): Use a log-spaced redshift grid? Defaults to False.

    Returns:
        tuple: zDMgrid, zvals, dmvals
    """
    # gets filenames in case these are being saved
    if save:
        savefile,datfile,zfile,dmfile,C0file = get_filenames(datdir,state,tag,method)

    # labelled pickled files with H0
    if new:
        
        ddm = dmmax / ndm
        # the DMvals and the zvals generated below
        # represent bin centres. i.e. characteristic values.
        # Probabilities then derived will correspond
        # to p(zbin-0.5*dz < z < zbin+0.5*dz) etc.
        
        if zlog:
            # generates a pseudo-log spacing
            # grid values increase with \sqrt(log)
            lzmax = np.log10(zmax)
            lzmin = np.log10(zmin)
            zvals = np.logspace(lzmin, lzmax, nz)
        else:
            dz = zmax / nz
            zvals = (np.arange(nz) + 0.5) * dz
        dmvals = (np.arange(ndm) + 0.5) * ddm
        
        # Deprecated. dmvals now mean bin centre values
        # dmmeans used to be those bin centres
        #dmmeans = dmvals[1:] - (dmvals[1] - dmvals[0]) / 2.0
        
        # initialises zDM grid
        zdmgrid = np.zeros([nz, ndm])

        if method == "MC":
            # generate DM grid from the models
            # NOT CHECKED
            if verbose:
                print("Generating the zdm Monte Carlo grid")
            nfrb = 10000
            t0 = time.process_time()
            DMs = dlas.monte_DM(np.array(zvals) * 3000, nrand=nfrb)
            # DMs *= 200000 #seems to be a good fit...
            t1 = time.process_time()
            dt = t1 - t0
            hists = []
            for i, z in enumerate(zvals):
                hist, bins = np.histogram(DMs[:, i], bins=dmvals)
                hists.append(hist)
            all_hists = np.array(hists)
        elif method == "analytic":
            if verbose:
                print("Generating the zdm analytic grid")
            t0 = time.process_time()
            # calculate constants for p_DM distribution
            if orig:
                C0s = pcosmic.make_C0_grid(zvals, state.IGM.logF)
            else:
                # interpolate C0 as a function of log10F
                f_C0_3 = cosmic.grab_C0_spline()
                actual_F = 10 ** (state.IGM.logF)
                sigma = actual_F / np.sqrt(zvals)
                C0s = f_C0_3(sigma)
            # generate pDM grid using those COs
            zDMgrid = pcosmic.get_pDM_grid(state, dmvals, zvals, C0s, zlog=zlog)

        metadata = np.array([nz, ndm, state.IGM.logF])
        if save:
            np.save(savefile, zDMgrid)
            np.save(datfile, metadata)
            np.save(zfile, zvals)
            np.save(dmfile, dmvals)
    else:
        zDMgrid = np.load(savefile)
        zvals = np.load(zfile)
        dmvals = np.load(dmfile)
        metadata = np.load(datfile)
        nz, ndm, F = metadata

    if plot:
        plt.figure()
        plt.xlabel("DM_{\\rm EG} [pc cm$^{-3}$]")
        plt.ylabel("p(DM_{\\rm EG})")

        nplot = int(zvals.size / 10)
        for i, z in enumerate(zvals):
            if i % nplot == 0:
                plt.plot(dmvals, zDMgrid[i, :], label="z=" + str(z)[0:4])
        plt.legend()
        plt.tight_layout()
        plt.savefig("p_dm_slices.pdf")
        plt.close()

    return zDMgrid, zvals, dmvals


def plot_zdm_basic_paper(
    zDMgrid,
    zvals,
    dmvals,
    zmax=1,
    DMmax=1000,
    norm=0,
    log=True,
    name="temp.pdf",
    ylabel=None,
    label="$\\log_{10}p(DM_{\\rm EG},z)$",
    project=False,
    conts=False,
    FRBZ=None,
    FRBDM=None,
    title="Plot",
    H0=None,
    showplot=False,
):
    """ Plots basic distributions of z and dm for the paper """
    if H0 is None:
        H0 = cos.cosmo.H0
    cmx = plt.get_cmap("cubehelix")

    ##### imshow of grid #######

    # we protect these variables
    zDMgrid = np.copy(zDMgrid)
    zvals = np.copy(zvals)
    dmvals = np.copy(dmvals)

    plt.figure()
    # rect_2D=[0,0,1,1]
    ax1 = plt.axes()

    plt.sca(ax1)

    plt.xlabel("z")
    if ylabel is not None:
        plt.ylabel(ylabel)
    else:
        plt.ylabel("${\\rm DM}_{\\rm EG}$")

    nz, ndm = zDMgrid.shape

    ixmax = np.where(zvals > zmax)[0]
    if len(ixmax) > 0:
        zvals = zvals[: ixmax[0]]
        nz = zvals.size
        zDMgrid = zDMgrid[: ixmax[0], :]

    ### generates contours *before* cutting array in DM ###
    ### might need to normalise contours by integer lengths, oh well! ###
    if conts:
        nc = len(conts)
        carray = np.zeros([nc, nz])
        for i in np.arange(nz):
            cdf = np.cumsum(zDMgrid[i, :])
            cdf /= cdf[-1]

            for j, c in enumerate(conts):
                less = np.where(cdf < c)[0]

                if len(less) == 0:
                    carray[j, i] = 0.0
                    dmc = 0.0
                    il1 = 0
                    il2 = 0
                else:
                    il1 = less[-1]

                    if il1 == ndm - 1:
                        il1 = ndm - 2

                    il2 = il1 + 1
                    k1 = (cdf[il2] - c) / (cdf[il2] - cdf[il1])
                    dmc = k1 * dmvals[il1] + (1.0 - k1) * dmvals[il2]
                    carray[j, i] = dmc

        ddm = dmvals[1] - dmvals[0]
        carray /= ddm  # turns this into integer units for plotting

    iymax = np.where(dmvals > DMmax)[0]
    if len(iymax) > 0:
        dmvals = dmvals[: iymax[0]]
        zDMgrid = zDMgrid[:, : iymax[0]]
        ndm = dmvals.size

    # currently this is "per cell" - now to change to "per DM"
    # normalises the grid by the bin width, i.e. probability per bin, not probability density
    ddm = dmvals[1] - dmvals[0]
    dz = zvals[1] - zvals[0]

    zDMgrid /= ddm  # norm=1

    # checks against zeros for a log-plot
    orig = np.copy(zDMgrid)
    zDMgrid = zDMgrid.reshape(zDMgrid.size)
    setzero = np.where(zDMgrid == 0.0)
    zDMgrid = np.log10(zDMgrid)
    zDMgrid[setzero] = -100
    zDMgrid = zDMgrid.reshape(nz, ndm)

    # gets a square plot
    aspect = nz / float(ndm)

    # sets the x and y tics
    xtvals = np.arange(zvals.size)
    everx = int(zvals.size / 5)
    plt.xticks(xtvals[everx - 1 :: everx], zvals[everx - 1 :: everx])

    ytvals = np.arange(dmvals.size)
    every = int(dmvals.size / 5)
    plt.yticks(ytvals[every - 1 :: every], dmvals[every - 1 :: every])

    im = plt.imshow(
        zDMgrid.T, cmap=cmx, origin="lower", interpolation="None", aspect=aspect
    )

    ###### gets decent axis labels, down to 1 decimal place #######
    ax = plt.gca()
    labels = [item.get_text() for item in ax.get_xticklabels()]
    for i in np.arange(len(labels)):
        labels[i] = labels[i][0:4]
    ax.set_xticklabels(labels)
    labels = [item.get_text() for item in ax.get_yticklabels()]
    for i in np.arange(len(labels)):
        if "." in labels[i]:
            labels[i] = labels[i].split(".")[0]
    ax.set_yticklabels(labels)
    ax.yaxis.labelpad = 0

    # plots contours i there
    cls = [":", "--", "-", "--", ":"]
    if conts:
        plt.ylim(0, ndm - 1)
        for i in np.arange(nc):
            j = int(nc - i - 1)
            plt.plot(
                np.arange(nz),
                carray[j, :],
                label=str(conts[j]),
                color="white",
                linestyle=cls[i],
            )
        l = plt.legend(loc="lower right", fontsize=12)
        # l=plt.legend(bbox_to_anchor=(0.2, 0.8),fontsize=8)
        for text in l.get_texts():
            text.set_color("white")

    # limit to a reasonable range if logscale
    themax = zDMgrid.max()
    themin = int(themax - 4)
    themax = int(themax)
    plt.clim(themin, themax)

    cbar = plt.colorbar(im, fraction=0.046, shrink=1.2, aspect=15, pad=0.05)
    cbar.set_label(label)
    plt.clim(-4, 0)
    plt.tight_layout()

    plt.savefig(name)
    plt.title(title + str(H0))
    if showplot:
        plt.show()
    plt.close()

def get_alevels(zDMgrid,Aconts):
    """
    Gets contour levels giving 
    
    Grid: inoput zDM grid
    Aconts: list of contour levels giving %
    
    """
    slist = np.sort(zDMgrid.flatten())
    cslist = np.cumsum(slist)
    cslist /= cslist[-1]
    nAc = len(Aconts)
    alevels = np.zeros([nAc])
    for i, ac in enumerate(Aconts):
        # cslist is the cumulative probability distribution
        # Where cslist > ac determines the integer locations
        #    of all cells exceeding the threshold
        # The first in this list is the first place exceeding
        #    the threshold
        # The value of slist at that point is the
        #    level of the countour to draw
        iwhich = np.where(cslist > ac)[0][0]
        alevels[i] = slist[iwhich]
    return alevels
    
def plot_grid_2(
    zDMgrid,
    zvals,
    dmvals,
    zmax=1,
    DMmax=1000,
    norm=0,
    log=True,
    name="temp.pdf",
    label='$\\log_{10}p(DM_{\\rm cosmic},z)$',
    ylabel="${\\rm DM}_{\\rm cosmic}$ (pc cm$^{-3}$)",
    project=False,
    conts=False,
    FRBZs=None,
    FRBDMs=None,
    plt_dicts=None,
    cmap=None,
    cont_colours=None,
    Aconts=False,
    Macquart=None,
    title=None,
    H0=None,
    showplot=False,
    DMlines=None,
    DMlims=None,
    clim=False,
    special=None,
    pdmgz=None,
    save=True,
    othergrids=None,
    othernames=None
):
    """
    Very complicated routine for plotting 2D zdm grids 

    Args:
        zDMgrid (2D array): P(z,DM) grid
        zvals (1D array): z values corresponding to zDMgrid
        dmvals (1D array): DM values corresponding to zDMgrid
        zmax (int, optional): Maximum z value to display
        DMmax (int, optional): Maximum DM value to display
        norm (int, optional): Method to normalise zDMgrid.
                                0: No normalisation
                                1: Normalise by dm bin
                                2: Normalise by sum of zDMgrid
                                3: Normalise by max value of zDMgrid
        log (bool, optional): Plot P(z,DM) in log space
        name (str, optional): Outfile name
        label (str, optional): Colourbar label
        ylabel (str,optional): Label on y axis of plot
        project (bool, optional): Add projections of P(z) and P(DM)
        conts (bool, optional): [description]. Defaults to False.
        FRBZs (list of 1D arrays, optional): List of FRB Zs to plot (each list can have customised plotting styles, e.g. for different surveys)
        FRBDMs (list of 1D arrays, optional): List of FRB DMs to plot (corrseponding to FRBZs)
        plt_dicts (list of dictionaries, optional): List of dictionaries containing the plotting parameters for each 'set' of data points (corresponding to FRBZs and FRBDMs). E.g. can contain marker, color, label etc
        cmap (str, optional): Alternate color map for PDF
        Aconts (bool, optional): [description]. Defaults to False.
        Macquart (state, optional): state object.  Used to generate the Maquart relation.
            Defaults to None, i.e. do not show the Macquart relation.
        title (str, optional): Title of the plot
        H0 ([type], optional): [description]. Defaults to None.
        showplot (bool, optional): use plt.show to show plot. Defaults to False.
        DMlines (list, optional): plot lines for unlocalised FRBs at these DMs
        DMlims (list, optional): plot horizontal lines to indicate the maximum searched DM of a given survey
        clim ([float,float], optional): pair of floats giving colorbar limits.
            Defaults to False (automatic limit)
        special(list,optional): list of [z,dm] values to show as a special big star
        pdmgz(list of floats, optional): a list of cumulative values of p(DM|z) to
            plot. Must range from 0 to 1.
        othergrids (list of grids) [None]: a list of grids to plot contours for. Uses
            Aconts
        othernames (list of names) [None]: list of names for original *and* other grid.
            Used only if othergrids is not None. Must be length of othergrids +1.
    """
    if H0 is None:
        H0 = cos.cosmo.H0
    if cmap is None:
        cmx = plt.get_cmap("cubehelix")
    else:
        cmx = plt.get_cmap(cmap)

    ##### imshow of grid #######

    # we protect these variables
    zDMgrid = np.copy(zDMgrid)
    zvals = np.copy(zvals)
    dmvals = np.copy(dmvals)

    if project:
        plt.figure(1, figsize=(8, 8))
        left, width = 0.1, 0.65
        bottom, height = 0.1, 0.65
        gap = 0.02
        woff = width + gap + left
        hoff = height + gap + bottom
        dw = 1.0 - woff - gap
        dh = 1.0 - hoff - gap

        delta = 1 - height - bottom - 0.05
        gap = 0.11
        rect_2D = [left, bottom, width, height]
        rect_1Dx = [left, hoff, width, dh]
        rect_1Dy = [woff, bottom, dw, height]
        rect_cb = [woff, hoff, dw * 0.5, dh]
        ax1 = plt.axes(rect_2D)
        axx = plt.axes(rect_1Dx)
        axy = plt.axes(rect_1Dy)
        acb = plt.axes(rect_cb)
    else:
        plt.figure()
        ax1 = plt.axes()

    plt.sca(ax1)

    plt.xlabel("z")
    plt.ylabel(ylabel)

    nz, ndm = zDMgrid.shape
    
    # attenuate grids in x-direction
    ixmax = np.where(zvals > zmax)[0]
    if len(ixmax) > 0:
        zvals = zvals[: ixmax[0]]
        nz = zvals.size
        zDMgrid = zDMgrid[: ixmax[0], :]
        if othergrids:
            for grid in othergrids:
                grid = grid[: ixmax[0], :]
    
    # currently this is "per cell" - now to change to "per DM"
    # normalises the grid by the bin width, i.e. probability per bin, not probability density
    ddm = dmvals[1] - dmvals[0]
    dz = zvals[1] - zvals[0]
    if norm == 1:
        zDMgrid /= ddm
        if othergrids is not None:
            for grid in othergrids:
                grid /= ddm
        # if Aconts:
        #    alevels /= ddm
    elif norm == 2:
        xnorm = np.sum(zDMgrid)
        zDMgrid /= xnorm
        if othergrids is not None:
            for grid in othergrids:
                grid /= np.sum(grid)
        # if Aconts:
        #    alevels /= xnorm
    elif norm == 3:
        zDMgrid /= np.max(zDMgrid)
        if othergrids is not None:
            for grid in othergrids:
                grid /= np.max(grid)

    # sets up to plot contour-like things as a function of p(dm given z)
    if pdmgz is not None:
        # gets all values where zsum is not zero
        z1d = np.sum(zDMgrid,axis=1) # sums over DM
        OK = np.where(z1d > 0.)[0]
        pdmgz_z = zvals[OK]
        pdmgz_cs = np.cumsum(zDMgrid[OK,:],axis=1)
        pdmgz_dm = np.zeros([pdmgz_z.size, len(pdmgz)])
        for iz,z in enumerate(pdmgz_z):
            this_cs = pdmgz_cs[iz,:]/pdmgz_cs[iz,-1]
            for iv,val in enumerate(pdmgz):
                i1 = np.where(this_cs < val)[0][-1]
                i2 = i1+1
                k2 = (val - this_cs[i1])/(this_cs[i2] - this_cs[i1])
                k1 = 1.-k2
                dmval = k1*dmvals[i1] + k2*dmvals[i2]
                pdmgz_dm[iz,iv] = dmval
    
    # sets contours according to norm
    if Aconts:
        
        alevels = get_alevels(zDMgrid,Aconts)
        if norm == 1:
            alevels /= ddm
        elif norm == 2:
            alevels /= xnorm
        
        if othergrids is not None:
            other_alevels=[]
            for grid in othergrids:
                other_alevels.append(get_alevels(grid,Aconts))

    ### generates contours *before* cutting array in DM ###
    ### might need to normalise contours by integer lengths, oh well! ###
    if conts:
        nc = len(conts)
        carray = np.zeros([nc, nz])
        for i in np.arange(nz):
            cdf = np.cumsum(zDMgrid[i, :])
            cdf /= cdf[-1]

            for j, c in enumerate(conts):
                less = np.where(cdf < c)[0]

                if len(less) == 0:
                    carray[j, i] = 0.0
                    dmc = 0.0
                    il1 = 0
                    il2 = 0
                else:
                    il1 = less[-1]

                    if il1 == ndm - 1:
                        il1 = ndm - 2

                    il2 = il1 + 1
                    k1 = (cdf[il2] - c) / (cdf[il2] - cdf[il1])
                    dmc = k1 * dmvals[il1] + (1.0 - k1) * dmvals[il2]
                    carray[j, i] = dmc

        ddm = dmvals[1] - dmvals[0]
        carray /= ddm  # turns this into integer units for plotting

    iymax = np.where(dmvals > DMmax)[0]
    if len(iymax) > 0:
        dmvals = dmvals[: iymax[0]]
        zDMgrid = zDMgrid[:, : iymax[0]]
        ndm = dmvals.size
        if othergrids:
            for i,grid in enumerate(othergrids):
                othergrids[i] = grid[:, : iymax[0]]
    
    # now sets the limits to the actual size of the grid
    NX,NY = zDMgrid.shape
    plt.xlim(0,NX)
    plt.ylim(0,NY)
    
    if log:
        # checks against zeros for a log-plot
        orig = np.copy(zDMgrid)
        zDMgrid = zDMgrid.reshape(zDMgrid.size)
        setzero = np.where(zDMgrid == 0.0)
        zDMgrid = np.log10(zDMgrid)
        zDMgrid[setzero] = -100
        zDMgrid = zDMgrid.reshape(nz, ndm)
        if Aconts:
            alevels = np.log10(alevels)
    else:
        orig = zDMgrid

    # gets a square plot
    aspect = nz / float(ndm)

    # sets the x and y tics. These are now bin edges
    
    xtvals = np.arange(zvals.size+1)
    xtlabels = np.linspace(0.,zvals[0]+zvals[-1],zvals.size+1)
    everx = int(zvals.size / 5)
    # adds xticks at "edges"
    xtvals[-1] *= 0.999 # just allows it to squeeze on
    plt.xticks(xtvals[0 :: everx], xtlabels[0 :: everx])
    
    ytvals = np.arange(dmvals.size+1)
    ytvals[-1] *= 0.999 # just allows it to squeeze on
    ytlabels = np.linspace(0.,dmvals[0]+dmvals[-1],dmvals.size+1)
    every = int(dmvals.size / 5)
    plt.yticks(ytvals[0 :: every], ytlabels[0 :: every])

    im = plt.imshow(
        zDMgrid.T, cmap=cmap, origin="lower", interpolation="None", aspect=aspect
    )
    
    # plots "A"contours (i.e., over "Amplitudes"). Doing so for multiple grids
    # if necessary
    # NOTE: currently no way to plot contour labels, hence the use of dummy plots
    if Aconts:
        styles = [":", "-.", "--","-"]
        
        ax = plt.gca()
        cs = ax.contour(
            zDMgrid.T, levels=alevels, origin="lower", colors="white", linestyles=styles, linewidths=2
        )
        cntrs=[cs]
        if othernames is not None:
            h,=plt.plot([-1e6,-2e6],[-1e6,-2e6],linestyle=styles[0],
                                    color = cont_colours[0],label=othernames[0])
            handles=[h]
            
        if othergrids is not None:
            for i,grid in enumerate(othergrids):
                cntr = ax.contour(grid.T, levels=other_alevels[i], origin="lower",
                    linestyles=[styles[i+1]],colors = [cont_colours[i+1]])
                if othernames is not None:
                    #make a dummy plot
                    h,=plt.plot([-1e6,-2e6],[-1e6,-2e6],linestyle=styles[i+1], marker=plt_dicts[i+1]['marker'], 
                    markeredgewidth=plt_dicts[i+1]['markeredgewidth'], color=cont_colours[i+1],label=othernames[i+1])
                    handles.append(h)
            if othernames is not None:
                plt.legend(handles=handles,loc="lower right")
    
    
    ###### gets decent axis labels, down to 1 decimal place #######
    ax = plt.gca()
    labels = [item.get_text() for item in ax.get_xticklabels()]
    for i in np.arange(len(labels)):
        labels[i] = labels[i][0:4]
    ax.set_xticklabels(labels)
    labels = [item.get_text() for item in ax.get_yticklabels()]
    for i in np.arange(len(labels)):
        if "." in labels[i]:
            labels[i] = labels[i].split(".")[0]
    ax.set_yticklabels(labels)
    ax.yaxis.labelpad = 0

    # draw horizontal lines for a fixed DM
    if DMlines is not None:
        if log:
            tempgrid = np.copy(zDMgrid)
            tempgrid = zDMgrid - np.max(zDMgrid)
            tempgrid = 10.0 ** zDMgrid
        else:
            tempgrid = zDMgrid
        for DM in DMlines:
            if DM > np.max(dmvals):
                print(
                    "Cannot draw DM line ",
                    DM,
                    " - range ",
                    np.max(dmvals),
                    " too small...",
                )
                continue
            # determines how far to draw line
            iDM2 = np.where(dmvals > DM)[0][0]  # lowest value
            iDM1 = iDM2 - 1
            kDM = (DM - dmvals[iDM1]) / (dmvals[iDM2] - dmvals[iDM1])
            cDM1 = np.cumsum(tempgrid[:, iDM1])
            cDM1 /= cDM1[-1]
            cDM2 = np.cumsum(tempgrid[:, iDM2])
            cDM2 /= cDM2[-1]
            
            stop1 = np.where(cDM1 < 0.99)[0][-1]
            stop2 = np.where(cDM2 < 0.99)[0][-1]
            zstop = kDM * zvals[stop2] + (1.0 - kDM) * zvals[stop1]
            zstop /= zvals[1] - zvals[0]
            DM /= dmvals[1] - dmvals[0]
            plt.plot([0, zstop], [DM, DM], color=data_clr, linestyle=":")

    if DMlims is not None:
        for DMlim in DMlims:
            if DMlim is not None and DMlim < DMmax:
                DMlim /= dmvals[1] - dmvals[0]
                ax.axhline(DMlim, 0, 1, color='white', linestyle="-")

    # performs plots for the pdmgz variable
    if pdmgz is not None:
        styles = ['-','-','-']
        widths = [2,3,2]
        plt.ylim(0,ndm-1)
        plt.xlim(0,nz-1)
        # now converts to plot units [urgh...]
        plot_z = np.arange(pdmgz_z.size)
        for iv,val in enumerate(pdmgz):
            plot_dm = pdmgz_dm[:,iv]/ddm # plot is in integer units
            plt.plot(plot_z,plot_dm,linestyle=styles[iv],linewidth=widths[iv],color='white')
    
    # plots contours i there
    if conts:
        plt.ylim(0, ndm - 1)
        for i in np.arange(nc):
            j = int(nc - i - 1)
            plt.plot(np.arange(nz), carray[j, :], label=str(conts[j]), color="white")
        l = plt.legend(loc="upper left", fontsize=8)
        # l=plt.legend(bbox_to_anchor=(0.2, 0.8),fontsize=8)
        for text in l.get_texts():
            text.set_color("white")

    if Macquart is not None:
        # Note this is the Median for the lognormal, not the mean
        muDMhost = np.log(10 ** Macquart.host.lmean)
        sigmaDMhost = np.log(10 ** Macquart.host.lsigma)
        meanHost = np.exp(muDMhost + sigmaDMhost ** 2 / 2.0)
        medianHost = np.exp(muDMhost)
        # print(f"Host: mean={meanHost}, median={medianHost}")
        plt.ylim(0, ndm - 1)
        plt.xlim(0, nz - 1)
        zmax = zvals[-1]
        nz = zvals.size
        # DMbar, zeval = igm.average_DM(zmax, cumul=True, neval=nz+1)
        DM_cosmic = pcosmic.get_mean_DM(zvals, Macquart)

        # idea is that 1 point is 1, hence...
        zeval = zvals / dz
        DMEG_mean = (DM_cosmic + meanHost) / ddm
        DMEG_median = (DM_cosmic + medianHost) / ddm
        plt.plot(
            zeval,
            DMEG_mean,
            color="blue",
            linewidth=2,
            label="Macquart relation (mean)",
        )
        # removed median, because it is only media of HOST not DM cosmic
        # plt.plot(zeval,DMEG_median,color='blue',
        #         linewidth=2, ls='--',
        #         label='Macquart relation (median)')
        l = plt.legend(loc="lower right", fontsize=12)
        # l=plt.legend(bbox_to_anchor=(0.2, 0.8),fontsize=8)
        # for text in l.get_texts():
        # 	text.set_color("white")

    # limit to a reasonable range if logscale
    
    if log:
        themax = np.nanmax(zDMgrid)
        themin = int(themax - 4)
        themax = int(themax)
        plt.clim(themin, themax)
    
    if clim:
        plt.clim(clim[0], clim[1])
    
    ##### add FRB host galaxies at some DM/redshift #####
    if FRBZs is not None and len(FRBZs) != 0:
        for i, FRBZ in enumerate(FRBZs):
            if FRBZ is not None and len(FRBZ) != 0:
                FRBDM = FRBDMs[i]
            
                iDMs = FRBDM / ddm
                iZ = FRBZ / dz
                OK = np.where(FRBZ > 0)[0]

                plt.plot(iZ[OK], iDMs[OK], linestyle="", **plt_dicts[i])
            
    legend = plt.legend(loc='upper left')
    # legend = plt.legend(loc='upper left', bbox_to_anchor=(0.0, -0.15), fontsize=12, markerscale=1, ncol=2)
    # legend.get_frame().set_facecolor('lightgrey')

    if special is not None:
        iDM = special[0] / ddm
        iz = special[1] / dz
        plt.plot([iz], [iDM], "*", markersize=10, color="blue", linestyle="")

    # do 1-D projected plots
    if project:
        plt.sca(acb)
        cbar = plt.colorbar(
            im, fraction=0.046, shrink=1.2, aspect=20, pad=0.00, cax=acb
        )
        cbar.ax.tick_params(labelsize=6)
        cbar.set_label(label, fontsize=8)

        axy.set_yticklabels([])
        # axy.set_xticklabels([])
        # axx.set_yticklabels([])
        axx.set_xticklabels([])
        yonly = np.sum(orig, axis=0)
        xonly = np.sum(orig, axis=1)

        axy.plot(yonly, dmvals)  # DM is the vertical axis now
        axx.plot(zvals, xonly)

        # if plotting DM only, put this on the axy axis showing DM distribution
        if FRBDMs is not None:
            for FRBDM in FRBDMs:
                if FRBDM is not None:
                    hvals=np.zeros(FRBDM.size)
                    for i,DM in enumerate(FRBDM):
                        if DM > dmvals[-1]:
                            havls[i] = 0
                        else:
                            hvals[i] = yonly[np.where(dmvals > DM)[0][0]]
                    
                    axy.plot(hvals,FRBDM,'ro',linestyle="")
                    for tick in axy.yaxis.get_major_ticks():
                        tick.label.set_fontsize(6)

        if FRBZs is not None:
            for FRBZ in FRBZs:
                if FRBZ is not None:
                    OK = np.where(FRBZ > 0)[0]
                    hvals = np.zeros(FRBZ[OK].size)
                    for i, Z in enumerate(FRBZ[OK]):
                        hvals[i] = xonly[np.where(zvals > Z)[0][0]]
                    axx.plot(FRBZ[OK], hvals, "ro", color=data_clr, linestyle="", markersize=markersize)
                    for tick in axx.xaxis.get_major_ticks():
                        tick.label.set_fontsize(6)
    else:
        cbar = plt.colorbar(im, fraction=0.046, shrink=1.2, aspect=15, pad=0.05)
        cbar.set_label(label)
        plt.tight_layout()

    if title is not None:
        plt.title(title)

    if save:
        plt.tight_layout()
        plt.savefig(name, dpi=300, bbox_inches='tight')
    if showplot:
        plt.show()
    plt.close()


def plot_grid(grid, zvals, dmvals, showplot=False):
    """ Plots a simple 2D grid """
    cmx = plt.get_cmap("cubehelix")

    plt.figure()
    plt.zlabel("z")
    plt.ylabel("DM_{\\rm EG}")

    plt.xtics(np.arange(zvals.size)[::10], zvals[::10])
    plt.ytics(np.arange(dmvals.size)[::10], dmvals[::10])
    plt.imshow(
        grid,
        extent=(zvals[0], zvals[-1], dmvals[0], dmvals[-1]),
        origin="lower",
        cmap=cmx,
    )
    cbar = plt.colorbar()
    cbar.set_label("$p(DM_{\\rm EG}|z)")
    if showplot:
        plt.show()


def plot_efficiencies_paper(survey, savename, label):
    """ Plots a final version of efficiencies for the purpose of the paper """
    dm = survey.DMlist
    eff1s = survey.get_efficiency(dm, model="Quadrature", dsmear=True)
    eff1mean = np.copy(survey.mean_efficiencies)
    eff2s = survey.get_efficiency(dm, model="Sammons", dsmear=True)
    eff2mean = np.copy(survey.mean_efficiencies)
    DMobs = survey.DMs
    NDM = DMobs.size
    DMx = np.zeros([NDM])
    DMy = np.zeros([NDM])
    for i, DMo in enumerate(DMobs):
        pos = np.where(dm > DMo)[0][0]
        DMy[i] = eff1s[i, pos]
        DMx[i] = dm[pos]

    plt.figure()
    plt.ylim(0, 1)

    plt.text(1500, 0.9, label)

    eff = survey.efficiencies
    if "ID" in survey.frbs:
        labels = survey.frbs["ID"]
    else:
        labels = np.arange(survey.NFRB)

    plt.xlabel("DM [pc cm$^{-3}$]")
    plt.ylabel("Efficiency $\\epsilon$")  # \\dfrac{F_{\\rm 1\,ms}}{F_{\\rm th}}$')

    ls = ["-", ":", "--", "-."]

    for i in np.arange(survey.NFRB):
        ils = int(i / 10)
        if i == 0:
            plt.plot(dm, eff1s[i], linestyle=":", color="black", label="$\\epsilon_i$")
        else:
            plt.plot(dm, eff1s[i], linestyle=":", color="black")
        # plt.plot(dm,eff2s[i],linestyle=ls[ils],lw=1)
    plt.plot(
        dm, eff1mean, linewidth=3, color="blue", ls="-", label="$\\bar{\\epsilon}$"
    )
    plt.plot(DMx, DMy, "ro", label="${\\rm DM}_i$")

    # plt.plot(dm,eff2mean,linewidth=1,color='black',ls='-')
    ncol = int((survey.NFRB + 9) / 10)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(savename)
    plt.close()


def plot_efficiencies(survey, savename="Plots/efficiencies.pdf", showplot=False):
    """ Plots efficiency as function of DM """
    plt.figure()

    eff = survey.efficiencies
    dm = survey.DMlist
    if "ID" in survey.frbs:
        labels = survey.frbs["ID"]
    else:
        labels = np.arange(survey.NFRB)

    plt.xlabel("DM [pc cm$^{-3}$]")
    plt.ylabel("Efficiency $\\epsilon$")

    ls = ["-", ":", "--", "-."]

    for i in np.arange(survey.NFRB):
        ils = int(i / 10)
        plt.plot(dm, eff[i], label=labels[i], linestyle=ls[ils])
    plt.plot(dm, survey.mean_efficiencies, linewidth=2, color="black", ls="-")
    ncol = int((survey.NFRB + 9) / 10)
    plt.legend(loc="upper right", fontsize=min(14, 200.0 / survey.NFRB), ncol=ncol)
    plt.tight_layout()
    plt.savefig(savename)
    if showplot:
        plt.show()
    plt.close()


def plot_beams(prefix):
    """ Plots something to do with beams """
    logb, omega_b = beams.load_beam(prefix)
    total = np.sum(omega_b)
    print("Total length of histogram is ", omega_b.size)

    # rate of -1.5
    b = 10 ** logb
    rate = omega_b * b ** 1.5
    nbins = 10
    b2, o2 = beams.simplify_beam(logb, omega_b, nbins)
    print(b2, o2)

    # note that omega_b is just unscaled total solid angle
    plt.figure()
    plt.xlabel("$B$")
    plt.ylabel("$\\Omega(B)$/bin")
    plt.yscale("log")
    plt.xscale("log")
    plt.plot(b, omega_b, label="original_binning")
    plt.plot(b2, o2, "ro", label="simplified", linestyle=":")
    plt.plot(b, rate, label="Relative rate")
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig("Plots/lat50_beam.pdf")
    plt.close()

    crate = np.cumsum(rate)
    crate /= crate[-1]
    plt.figure()
    plt.xlabel("$\\log_{10}(B)$")
    plt.ylabel("cumulative rate")
    # plt.yscale('log')
    plt.plot(logb, crate)
    plt.tight_layout()
    plt.savefig("Plots/crate_lat50_beam.pdf")
    plt.close()
    crate = np.cumsum(rate)


def process_missing_pfile(pfile, number, howmany):
    """ searches for missing data in cube output and iterates for that only """
    NPARAMS = 8
    current = np.zeros([1, NPARAMS])
    last = np.zeros([NPARAMS])
    this = np.zeros([NPARAMS])
    n = 0  # counts number of lines in this calculation
    new = 0  # counts new significant calculations
    count = 0  # counts total lines

    # the range of calculations to do. These are *inclusive* values
    start = (number - 1) * howmany + 1
    stop = number * howmany
    print("Calculated start and stop as ", start, stop)

    max_number = np.zeros([howmany * 11, NPARAMS])

    # I am now testing how many in the file in total

    with open(pfile) as pf:

        for line in pf:
            # if count==NPARAMS:
            #    break
            vals = line.split()
            for j, v in enumerate(vals):
                this[j] = float(v)

            # tests to see if this is a serious calculation
            for j in np.arange(NPARAMS - 1):
                if j == 4:
                    continue
                if this[j] != last[j]:
                    new += 1
                    break

            # tests to see if we have gone far enough
            if new > stop:
                break

            # test to see if we now do this
            if new >= start:
                max_number[n, :] = this[:]
                n += 1

            # sets last to this one
            last[:] = this[:]

            count += 1

    if n == 0:
        print("Reached the end of the file, exiting")
        exit()
    # concatenate max number to true size
    todo = max_number[0:n, :]
    starti = count - n + 1
    return todo, starti


def process_pfile(pfile):
    """ used for cube.py to input multi-dimensional grid to iterate over"""
    NPARAMS = 8
    mins = np.zeros([NPARAMS])
    maxs = np.zeros([NPARAMS])
    Nits = np.zeros([NPARAMS], dtype="int")
    with open(pfile) as pf:
        count = 0
        for line in pf:
            if count == NPARAMS:
                break
            vals = line.split()
            mins[count] = float(vals[0])
            maxs[count] = float(vals[1])
            Nits[count] = int(vals[2])
            count += 1
    return mins, maxs, Nits

