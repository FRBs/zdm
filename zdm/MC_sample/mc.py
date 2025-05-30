"""
This script generates MC samples

"""


from pkg_resources import resource_filename
import os
import copy
import numpy as np
import time

from zdm import misc_functions
from zdm import iteration as it
from zdm import loading

# from ne2001 import density

from astropy.cosmology import Planck18
from zdm import parameters

from astropy.table import Table

from IPython import embed

import matplotlib.pyplot as plt
import matplotlib

import json

# matplotlib.rcParams["image.interpolation"] = None

defaultsize = 14
ds = 4
font = {"family": "normal", "weight": "normal", "size": defaultsize}
matplotlib.rc("font", **font)


def generate(
    Nsamples=10000,
    do_plots=False,
    base_survey="CRAFT_CRACO_MC_base",
    outfile="FRBs",
    savefile=None,
    param_dict=None,
    meta_data=None,
    plotfile="MC_Plots/mc_frbs_best_zdm_grid.pdf",
):

    state = parameters.State()
    state.set_astropy_cosmo(Planck18)
    state.update_params(param_dict)

    surveys, grids = loading.surveys_and_grids(survey_names=[base_survey], init_state=state, discard_empty=False)
    survey = surveys[0]
    grid = grids[0]

    print("Generating ", Nsamples, " samples from ", base_survey)
    sample = grid.GenMCSample(Nsamples)

    if savefile is not None:
        np.save(savefile, sample)

    if do_plots:
        Zs = sample[:, 0]
        DMEGs = sample[:, 1]
        misc_functions.plot_grid_2(
            grid.rates,
            grid.zvals,
            grid.dmvals,
            # FRBZ=craco.frbs["Z"],FRBDM=craco.DMEGs,
            FRBZ=Zs,
            FRBDM=DMEGs,
            zmax=1.8,
            DMmax=2000,
            name=plotfile,
            norm=2,
            log=True,
            label="$\\log_{10} p({\\rm DM}_{\\rm EG},z)$",
            project=False,
            Aconts=[0.01, 0.1, 0.5],
            Macquart=grid.state,
        )

    # sample
    # 0: z
    # 1: DM
    # 2: b
    # 3: w
    # 4: s
    # plot some sample plots
        # do_basic_sample_plots(sample, opdir="MC_Plots")

    # Generate DMG
    # Gb: -90 to 90
    # Gl: 0 to 360
    Gl = np.random.uniform(0, 360, Nsamples)
    Gb = np.random.uniform(0, 1, Nsamples)
    Gb = np.arcsin(2*Gb - 1) / np.pi * 180
    
    # ne = density.ElectronDensity()
    DMGs = np.zeros(Nsamples)
    # for i,l in enumerate(Gl):
    #     b=Gb[i]
    #     ismDM = ne.DM(l, b, 100.)
    #     DMGs[i] = ismDM.value

    write_samples(sample, survey, outfile, DMGs, Gb, Gl)

def write_samples(sample, survey, outfile, DMGs, Gb, Gl):
    Nsamples = sample.shape[0]
    
    # Do meta data for survey
    t = Table()
    t.meta = meta_data
    t.meta['survey_data'] = json.dumps(t.meta['survey_data'])

    # Format Table
    t['TNS'] = np.array(range(Nsamples)).astype(str)
    t['BW'] = survey.meta['BW'] * np.ones(Nsamples)
    t['DM'] = DMGs + sample[:,1] + survey.DMhalo
    t['DMG'] = DMGs
    t['FBAR'] = survey.meta['FBAR'] * np.ones(Nsamples)
    t['FRES'] = survey.meta['FRES'] * np.ones(Nsamples)
    t['Gb'] = Gb
    t['Gl'] = Gl
    t['SNR'] = sample[:,3] * survey.meta['SNRTHRESH']
    t['SNRTHRESH'] = survey.meta['SNRTHRESH'] * np.ones(Nsamples)
    t['THRESH'] = survey.meta['THRESH'] * np.ones(Nsamples)
    t['TRES'] = survey.meta['TRES'] * np.ones(Nsamples)
    t['WIDTH'] = sample[:,4]
    t['XDec'] = np.array(["" for i in range(Nsamples)])
    t['XRA'] = np.array(["" for i in range(Nsamples)])
    t['Z'] = sample[:,0]
    t['DMEG'] = sample[:,1]
    t['B'] = sample[:,2]

    # Write to file
    t.write(outfile + '.ecsv', format='ascii.ecsv')


def evaluate_mc_sample_v1(grid, survey, pset, sample, opdir="Plots"):
    """
    Evaluates the likelihoods for an MC sample of events
    Simply replaces individual sets of z, DM, s with MC sets
    Will produce a plot of Nsamples/NFRB pseudo datasets.
    """
    t0 = time.process_time()

    nsamples = sample.shape[0]

    # get number of FRBs per sample
    Npersurvey = survey.NFRB
    # determines how many false surveys we have stats for
    Nsurveys = int(nsamples / Npersurvey)

    print(
        "We can evaluate ",
        Nsurveys,
        "MC surveys given a total of ",
        nsamples,
        " and ",
        Npersurvey,
        " FRBs in the original data",
    )

    # makes a deep copy of the survey
    s = copy.deepcopy(survey)

    lls = []
    # Data order is DM,z,b,w,s
    # we loop through, artificially altering the survey with the composite values.
    for i in np.arange(Nsurveys):
        this_sample = sample[i * Npersurvey : (i + 1) * Npersurvey, :]
        s.DMEGs = this_sample[:, 0]
        s.Ss = this_sample[:, 4]
        if s.nD == 1:  # DM, snr only
            ll = it.calc_likelihoods_1D(grid, s, pset, psnr=True, Pn=True, dolist=0)
        else:
            s.Zs = this_sample[:, 1]
            ll = it.calc_likelihoods_2D(grid, s, pset, psnr=True, Pn=True, dolist=0)
        lls.append(ll)
    t1 = time.process_time()
    dt = t1 - t0
    print("Finished after ", dt, " seconds")

    lls = np.array(lls)

    plt.figure()
    plt.hist(lls, bins=20)
    plt.xlabel("log likelihoods [log10]")
    plt.ylabel("p(ll)")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(opdir + "/ll_histogram.pdf")
    plt.close()


def evaluate_mc_sample_v2(grid, survey, pset, sample, opdir="Plots", Nsubsamp=1000):
    """
    Evaluates the likelihoods for an MC sample of events
    First, gets likelihoods for entire set of FRBs
    Then re-samples as needed, a total of Nsubsamp times
    """
    t0 = time.process_time()

    nsamples = sample.shape[0]

    # makes a deep copy of the survey
    s = copy.deepcopy(survey)
    NFRBs = s.NFRB

    s.NFRB = nsamples  # NOTE: does NOT change the assumed normalised FRB total!
    s.DMEGs = sample[:, 1]
    s.Ss = sample[:, 4]
    if s.nD == 1:  # DM, snr only
        llsum, lllist, expected, longlist = it.calc_likelihoods_1D(
            grid, s, pset, psnr=True, Pn=True, dolist=2
        )
    else:
        s.Zs = sample[:, 0]
        llsum, lllist, expected, longlist = it.calc_likelihoods_2D(
            grid, s, pset, psnr=True, Pn=True, dolist=2
        )

    # we should preserve the normalisation factor for Tobs from lllist
    Pzdm, Pn, Psnr = lllist

    # plots histogram of individual FRB likelihoods including Psnr and Pzdm
    plt.figure()
    plt.hist(longlist, bins=100)
    plt.xlabel("Individual Psnr,Pzdm log likelihoods [log10]")
    plt.ylabel("p(ll)")
    plt.tight_layout()
    plt.savefig(opdir + "/individual_ll_histogram.pdf")
    plt.close()

    # generates many sub-samples of the data
    lltots = []
    for i in np.arange(Nsubsamp):
        thislist = np.random.choice(
            longlist, NFRBs
        )  # samples with replacement, by default
        lltot = Pn + np.sum(thislist)
        lltots.append(lltot)

    plt.figure()
    plt.hist(lltots, bins=20)
    plt.xlabel("log likelihoods [log10]")
    plt.ylabel("p(ll)")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(opdir + "/sampled_ll_histogram.pdf")
    plt.close()

    t1 = time.process_time()
    dt = t1 - t0
    print("Finished after ", dt, " seconds")


def do_basic_sample_plots(sample, opdir="Plots"):
    """
    Data order is DM,z,b,w,s
    
    """
    if not os.path.exists(opdir):
        os.mkdir(opdir)
    zs = sample[:, 0]
    DMs = sample[:, 1]
    plt.figure()
    plt.hist(DMs, bins=100)
    plt.xlabel("DM")
    plt.ylabel("Sampled DMs")
    plt.tight_layout()
    plt.savefig(opdir + "/DM_histogram.pdf")
    plt.close()

    plt.figure()
    plt.hist(zs, bins=100)
    plt.xlabel("z")
    plt.ylabel("Sampled redshifts")
    plt.tight_layout()
    plt.savefig(opdir + "/z_histogram.pdf")
    plt.close()

    bs = sample[:, 2]
    plt.figure()
    plt.hist(np.log10(bs), bins=5)
    plt.xlabel("log10 beam value")
    plt.yscale("log")
    plt.ylabel("Sampled beam bin")
    plt.tight_layout()
    plt.savefig(opdir + "/b_histogram.pdf")
    plt.close()

    ws = sample[:, 3]
    plt.figure()
    plt.hist(ws, bins=5)
    plt.xlabel("width bin (not actual width!)")
    plt.ylabel("Sampled width bin")
    plt.yscale("log")
    plt.tight_layout()
    plt.savefig(opdir + "/w_histogram.pdf")
    plt.close()

    s = sample[:, 4]
    plt.figure()
    plt.hist(np.log10(s), bins=100)
    plt.xlabel("$\\log_{10} (s={\\rm SNR}/{\\rm SNR}_{\\rm th})$")
    plt.yscale("log")
    plt.ylabel("Sampled $s$")
    plt.tight_layout()
    plt.savefig(opdir + "/s_histogram.pdf")
    plt.close()


# Generate em!

# Default run with Planck18
# generate(alpha_method=1, Nsamples=5000, do_plots=True,
#    outfile='MC_Surveys/CRACO_alpha1_Planck18.dat',
#    savefile=None)

"""  Run in February 2022
# Gamma function for energies
generate(alpha_method=1, lum_func=2, Nsamples=5000, do_plots=True,
    outfile='MC_Surveys/CRACO_alpha1_Planck18_Gamma.dat',
    savefile=None)
"""

param_dict={'sfr_n': 1.13, 'alpha': 1.5, 'lmean': 2.27, 'lsigma': 0.55, 'lEmax': 41.26, 'lEmin': 39.5, 'gamma': -0.95, 'H0': 73}
meta_data = {}
meta_data['survey_data'] = {}

# # DSA
# meta_data['survey_data']['observing'] = {"MAX_DM": 1500}
# meta_data['survey_data']['telescope'] = {"BMETHOD": 0, "BTHRESH": 0.001, "DIAM": 4.65, "NBEAMS": 1, "NBINS": 10}

# CRAFT_ICS_1300
meta_data['survey_data']['observing'] = {"MAX_IDT": 4096}
meta_data['survey_data']['telescope'] = {"BEAM": "ASKAP_1300", "DIAM": 12.0, "NBEAMS": 36, "NBINS": 5}

t0 = time.time()
generate(
    Nsamples=100000,
    do_plots=False,
    base_survey="CRAFT_ICS_1300",
    outfile="MC_CRAFT_ICS_1300_e5",
    savefile=None,
    param_dict=param_dict,
    meta_data=meta_data,
)
print(time.time() - t0)
    