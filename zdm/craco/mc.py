"""
This script generates MC samples for CRAFT CRACO.

"""


from pkg_resources import resource_filename
import os
import copy
import numpy as np
import time


from zdm import misc_functions
from zdm import iteration as it
from zdm import misc_functions
from zdm.craco import loading

from IPython import embed

import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams["image.interpolation"] = None

defaultsize = 14
ds = 4
font = {"family": "normal", "weight": "normal", "size": defaultsize}
matplotlib.rc("font", **font)


def generate(
    alpha_method=1,
    Nsamples=10000,
    do_plots=True,
    lum_func=0,
    base_survey="CRAFT_CRACO_MC_base",
    outfile="FRBs.txt",
    savefile=None,
    update_params=None,
    plotfile="MC_Plots/mc_frbs_best_zdm_grid.pdf",
):

    craco, grid = loading.survey_and_grid(
        alpha_method=alpha_method, survey_name=base_survey, lum_func=lum_func
    )

    if update_params is not None:
        grid.update(update_params)

    print("Generating ", Nsamples, " samples from CRACO survey/grid ")
    sample = grid.GenMCSample(Nsamples)
    sample = np.array(sample)
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
    do_basic_sample_plots(sample, opdir="MC_Plots")

    # Read base
    sdir = os.path.join(resource_filename("zdm", "craco"), "MC_Surveys")
    basefile = os.path.join(sdir, base_survey + ".dat")
    with open(basefile, "r") as f:
        base_lines = f.readlines()

    # Write FRBs to disk
    add_header = True
    with open(outfile, "w") as f:
        # Header
        for base_line in base_lines:
            if add_header:
                if "NFRB" in base_line:
                    f.write(f"NFRB {Nsamples}\n")
                elif "NORM_FRB" in base_line:
                    f.write(f"NORM_FRB {Nsamples}\n")
                else:
                    f.write(base_line)
                if "fake data" in base_line:
                    add_header = False
        #
        for i in np.arange(Nsamples):
            DMG = 35
            DMEG = sample[i, 1]
            DMtot = DMEG + DMG + grid.state.MW.DMhalo
            SNRTHRESH = 9.5
            SNR = SNRTHRESH * sample[i, 4]
            z = sample[i, 0]
            w = sample[i, 3]

            string = (
                "FRB "
                + str(i)
                + "  {:6.1f}  35   {:6.1f}  {:5.3f}   {:5.1f}  {:5.1f} \n".format(
                    DMtot, DMEG, z, SNR, w
                )
            )
            # print("FRB ",i,DMtot,SNR,DMEG,w)
            f.write(string)
    print(f"Wrote: {outfile}")

    # Write state
    state_file = outfile.replace(".dat", "_state.json")
    grid.state.write(state_file)
    print(f"Wrote: {state_file}")
    # evaluate_mc_sample_v1(g,s,pset,sample)
    # evaluate_mc_sample_v2(g,s,pset,sample)


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

# Made in May 2022
# generate(
#     alpha_method=1,
#     lum_func=2,
#     Nsamples=5000,
#     do_plots=True,
#     outfile="MC_Surveys/CRACO_std_May2022.dat",
#     savefile=None,
# )

# JB

# generate(
#     alpha_method=1,
#     lum_func=2,
#     Nsamples=1000,
#     do_plots=True,
#     outfile="MC_F/Surveys/F_vanilla_survey.dat",
#     plotfile="MC_F/Plots/F_vanilla.pdf",
#     savefile=None,
#     # update_params={"F": 0.01},
# )

# generate(
#     alpha_method=1,
#     lum_func=2,
#     Nsamples=1000,
#     do_plots=True,
#     outfile="MC_F/Surveys/F_0.01_survey.dat",
#     plotfile="MC_F/Plots/F_0.01.pdf",
#     savefile=None,
#     update_params={"F": 0.01},
# )

generate(
    alpha_method=1,
    lum_func=2,
    Nsamples=1000,
    do_plots=True,
    outfile="MC_F/Surveys/F_0.32_survey.dat",
    plotfile="MC_F/Plots/F_0.32.pdf",
    savefile=None,
    update_params={"F": 0.32},
)

# generate(
#     alpha_method=1,
#     lum_func=2,
#     Nsamples=1000,
#     do_plots=True,
#     outfile="MC_F/Surveys/F_0.7_survey.dat",
#     plotfile="MC_F/Plots/F_0.7.pdf",
#     savefile=None,
#     update_params={"F": 0.7},
# )

# generate(
#     alpha_method=1,
#     lum_func=2,
#     Nsamples=1000,
#     do_plots=True,
#     outfile="MC_F/Surveys/F_0.01_dmhost_suppressed_survey.dat",
#     plotfile="MC_F/Plots/F_0.01_dmhost_suppressed.pdf",
#     savefile=None,
#     update_params={"F": 0.01, "lmean": 1e-3, "lsigma": 0.1},
# )

# generate(
#     alpha_method=1,
#     lum_func=2,
#     Nsamples=1000,
#     do_plots=True,
#     outfile="MC_F/Surveys/F_0.9_dmhost_suppressed_survey.dat",
#     plotfile="MC_F/Plots/F_0.9_dmhost_suppressed.pdf",
#     savefile=None,
#     update_params={"F": 0.9, "lmean": 1e-3, "lsigma": 0.1},
# )

# generate(
#     alpha_method=1,
#     lum_func=2,
#     Nsamples=1000,
#     do_plots=True,
#     outfile="MC_F/Surveys/F_0.7_dmhost_suppressed_survey.dat",
#     plotfile="MC_F/Plots/F_0.7_dmhost_suppressed.pdf",
#     savefile=None,
#     update_params={"F": 0.7, "lmean": 1e-3, "lsigma": 0.1},
# )

# generate(
#     alpha_method=1,
#     lum_func=2,
#     Nsamples=1000,
#     do_plots=True,
#     outfile="MC_F/Surveys/F_vanilla_dmhost_suppressed_survey.dat",
#     plotfile="MC_F/Plots/F_vanilla_dmhost_suppressed.pdf",
#     savefile=None,
#     update_params={"lmean": 1e-3, "lsigma": 0.1},
# )
