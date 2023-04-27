import numpy as np
import os
import zdm
import scipy
from zdm import analyze_cube as ac
from IPython import embed
import matplotlib.pyplot as plt


def main():
    # cube_old = "../Real/Cubes/craco_real_old_cube.npz"
    cube_old = "../CRACO/Cubes/craco_full_cube.npz"
    cube = "../Real/Cubes/craco_real_cube.npz"

    # old cube
    funcs, interp_mins, interp_maxs = getlogPDFs(cube_old)
    _, _, _, flogF = funcs
    _, _, _, logF_min = interp_mins
    _, _, _, logF_max = interp_maxs
    res = 1e3
    thresh = 1e-3
    logFs = np.linspace(logF_min, logF_max, int(res))

    probs_old = np.exp(flogF(logFs))
    max_prob_old = np.max(probs_old)
    max_F_old = logFs[np.argmax(probs_old)]

    sort_idx_old = np.argsort(np.abs(probs_old - (max_prob_old / 2)))
    half_max_Fs_old = logFs[sort_idx_old[1]]

    # new cube
    funcs, interp_mins, interp_maxs = getlogPDFs(cube)
    _, _, _, flogF = funcs
    _, _, _, logF_min = interp_mins
    _, _, _, logF_max = interp_maxs

    probs = np.exp(flogF(logFs))
    max_prob = np.max(probs)
    max_F = logFs[np.argmax(probs)]

    sort_idx = np.argsort(np.abs(probs - (max_prob / 2)))
    half_max_Fs = logFs[sort_idx[3]]

    print("debugging", logFs[sort_idx])

    # Left-sided half width half max
    print(
        "Left-sided half-width half-max for old cube: ",
        np.abs(max_F_old - np.min(half_max_Fs_old)),
    )
    print(
        "Left-sided half-width half-max for new cube: ",
        np.abs(max_F - np.min(half_max_Fs)),
    )

    plt.figure()
    # old cube
    plt.plot(logFs, probs_old, color="red", label="Before 2022 FRBs")
    plt.scatter(half_max_Fs_old, probs_old[sort_idx_old[1]])
    # plt.scatter(max_F_old, max_prob_old)
    plt.axhline(max_prob_old / 2, color="red", alpha=0.5, ls="--")

    # new cube

    plt.plot(logFs, probs, color="blue", label="After 2022 FRBs")
    plt.scatter(half_max_Fs, probs[sort_idx[3]])
    # plt.scatter(max_F, max_prob)
    plt.axhline(max_prob / 2, color="blue", alpha=0.5, ls="--")

    plt.legend()

    plt.xlabel(r"$\log_{10} F$")
    plt.ylabel(r"$p(\log_{10} F)$")

    plt.text(
        0.05,
        0.75,
        f"LWHM (old) = {np.round(np.abs(max_F_old - np.min(half_max_Fs_old)), 3)} \
              \nLWHM (new) = {np.round(np.abs(max_F - np.min(half_max_Fs)), 3)}",
        transform=plt.gca().transAxes,
    )

    plt.savefig("diagnostic_2.png")
    plt.close()

    # embed(header="end of main")
    #


def getlogPDFs(cube):
    ######### sets the values of H0 for priors #####
    Planck_H0 = 67.4
    Planck_sigma = 0.5
    Reiss_H0 = 73.04
    Reiss_sigma = 1.42
    #########

    data = np.load(cube)

    uvals, latexnames = get_names_values(data)

    H0_dim = np.where(data["params"] == "H0")[0][0]
    wlls = ac.apply_H0_prior(
        data["ll"], H0_dim, data["H0"], Planck_H0, Planck_sigma, Reiss_H0, Reiss_sigma
    )
    deprecated, wH0_vectors, wvectors = ac.get_bayesian_data(wlls)

    nonnorm_pdfs = []
    pdf_mins = []
    pdf_maxs = []

    for i, vals in enumerate(uvals):
        ymax = np.max(wH0_vectors[i])
        temp = np.where((wH0_vectors[i] > 0.0) & (np.isfinite(wH0_vectors[i])))

        f = scipy.interpolate.interp1d(
            vals[temp], np.log(wH0_vectors[i][temp]), kind="cubic"
        )
        # remember to exponentiate output from f
        nonnorm_pdfs.append(f)
        pdf_mins.append(np.min(vals[temp]))
        pdf_maxs.append(np.max(vals[temp]))

    return nonnorm_pdfs, pdf_mins, pdf_maxs


def get_names_values(data):
    """
    Gets a list of latex names and corrected parameter values
    """
    # builds uvals list
    uvals = []
    latexnames = []
    for ip, param in enumerate(data["params"]):
        # switches for alpha
        if param == "alpha":
            uvals.append(data[param] * -1.0)
        else:
            uvals.append(data[param])
        if param == "alpha":
            latexnames.append("$\\alpha$")
            ialpha = ip
        elif param == "lEmax":
            latexnames.append("$\\log_{10} E_{\\rm max}$")
        elif param == "H0":
            latexnames.append("$H_0$")
        elif param == "gamma":
            latexnames.append("$\\gamma$")
        elif param == "sfr_n":
            latexnames.append("$n_{\\rm sfr}$")
        elif param == "lmean":
            latexnames.append("$\\mu_{\\rm host}$")
        elif param == "lsigma":
            latexnames.append("$\\sigma_{\\rm host}$")
        elif param == "logF":
            latexnames.append("$\\log F$")
    return uvals, latexnames


main()
