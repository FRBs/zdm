"""
This is a script to produce limit plots for a cube with priors on F

The plots produced with priors on F are stored in folders prefixed with "wF".
Plots generated with the synthetic CRACO cube are suffixed with "_forecast", while 
plots generated with the real observation cube are suffixed with "_measured".
Plots showing PDFs with and without priors are infixed with "others".

- The priors on F are:
    a) a Gaussian prior (with 20% error on F)
    b) No prior

"""

import numpy as np
import os
import zdm
from zdm import analyze_cube as ac

from matplotlib import pyplot as plt


def main(cube_path, outdir="./", verbose=False):
    ######### sets the values of F for priors #####
    F_0 = np.log10(0.32)
    F_sigma = np.abs(0.2 * F_0)  # error of 20% on F

    data = np.load(cube_path)
    if verbose:
        for thing in data:
            print(thing)
        print(data["params"])

    # gets values of cube parameters
    # param_vals=get_param_values(data,verbose)

    # gets latex names
    uvals, latexnames = get_names_values(data)

    ################ single plots, no priors ############
    deprecated, uw_vectors, wvectors = ac.get_bayesian_data(data["ll"])
    ac.do_single_plots(
        uvals,
        uw_vectors,
        None,
        data["params"],
        tag="",
        log=False,
        logspline=False,
        # kind="linear",
        truth=None,
        dolevels=True,
        latexnames=latexnames,
        outdir=outdir,
    )

    ########### F data for fixed values of other parameters ###########
    # extracts best-fit values
    list1 = []
    vals1 = []
    list2 = []
    vals2 = []
    vals3 = []
    for i, vec in enumerate(uw_vectors):
        n = np.argmax(vec)  # selects the most likely value
        val = uvals[i][n]
        if data["params"][i] == "logF":
            # enables us to select a slice corresponding to particular F values
            list1.append(data["params"][i])
            vals1.append(F_0)
            iF = i  # setting index for F param
        else:
            # enables us to select a slice correspondng to the best-fit values of all other params
            # i.e. ignoring uncertainty in them
            list2.append(data["params"][i])
            vals2.append(val)

    # gets the slice corresponding to specific values of F
    F_0_selection = ac.get_slice_from_parameters(data, list1, vals1, verbose=True)

    # will have Bayesian limits on all parameters over everything but F
    deprecated, F_vectors, deprecated = ac.get_bayesian_data(F_0_selection)

    ####### 1D plots for prior on F ########
    # generates plots for our standard prior on F only
    # applies a prior on F, which is a Gaussian
    F_dim = np.where(data["params"] == "logF")[0][0]

    wlls = ac.apply_F_prior(data["ll"], F_dim, data["logF"], F_0, F_sigma)

    deprecated, wF_vectors, wvectors = ac.get_bayesian_data(wlls)

    ac.do_single_plots(
        uvals,
        wF_vectors,
        None,
        data["params"],
        tag="wF_",
        truth=None,
        dolevels=True,
        latexnames=latexnames,
        logspline=False,
        outdir=outdir,
    )

    # now do this with others...
    # builds others...
    others = []
    for i, p in enumerate(data["params"]):
        if i == iF:
            oset = None
            others.append(oset)
        else:
            if i < iF:
                modi = i
            else:
                modi = i - 1
            oset = [uw_vectors[i]]
            others.append(oset)

    # generates plots for our standard prior on F values, and no prior also
    ac.do_single_plots(
        uvals,
        wF_vectors,
        None,
        data["params"],
        tag="wF_others_",
        truth=None,
        dolevels=True,
        latexnames=latexnames,
        logspline=False,
        others=others,
        outdir=outdir,
        others_labels=["No prior", "Prior on $F$"],
    )


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


def get_param_values(data, verbose=False):
    """
    Returns the unique cube values for each parameter in the cube

    Input:
        data cube (tuple from reading the .npz)

    Output:
        list of numpy arrays for each parameter giving their values
    """
    # gets unique values for each axis
    param_vals = []

    # for col in param_list:
    for col in data["params"]:
        # unique=np.unique(col)
        unique = np.unique(data[col])
        param_vals.append(unique)
        if verbose:
            print("For parameter ", col, " cube values are ", unique)
    return param_vals


# Real Cube Data
main("../Real/Cubes/craco_real_cube.npz", "measured/")
main("../CRACO/Cubes/craco_full_cube.npz", "forecast/")
