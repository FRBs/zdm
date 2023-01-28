"""
This is a script to produce limit plots for a cube

It produces three sets of plots:
- single parameter limits with a prior on H0 between Planck and SN1a values
- single parameter limits also showing results with priors on H0 equal to:
    a) Planck
    b) Reiss
    c) No prior
- 2D correlation plots with no prior onH0

It also collects data to plot a result on H0 for best-fit values of all
other parameters, but currently does not produce that plot

"""

import numpy as np
import os
import zdm
from zdm import analyze_cube as ac

from matplotlib import pyplot as plt


def main(verbose=False):

    ######### sets the values of H0 for priors #####
    Planck_H0 = 67.4
    Planck_sigma = 0.5
    Reiss_H0 = 73.04
    Reiss_sigma = 1.42

    ##### loads cube data #####
    # cube = "../../papers/F/Analysis/Real/Cubes/craco_real_cube.npz"
    cube = "../../papers/F/Analysis/CRACO/Cubes/craco_full_cube.npz"
    data = np.load(cube)
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
    )

    ########### H0 data for fixed values of other parameters ###########
    # extracts best-fit values
    list1 = []
    vals1 = []
    list2 = []
    vals2 = []
    vals3 = []
    for i, vec in enumerate(uw_vectors):
        n = np.argmax(vec)  # selects the most likely value
        val = uvals[i][n]
        if data["params"][i] == "H0":
            # enables us to select a slice corresponding to particular H0 values
            list1.append(data["params"][i])
            vals1.append(Reiss_H0)

            vals3.append(Planck_H0)

            iH0 = i  # setting index for Hubble
        else:
            # enables us to select a slice correspondng to the best-fit values of all other params
            # i.e. ignoring uncertainty in them
            list2.append(data["params"][i])
            vals2.append(val)

    # gets the slice corresponding to specific values of H0
    Reiss_H0_selection = ac.get_slice_from_parameters(data, list1, vals1, verbose=True)
    Planck_H0_selection = ac.get_slice_from_parameters(data, list1, vals3, verbose=True)

    # will have Bayesian limits on all parameters over everything but H0
    deprecated, ReissH0_vectors, deprecated = ac.get_bayesian_data(Reiss_H0_selection)
    deprecated, PlanckH0_vectors, deprecated = ac.get_bayesian_data(Planck_H0_selection)

    # gets the slice corresponding to the best-fit values of all other parameters
    # this is 1D, so is our limit on H0 keeping all others fixed
    pH0_fixed = ac.get_slice_from_parameters(data, list2, vals2)

    ####### 1D plots for prior on H0 ########
    # generates plots for our standard prior on H0 only
    # applies a prior on H0, which is flat between systematic differences, then falls off as a Gaussian either side
    H0_dim = np.where(data["params"] == "H0")[0][0]
    wlls = ac.apply_H0_prior(
        data["ll"], H0_dim, data["H0"], Planck_H0, Planck_sigma, Reiss_H0, Reiss_sigma
    )
    deprecated, wH0_vectors, wvectors = ac.get_bayesian_data(wlls)
    ac.do_single_plots(
        uvals,
        wH0_vectors,
        None,
        data["params"],
        tag="wH0_",
        truth=None,
        dolevels=True,
        latexnames=latexnames,
        logspline=False,
    )

    # now do this with others...
    # builds others...
    others = []
    for i, p in enumerate(data["params"]):
        if i == iH0:
            oset = None
            others.append(oset)
        else:
            if i < iH0:
                modi = i
            else:
                modi = i - 1
            oset = [uw_vectors[i], ReissH0_vectors[modi], PlanckH0_vectors[modi]]
            others.append(oset)

    # generates plots for our standard prior on H0, Planck and SN1a values, and no prior also
    ac.do_single_plots(
        uvals,
        wH0_vectors,
        None,
        data["params"],
        tag="wH0_others_",
        truth=None,
        dolevels=True,
        latexnames=latexnames,
        logspline=False,
        others=others,
    )

    ############## 2D plots for total likelihood ###########
    # these are for nor priors on anything
    baduvals, ijs, arrays, warrays = ac.get_2D_bayesian_data(data["ll"])
    for which, array in enumerate(arrays):

        i = ijs[which][0]
        j = ijs[which][1]

        savename = "2D/lls_" + data["params"][i] + "_" + data["params"][j] + ".png"
        ac.make_2d_plot(
            array, latexnames[i], latexnames[j], uvals[i], uvals[j], savename=savename
        )
        # ac.make_2d_plot(array,latexnames[i],latexnames[j],
        #    param_vals[i],param_vals[j],
        #    savename=savename)
        if False and data["params"][i] == "H0":
            savename = (
                "normed2D/lls_" + data["params"][j] + "_" + data["params"][i] + ".png"
            )

            ac.make_2d_plot(
                array.T,
                latexnames[j],
                latexnames[i],
                uvals[j],
                uvals[i],
                savename=savename,
                norm=1,
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


main()
