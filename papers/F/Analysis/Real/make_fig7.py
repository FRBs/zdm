"""
This is a script used to produce figures for fig 7

It generates two sets of results:
- constraints on alpha (in directory fig8_alphaSingleFigures)
- constraints on other 5 non-H0 parameters (in directory fig_othersSingleFigures)

Alpha requires special treatment due to the prior not covering
the full range of possible values.
"""

import numpy as np
import os
import zdm
from zdm import analyze_cube as ac

from matplotlib import pyplot as plt
import scipy
from IPython import embed

def main(verbose=False):
    
    ######### other results ####
    Planck_H0 = 67.66
    Planck_sigma = 0.5
    Reiss_H0 = 73.04
    Reiss_sigma = 1.42
    
    # output directory
    opdir="Figure7/"
    if not os.path.exists(opdir):
        os.mkdir(opdir)
    
    CubeFile='Cubes/craco_real_cube.npz'
    if os.path.exists(CubeFile):
        data=np.load(CubeFile)
    else:
        print("Could not file cube output file ",CubeFile)
        print("Please obtain it from [repository]")
        exit()
    
    # builds uvals list
    uvals=[]
    latexnames=[]
    for ip,param in enumerate(data["params"]):
        # switches for alpha
        if param=="alpha":
            uvals.append(data[param]*-1.)
        else:
            uvals.append(data[param])
        if param=="alpha":
            latexnames.append('$\\alpha$')
            ialpha=ip
        elif param=="lEmax":
            latexnames.append('$\\log_{10} E_{\\rm max}$')
        elif param=="H0":
            latexnames.append('$H_0$')
        elif param=="gamma":
            latexnames.append('$\\gamma$')
        elif param=="sfr_n":
            latexnames.append('$n_{\\rm sfr}$')
        elif param=="lmean":
            latexnames.append('$\\mu_{\\rm host}$')
        elif param=="lsigma":
            latexnames.append('$\\sigma_{\\rm host}$')
        elif param=="logF":
            latexnames.append('$\\log_{10} F$')
    
    # 1D plots by surveys s only
    contributions=[data["lls0"],data["lls2"],data["lls3"],data["lls1"],data["lls4"]]
    labels=["CRAFT/FE","CRAFT/ICS 900 MHz","CRAFT/ICS 1.3 GHz","CRAFT/ICS 1.6 GHz","Parkes/Mb"] #correct
    
    colors=['blue','green','orange','purple','red']
    linestyles=['-',':','--','-','-.']
    make_1d_plots_by_contribution(data,contributions,labels,prefix="Figure7/by_survey_",
        colors=colors,linestyles=linestyles)#,latexnames=latexnames)
    exit()

def make_1d_plots_by_contribution(
    data,
    contributions,
    labels,
    prefix="",
    fig_exten=".png",
    log=False,
    splines=True,
    latexnames=None,
    units=None,
    linestyles=None,
    colors=None,
):
    """
    contributions: list of vectors giving various likelihood terms
    args:
        splines (bool): draw cubic splines
        Labels: lists labels stating what these are
        latexnames: latex for x and p(X)
        units: appends units to x axis but not p(X)
    """
    ######################### 1D plots, split by terms ################
    all_uvals = []
    all_vectors = []
    all_wvectors = []
    combined = data["pzDM"] + data["pDM"]

    # gets 1D Bayesian curves for each contribution
    for datatype in contributions:
        uvals, vectors, wvectors = ac.get_bayesian_data(datatype)
        all_uvals.append(uvals)
        all_vectors.append(vectors)
        all_wvectors.append(wvectors)
    params = data["params"]

    # gets unique values for each axis
    param_vals = []
    param_list = [
        data["H0"],
        data["lmean"],
        data["lsigma"],
        data["logF"]
    ]
    xlatexnames = [
        "H_0 {\\rm [km\,s^{-1}\,Mpc^{-1}]}",
        "\\mu_{\\rm host} {\\rm [pc\,cm^{-3}]}",
        "\\sigma_{\\rm host}",
        "\\log_{10} F",
    ]
    ylatexnames = [
        "H_0",
        "\\mu_{\\rm host}",
        "\\sigma_{\\rm host}",
        "\\log_{10} F",
    ]

    for col in param_list:
        unique = np.unique(col)
        param_vals.append(unique)
    # assigns different plotting styles to help distinguish curves
    if linestyles is None:
        linestyles = ["-", "--", "-.", ":", "-", "--", "-.", ":", "-", "--", "-.", ":"]
    if colors is None:
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for which in np.arange(len(param_list)):
        plt.figure()
        plt.xlabel("$" + xlatexnames[which] + "$")
        plt.ylabel("$p(" + ylatexnames[which] + ")$")
        xvals = param_vals[which]

        for idata, vectors in enumerate(all_vectors):
            # print(which, idata, len(vectors[which]), len(xvals))
            if splines:
                xdata = np.linspace(xvals[0], xvals[-1], 100)
                f = scipy.interpolate.interp1d(
                    xvals, np.log(vectors[which]), kind="cubic"
                )
                ydata = np.exp(f(xdata))
                plt.plot(
                    xdata,
                    ydata,
                    label=labels[idata],
                    linestyle=linestyles[idata],
                    color=colors[idata],
                )
                plt.scatter(
                    xvals, vectors[which], color=plt.gca().lines[-1].get_color()
                )
            else:
                ydata = vectors[which]
                xdata = xvals
                # print(labels[idata]," has values ",vector)
                plt.plot(
                    xdata,
                    ydata,
                    label=labels[idata],
                    linestyle=linestyles[idata],
                    color=colors[idata],
                )

        if log:
            plt.yscale("log")
            # plt.ylim(np.max(vector)*1e-3,np.max(vector)) #improve this
        plt.legend()
        plt.savefig(prefix + params[which] + fig_exten, dpi=200)
        plt.close()

main()
