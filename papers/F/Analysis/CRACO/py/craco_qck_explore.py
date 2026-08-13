"""
Generates 1D likelihood PDFs of each parameter from a `.npz` cube.

The only argument in running the file corresponds to a hard-coded location of the `.npz` cube file. 
"""
# imports
from importlib import reload
import numpy as np
import sys, os

from zdm import analyze_cube
from zdm import iteration as it
from zdm import io
from zdm.MC_sample import loading

from IPython import embed


# sys.path.append(os.path.abspath("../../Figures/py"))


def main(pargs):
    jroot = None

    if pargs.run == "F":
    # 2D cube run with H0 and F
        scube = "H0_F"
        outdir = "H0_F/"
    elif pargs.run == "H0_logF":
    # 2D cube run with H0 and logF
        scube = "H0_logF"
        outdir = "H0_logF/"
    # Main #
    elif pargs.run == "logF_full":
    # Full CRACO likelihood cube
        scube = "full"
        outdir = "logF_Full/"

    if jroot is None:
        jroot = scube

    # Load
    npdict = np.load(f"Cubes/craco_{scube}_cube.npz")

    ll_cube = npdict["ll"]

    # Deal with Nan
    ll_cube[np.isnan(ll_cube)] = -1e99
    params = npdict["params"]

    # Cube parameters
    ############## Load up ##############
    pfile = f"Cubes/craco_{jroot}_cube.json"
    input_dict = io.process_jfile(pfile)

    # Deconstruct the input_dict
    state_dict, cube_dict, vparam_dict = it.parse_input_dict(input_dict)

    latexnames = []
    for ip, param in enumerate(npdict["params"]):
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
            latexnames.append("$\\log_{10} F$")

    units = []
    for ip, param in enumerate(npdict["params"]):
        if param == "alpha":
            units.append(" ")
            ialpha = ip
        elif param == "lEmax":
            units.append("[$\rm erg$]")
        elif param == "H0":
            units.append(r"[$\rm km \, s^{-1} \, Mpc^{-1}$]")
        elif param == "gamma":
            units.append("")
        elif param == "sfr_n":
            units.append(" ")
        elif param == "lmean":
            units.append(r"[$\rm pc \, cm^{-3}$]")
        elif param == "lsigma":
            units.append(r"[$\rm pc \, cm^{-3}$]")
        elif param == "logF":
            units.append(" ")

    # Run Bayes

    # Offset by max
    ll_cube = ll_cube - np.max(ll_cube)

    uvals, vectors, wvectors = analyze_cube.get_bayesian_data(ll_cube)

    analyze_cube.do_single_plots(
        uvals,
        vectors,
        None,
        params,
        vparams_dict=vparam_dict,
        outdir=outdir,
        compact=True,
        latexnames=latexnames,
        units=units,
        dolevels=True,
    )
    print(f"Wrote figures to {outdir}")


def parse_option():
    """
    This is a function used to parse the arguments in the training.

    Returns:
        args: (dict) dictionary of the arguments.
    """
    import argparse

    parser = argparse.ArgumentParser("Slurping the cubes")
    parser.add_argument("run", type=str, help="Run to slurp")
    # parser.add_argument('--debug', default=False, action='store_true',
    #                    help='Debug?')
    args = parser.parse_args()

    return args


# Command line execution
if __name__ == "__main__":
    pargs = parse_option()
    main(pargs)

#  python py/craco_qck_explore.py logF_full