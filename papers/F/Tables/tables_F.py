# Module for Tables for the Baptista+23 paper
# Imports
import numpy as np
import os, sys
import pandas


from zdm.craco import loading
from zdm import survey
from zdm import parameters
from zdm import misc_functions
from zdm import io
from zdm import iteration as it

from IPython import embed

# Local
sys.path.append(os.path.abspath("../Analysis/py"))
import analy_F_I
import pandas as pd


def mktab_model_params(outfile="tab_model_params.tex", sub=False):
    isurvey, grid = analy_F_I.craco_mc_survey_grid()
    tb = pd.read_json("../Analysis/CRACO/Cubes/craco_full_cube.json")

    # Open
    tbfil = open(outfile, "w")

    # Header
    tbfil.write("\\begin{deluxetable}{cccccc} \n")
    tbfil.write("\\tablewidth{20pt} \n")
    tbfil.write("\\tablecaption{z-DM grid parameters \label{tab:fullcube}} \n")
    tbfil.write("\\tabletypesize{\\normalsize} \n")
    tbfil.write(
        "\\tablehead{ \colhead{Parameter} & \n \colhead{Unit} & \n \colhead{Fiducial} & \n \colhead{Min} & \n \colhead{Max} & \n \colhead{N} }"
    )
    tbfil.write("\\startdata \n")

    params_vary = ["H0", "logF", "lmean", "lsigma"]
    params_fix = ["alpha", "gamma", "sfr_n", "lEmax"]

    for key in params_vary:
        item = grid.state.params[key]
        latex = getattr(grid.state, item).meta(key)["Notation"]
        min_val = tb[key]["min"]
        max_val = tb[key]["max"]
        n_val = tb[key]["n"]

        line = ""
        # Name
        line += f"${latex}$ &"
        # Unit
        line += f"{getattr(grid.state, item).meta(key)['unit']} &"
        # Fiducial
        line += f"{getattr(getattr(grid.state, item), key):.2f} &"
        # Min
        line += f"{min_val:.2f} &"
        # Max
        line += f"{max_val:.2f} &"
        # N
        line += f"{n_val} \\\\ \n"
        tbfil.write(line)

    for key in params_fix:
        item = grid.state.params[key]
        latex = getattr(grid.state, item).meta(key)["Notation"]

        line = ""
        # Name
        line += f"${latex}$ &"
        # Unit
        line += f"{getattr(grid.state, item).meta(key)['unit']} &"
        # Fiducial
        line += f"{getattr(getattr(grid.state, item), key):.2f} &"
        # Min
        line += f"-- &"
        # Max
        line += f"-- &"
        # N
        line += f"-- \\\\ \n"
        tbfil.write(line)

    tbfil.write("\\hline \n")
    tbfil.write("\\enddata \n")
    tbfil.write(
        "\\tablecomments{This table indicates the parameters of the high-resolution grid run. Non-degenerate parameters are held to the fiducial values. $N$ is the number of cells between the minimum and maximum parameter values.} \n"
    )
    tbfil.write("\\end{deluxetable} \n")
    tbfil.close()

    print("Wrote {:s}".format(outfile))


def mktex_measurements(outfile="results.tex"):
    # Open
    tbfil = open(outfile, "w")

    # Files where measurements are stored
    craco_no_prior = "../Analysis/CRACO/logF_Full/limits.dat"
    real_no_prior = "../Analysis/Real/real/limits.dat"

    real_F_wH0prior_file = "../Analysis/py/wH0_others_measured/limits.dat"
    real_F_CMB_file = (
        "../Analysis/py/wH0_others_measured/limits_others_$H_0 = 67.4$.dat"
    )
    real_F_SNe_file = (
        "../Analysis/py/wH0_others_measured/limits_others_$H_0 = 73.04$.dat"
    )

    craco_F_wH0prior_file = "../Analysis/py/wH0_others_forecast/limits.dat"
    craco_F_CMB_file = (
        "../Analysis/py/wH0_others_forecast/limits_others_$H_0 = 67.4$.dat"
    )
    craco_F_SNe_file = (
        "../Analysis/py/wH0_others_forecast/limits_others_$H_0 = 73.04$.dat"
    )

    real_H0_wFprior_file = "../Analysis/py/wF_others_measured/limits.dat"
    craco_H0_wFprior_file = "../Analysis/py/wF_others_forecast/limits.dat"

    def process_limits(lim, precision=1):
        lower = str(round(float(lim[5:9]), precision))
        upper = str(round(float(lim[13:17]), precision))

        return f"_{{-{lower}}}^{{+{upper}}}"

    # No Prior Measurements

    ############### H0 with no prior ##############
    with open(craco_no_prior) as f:
        craco_lines = f.readlines()

    with open(real_no_prior) as f:
        real_lines = f.readlines()

    craco_H0 = str(round(float(craco_lines[0].split("&")[1]), 1))
    craco_H0_lim = process_limits(craco_lines[0].split("&")[-2])

    #
    real_H0 = str(round(float(real_lines[0].split("&")[1]), 1))
    real_H0_lim = process_limits(real_lines[0].split("&")[-2])

    real_H0_tex = (
        f"\\newcommand{{\\Hubble}}{{\\ensuremath{{{real_H0}{real_H0_lim}}}}} \n"
    )
    craco_H0_tex = (
        f"\\newcommand{{\\fctH}}{{\\ensuremath{{{craco_H0}{craco_H0_lim}}}}} \n"
    )

    tbfil.write(real_H0_tex)
    tbfil.write(craco_H0_tex)

    ############### F with no prior ###############
    craco_F = craco_lines[-1].split("&")[1]
    craco_F_lim = process_limits(craco_lines[-1].split("&")[-2], 2)

    real_F = real_lines[-1].split("&")[1]
    real_F_lim = process_limits(real_lines[-1].split("&")[-2], 2)

    real_F_tex = (
        f"\\newcommand{{\\FnoPrior}}{{\\ensuremath{{{real_F}{real_F_lim}}}}} \n"
    )
    craco_F_tex = (
        f"\\newcommand{{\\fctFnoPrior}}{{\\ensuremath{{{craco_F}{craco_F_lim}}}}} \n"
    )
    tbfil.write(real_F_tex)
    tbfil.write(craco_F_tex)

    # Uses real data only #
    ############### lmean with no prior ##############
    real_lmean = real_lines[1].split("&")[1]
    real_lmean_lim = process_limits(real_lines[1].split("&")[-2], 2)
    real_lmean_tex = f"\\newcommand{{\\lmeannoPrior}}{{\\ensuremath{{{real_lmean}{real_lmean_lim}}}}} \n"
    tbfil.write(real_lmean_tex)

    ############### lsigma with no prior ##############
    real_lsigma = real_lines[2].split("&")[1]
    real_lsigma_lim = process_limits(real_lines[2].split("&")[-2], 2)
    real_lsigma_tex = f"\\newcommand{{\\lhostnoPrior}}{{\\ensuremath{{{real_lsigma}{real_lsigma_lim}}}}} \n"
    tbfil.write(real_lsigma_tex)

    ### -------- Measurements with priors --------- ###

    ############### H0 with F prior ###############

    ## Synthetic Data ##

    with open(craco_H0_wFprior_file) as f:
        craco_H0_wFprior_lines = f.readlines()

    craco_H0_wFprior = str(round(float(craco_H0_wFprior_lines[0].split("&")[1]), 1))
    craco_H0_wFprior_lim = process_limits(craco_H0_wFprior_lines[0].split("&")[-2])
    craco_H0_wFprior_tex = f"\\newcommand{{\\fctHwPrior}}{{\\ensuremath{{{craco_H0_wFprior}{craco_H0_wFprior_lim}}}}} \n"

    # craco_lmean_wFprior = str(round(float(craco_H0_wFprior_lines[1].split("&")[1]), 1))
    # craco_lmean_wFprior_lim = process_limits(craco_H0_wFprior_lines[1].split("&")[-2])
    # craco_lmean_wFprior_tex = f"\\newcommand{{\\fctlmeanwFPrior}}{{\\ensuremath{{{craco_lmean_wFprior}{craco_lmean_wFprior_lim}}}}} \n"

    # craco_lsigma_wFprior = str(round(float(craco_H0_wFprior_lines[2].split("&")[1]), 1))
    # craco_lsigma_wFprior_lim = process_limits(craco_H0_wFprior_lines[2].split("&")[-2])
    # craco_lsigma_wFprior_tex = f"\\newcommand{{\\fctlhostwFPrior}}{{\\ensuremath{{{craco_lsigma_wFprior}{craco_lsigma_wFprior_lim}}}}} \n"

    tbfil.write(craco_H0_wFprior_tex)
    # tbfil.write(craco_lmean_wFprior_tex)
    # tbfil.write(craco_lsigma_wFprior_tex)

    ## Real Data  -- This doesn't make sense lol##
    # with open(real_H0_wFprior_file) as f:
    #     real_H0_wFprior_lines = f.readlines()

    # real_H0_wFprior = str(round(float(real_H0_wFprior_lines[0].split("&")[1]), 1))
    # real_H0_wFprior_lim = process_limits(real_H0_wFprior_lines[0].split("&")[-2])
    # real_H0_wFprior_tex = f"\\newcommand{{\\HwFPrior}}{{\\ensuremath{{{real_H0_wFprior}{real_H0_wFprior_lim}}}}} \n"

    # real_lmean_wFprior = str(round(float(real_H0_wFprior_lines[1].split("&")[1]), 1))
    # real_lmean_wFprior_lim = process_limits(real_H0_wFprior_lines[1].split("&")[-2])
    # real_lmean_wFprior_tex = f"\\newcommand{{\\lmeanwFPrior}}{{\\ensuremath{{{real_lmean_wFprior}{real_lmean_wFprior_lim}}}}} \n"

    # real_lsigma_wFprior = str(round(float(real_H0_wFprior_lines[2].split("&")[1]), 1))
    # real_lsigma_wFprior_lim = process_limits(real_H0_wFprior_lines[2].split("&")[-2])
    # real_lsigma_wFprior_tex = f"\\newcommand{{\\lsigmawFPrior}}{{\\ensuremath{{{real_lsigma_wFprior}{real_lsigma_wFprior_lim}}}}} \n"

    # tbfil.write(real_H0_wFprior_tex)
    # tbfil.write(real_lmean_wFprior_tex)
    # tbfil.write(real_lsigma_wFprior_tex)

    ############### F with H0 prior ###############
    # Measurements
    # Uniform Prior
    with open(real_F_wH0prior_file) as f:
        real_F_wH0prior_lines = f.readlines()

    real_F_wH0prior = real_F_wH0prior_lines[-1].split("&")[1]
    real_F_wH0prior_lim = process_limits(real_F_wH0prior_lines[-1].split("&")[-2], 2)
    real_F_wH0prior_tex = f"\\newcommand{{\\FwHPrior}}{{\\ensuremath{{{real_lsigma_wFprior}{real_lsigma_wFprior_lim}}}}} \n"

    real_lmean_wH0prior = str(round(float(real_F_wH0prior_lines[1].split("&")[1]), 1))
    real_lmean_wH0prior_lim = process_limits(real_F_wH0prior_lines[1].split("&")[-2])
    real_lmean_wH0prior_tex = f"\\newcommand{{\\lmeanwHPrior}}{{\\ensuremath{{{real_lmean_wH0prior}{real_lmean_wH0prior_lim}}}}} \n"

    real_lsigma_wH0prior = str(round(float(real_F_wH0prior_lines[2].split("&")[1]), 1))
    real_lsigma_wH0prior_lim = process_limits(real_F_wH0prior_lines[2].split("&")[-2])
    real_lsigma_wH0prior_tex = f"\\newcommand{{\\lsigmawHPrior}}{{\\ensuremath{{{real_lsigma_wH0prior}{real_lsigma_wH0prior_lim}}}}} \n"

    # CMB
    with open(real_F_CMB_file) as f:
        real_F_CMB_lines = f.readlines()

    real_F_CMB = real_F_CMB_lines[0].split("&")[1]
    real_F_CMB_lim = process_limits(real_F_CMB_lines[0].split("&")[-2], 2)

    # SNe
    with open(real_F_SNe_file) as f:
        real_F_SNe_lines = f.readlines()

    real_F_SNe = real_F_SNe_lines[0].split("&")[1]
    real_F_SNe_lim = process_limits(real_F_SNe_lines[0].split("&")[-2], 2)

    real_F_wH0prior_tex = f"\\newcommand{{\\FwPrior}}{{\\ensuremath{{{real_F_wH0prior}{real_F_wH0prior_lim}}}}} \n"
    real_F_CMB_tex = (
        f"\\newcommand{{\\FCMB}}{{\\ensuremath{{{real_F_CMB}{real_F_CMB_lim}}}}} \n"
    )
    real_F_SNe_tex = (
        f"\\newcommand{{\\FSNe}}{{\\ensuremath{{{real_F_SNe}{real_F_SNe_lim}}}}} \n"
    )

    tbfil.write(real_F_wH0prior_tex)
    tbfil.write(real_F_CMB_tex)
    tbfil.write(real_F_SNe_tex)

    tbfil.write(real_lmean_wH0prior_tex)
    tbfil.write(real_lsigma_wH0prior_tex)

    # Forecasts
    with open(craco_F_wH0prior_file) as f:
        craco_F_wH0prior_lines = f.readlines()

    craco_F_wH0prior = craco_F_wH0prior_lines[-1].split("&")[1]
    craco_F_wH0prior_lim = process_limits(craco_F_wH0prior_lines[-1].split("&")[-2], 2)

    with open(craco_F_CMB_file) as f:
        craco_F_CMB_lines = f.readlines()

    craco_F_CMB = craco_F_CMB_lines[0].split("&")[1]
    craco_F_CMB_lim = process_limits(craco_F_CMB_lines[0].split("&")[-2], 2)

    with open(craco_F_SNe_file) as f:
        craco_F_SNe_lines = f.readlines()

    craco_F_SNe = real_F_SNe_lines[0].split("&")[1]
    craco_F_SNe_lim = process_limits(real_F_SNe_lines[0].split("&")[-2], 2)

    craco_F_wH0prior_tex = f"\\newcommand{{\\fctFwPrior}}{{\\ensuremath{{{craco_F_wH0prior}{craco_F_wH0prior_lim}}}}} \n"
    craco_F_CMB_tex = f"\\newcommand{{\\fctFCMB}}{{\\ensuremath{{{craco_F_CMB}{craco_F_CMB_lim}}}}} \n"
    craco_F_SNe_tex = f"\\newcommand{{\\fctFSNe}}{{\\ensuremath{{{craco_F_SNe}{craco_F_SNe_lim}}}}} \n"

    tbfil.write(craco_F_wH0prior_tex)
    tbfil.write(craco_F_CMB_tex)
    tbfil.write(craco_F_SNe_tex)

    # Lower limit on F
    arr = craco_F_wH0prior_lines[-1].split("&")
    F_lower = str(float(arr[1]) - float(arr[2][5:9]))
    F_lower_tex = f"\\newcommand{{\Flower}}{{\\ensuremath{{{F_lower}}}}} \n"
    tbfil.write(F_lower_tex)

    tbfil.close()

    print("Wrote {:s}".format(outfile))


def mktab_frbs(outfile="tab_frbs.tex", sub=False):
    state = parameters.State()
    zDMgrid, zvals, dmvals = misc_functions.get_zdm_grid(
        state, new=True, plot=False, method="analytic"
    )

    # Load up the surveys
    names = ["CRAFT/ICS892", "CRAFT/ICS", "CRAFT/ICS1632"]
    survey_name = ["\icslow", "\icsmid", "\icshigh"]
    new_frb_count = [3, 3, 2]

    # Open
    tbfil = open(outfile, "w")

    # Header
    tbfil.write("\\begin{table*}\n")
    tbfil.write("\\centering\n")
    tbfil.write("\\begin{minipage}{170mm} \n")
    tbfil.write("\\centering\n")
    tbfil.write(
        "\\caption{New FRB detections detected in 2022 used in addition to the FRB surveys used in \citet{j22b}. The FRB name, SNR-maximizing DM, \dmism\ estimated using the NE2001 model of \citet{CordesLazio01}, central frequency of observation $\\nu$, measured signal-to-noise ratio SNR, redshift $z$, and original reference. Where redshifts are not given, this is because (a): no voltage data were dumped, preventing radio localization; (b) optical follow-up observations are not yet complete; (c) Substantial Galactic extinction has challenged follow-up optical observations; (d) the host galaxy appears too distant to accurately measure a redshift. All FRBs referenced are from Shannon et al. (in prep.) with the exception of FRB20220610A \citep{ryder22}. \label{tab:frbs}}\n"
    )
    tbfil.write("\\begin{tabular}{ccccccc}\n")
    tbfil.write("\\hline \n")
    tbfil.write("Name & Survey & DM & \dmism  & $\\nu$ & SNR & $z$ \\\ \n")
    tbfil.write("& & (\dmunits) & (\dmunits) & (MHz) &  &  \n")
    tbfil.write("\\\\ \n")
    tbfil.write("\\hline \n")

    # Loop on survey
    for i, name in enumerate(names):
        isurvey = survey.load_survey(name, state, dmvals)
        # Loop on FRBs
        for frb_idx in range(new_frb_count[i]):
            idx = int(-(frb_idx + 1))
            slin = f'{isurvey.frbs["TNS"].iat[idx]}'
            slin += f"& {survey_name[i]}"
            slin += f'& {isurvey.frbs["DM"].iat[idx]}'
            slin += f'& {isurvey.frbs["DMG"].iat[idx]}'
            slin += f'& {isurvey.frbs["FBAR"].iat[idx]}'
            slin += f'& {isurvey.frbs["SNR"].iat[idx]}'
            redshift = isurvey.frbs["Z"].iat[idx]
            if redshift != -1:
                slin += f"& {redshift}"
            else:
                slin += f"& --"

            # Write
            tbfil.write(slin)
            tbfil.write("\\\\ \n")
        tbfil.write("\\hline \n")

    # End
    tbfil.write("\\hline")
    tbfil.write("\\end{tabular} \n")
    tbfil.write("\\end{minipage} \n")
    tbfil.write("\\end{table*} \n")

    tbfil.close()

    print("Wrote {:s}".format(outfile))


# Command line execution
if __name__ == "__main__":
    mktab_model_params()
    mktex_measurements()
    mktab_frbs()
