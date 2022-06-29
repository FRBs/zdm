import os, sys
import numpy as np

from matplotlib import pyplot as plt
from scipy.interpolate import interp1d

from frb.dm import igm as figm
from frb.figures import utils as fig_utils

from zdm import figures

sys.path.append(os.path.abspath("../Analysis/py"))
sys.path.append(os.path.abspath("../../Analysis/py"))
import analy_F_I

def fig_craco_varyF_zDM(
    outfile,
    zmax=2.3,
    DMmax=1500,
    norm=2,
    other_param="Emax",
    Aconts=[0.05],
    fuss_with_ticks: bool = False,
):
    """_summary_

    Args:
        outfile (_type_): _description_
        zmax (float, optional): _description_. Defaults to 2.3.
        DMmax (int, optional): _description_. Defaults to 1500.
        norm (int, optional): _description_. Defaults to 2.
        other_param (str, optional): _description_. Defaults to 'Emax'.
        Aconts (list, optional): _description_. Defaults to [0.05].
        fuss_with_ticks (bool, optional): _description_. Defaults to False.
    """
    # Generate the grid
    survey, grid = analy_F_I.craco_mc_survey_grid()

    # survey, grid = loading.survey_and_grid(
    #    survey_name='CRACO_alpha1_Planck18_Gamma',
    #    NFRB=100, lum_func=2)

    fiducial_Emax = grid.state.energy.lEmax
    fiducial_H0 = grid.state.cosmo.H0

    plt.figure()
    ax1 = plt.axes()

    plt.sca(ax1)


    plt.xlabel("z")
    plt.ylabel("${\\rm DM}_{\\rm EG}$")

    if other_param == "Emax":
        F_values = [0.01, 0.3, 0.7, 0.9]
        other_values = [0.0, 0.0, 0.0, -0.1]
        lstyles = ["-", "-", "-", ":"]
        zticks = [0.5, 1.0, 1.5, 2.0]
        ylim = (0.0, DMmax)
    elif other_param == "H0":
        F_values = [0.01, 0.3, 0.7, 0.9]
        other_values = [fiducial_H0, fiducial_H0, fiducial_H0, fiducial_H0]
        lstyles = ["-", "-", "-", ":"]
        zticks, ylim = None, None

    # Loop on grids
    legend_lines = []
    labels = []
    for F, scl, lstyle, clr in zip(
        F_values, other_values, lstyles, ["b", "k", "r", "gray"]
    ):

        # Update grid
        vparams = {}
        vparams["F"] = F

        vparams["lmean"] = 1e-3
        vparams["lsigma"] = 0.1

        if other_param == "Emax":
            vparams["lEmax"] = fiducial_Emax + scl
        elif other_param == "H0":
            vparams["H0"] = scl
        grid.update(vparams)

        # Unpack
        full_zDMgrid, zvals, dmvals = (
            grid.rates.copy(),
            grid.zvals.copy(),
            grid.dmvals.copy(),
        )

        # currently this is "per cell" - now to change to "per DM"
        # normalises the grid by the bin width, i.e. probability per bin, not probability density

        # checks against zeros for a log-plot

        zvals, dmvals, zDMgrid = figures.proc_pgrid(
            full_zDMgrid, zvals, (0, zmax), dmvals, (0, DMmax)
        )


        # Contours
        alevels = figures.find_Alevels(full_zDMgrid, Aconts)

        # sets the x and y tics
        # JXP fussing here!!

        tvals, ticks = figures.ticks_pgrid(zvals, these_vals=zticks)  # , fmt='str4')
        plt.xticks(tvals, ticks)
        tvals, ticks = figures.ticks_pgrid(dmvals, fmt="int")  # , fmt='str4')
        plt.yticks(tvals, ticks)

        ax = plt.gca()
        cs = ax.contour(
            zDMgrid.T, levels=alevels, origin="lower", colors=[clr], linestyles=lstyle
        )
        leg, _ = cs.legend_elements()
        legend_lines.append(leg[0])

        # Label
        if other_param == "Emax":
            labels.append(
                r"$F = $" + f"{F}, log " + r"$E_{\rm max}$" + f"= {vparams['lEmax']}"
            )
        elif other_param == "H0":
            labels.append(r"$F = $" + f"{F}, H0 = {vparams['H0']}")

    ###### gets decent axis labels, down to 1 decimal place #######
    ax = plt.gca()

    # Interpolators
    f_DM = interp1d(dmvals, np.arange(dmvals.size), fill_value='extrapolate', bounds_error=False)
    f_z = interp1d(zvals, np.arange(zvals.size), fill_value='extrapolate', bounds_error=False)

    from astropy.cosmology import FlatLambdaCDM
    cosmo = FlatLambdaCDM(H0=grid.state.cosmo.H0, 
                          Ob0=grid.state.cosmo.Omega_b, 
                          Om0=grid.state.cosmo.Omega_m)
    dms, zeval = figm.average_DM(2.0, cumul=True, cosmo=cosmo)

    ax.plot(f_z(zeval), f_DM(dms), 'k--', label='Macquart Relation')

    ax.legend(legend_lines, labels, loc="lower right")

    # Fontsize
    fig_utils.set_fontsize(ax, 16.0)

    # Ticks
    if fuss_with_ticks:
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

    # Finish
    plt.tight_layout()
    plt.savefig(outfile, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Wrote: {outfile}")

fig_craco_varyF_zDM("test.pdf", other_param="H0")