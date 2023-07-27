import os, sys
import numpy as np

from matplotlib import pyplot as plt
from scipy.interpolate import interp1d

from frb.dm import igm as figm
from frb.figures import utils as fig_utils

from zdm import figures, pcosmic

sys.path.append(os.path.abspath("../Analysis/py"))
sys.path.append(os.path.abspath("../../Analysis/py"))
import analy_F_I

from astropy.cosmology import FlatLambdaCDM


def fig_varyF(
    outfile,
    Fs,
    H0s,
    lmeans,
    lsigmas,
    zmax=2.3,
    DMmax=1500,
    Aconts=[0.05],
    lcolors=["b"],
    lstyles=["-"],
    labels=[""],
    zticks=None,
    ylim=None,
    iFRB=0,
    show_FRBs=True,
    plotMacquart=True,
):
    survey, grid = analy_F_I.craco_mc_survey_grid(iFRB=iFRB)

    fiducial_F = grid.state.IGM.logF
    fiducial_H0 = grid.state.cosmo.H0
    fiducial_lmean = grid.state.host.lmean
    fiducial_lsigma = grid.state.host.lsigma

    fig, ax = plt.subplots(dpi=300)

    ax.set_xlabel("z")
    ax.set_ylabel("${\\rm DM}_{\\rm EG}$")

    legend_lines = []

    for F, H0, lmean, lsigma, lstyle, color in zip(
        Fs, H0s, lmeans, lsigmas, lstyles, lcolors
    ):
        vparams = {}

        if F is None:
            F = fiducial_F
        if H0 is None:
            H0 = fiducial_H0
        if lmean is None:
            lmean = fiducial_lmean
        if lsigma is None:
            lsigma = fiducial_lsigma

        vparams["logF"] = F
        vparams["H0"] = H0
        vparams["lmean"] = lmean
        vparams["lsigma"] = lsigma

        grid.update(vparams)

        full_zDMgrid, zvals, dmvals = (
            grid.rates.copy(),
            grid.zvals.copy(),
            grid.dmvals.copy(),
        )

        zvals, dmvals, zDMgrid = figures.proc_pgrid(
            full_zDMgrid, zvals, (0, zmax), dmvals, (0, DMmax)
        )

        alevels = figures.find_Alevels(full_zDMgrid, Aconts)

        plt.sca(ax)

        tvals, ticks = figures.ticks_pgrid(zvals, these_vals=zticks)
        plt.xticks(tvals, ticks)

        tvals, ticks = figures.ticks_pgrid(dmvals, fmt="int")
        plt.yticks(tvals, ticks)

        cs = ax.contour(
            zDMgrid.T, levels=alevels, origin="lower", colors=[color], linestyles=lstyle
        )

        leg, _ = cs.legend_elements()
        legend_lines.append(leg[0])

        ### TEST
        # Interpolators
        f_DM = interp1d(
            dmvals, np.arange(dmvals.size), fill_value="extrapolate", bounds_error=False
        )
        f_z = interp1d(
            zvals, np.arange(zvals.size), fill_value="extrapolate", bounds_error=False
        )

        cosmo = FlatLambdaCDM(
            H0=grid.state.cosmo.H0,
            Ob0=grid.state.cosmo.Omega_b,
            Om0=grid.state.cosmo.Omega_m,
        )

        if plotMacquart:
            dms, zeval = figm.average_DM(3.0, cumul=True, cosmo=cosmo)
            l_mqr = ax.plot(f_z(zeval), f_DM(dms), ls="--", c=color, alpha=0.5)

        print("F = ", F, "H0 = ", H0, "lmean = ", lmean, "lsigma = ", lsigma)

    # put down FRBs
    FRBZ = survey.frbs["Z"]
    FRBDM = survey.DMEGs

    ddm = dmvals[1] - dmvals[0]
    dz = zvals[1] - zvals[0]
    nz, ndm = zDMgrid.shape

    ##### add FRB host galaxies at some DM/redshift #####
    if (FRBZ is not None) and show_FRBs:
        iDMs = FRBDM / ddm
        iZ = FRBZ / dz
        # Restrict to plot range
        gd = (FRBDM < DMmax) & (FRBZ < zmax)
        ax.plot(iZ[gd], iDMs[gd], "ko", linestyle="", markersize=2.0)

    ax.legend(legend_lines, labels, loc="lower right")

    # Fontsize
    fig_utils.set_fontsize(ax, 16.0)

    fig.tight_layout()
    plt.savefig(outfile, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Wrote: {outfile}")


# fig_varyF(
#     "fig_varyF_H0_compare.png",
#     Fs=[-0.57, -0.37, None],
#     H0s=[69.02, 77.14, None],
#     lmeans=[None, None, None],
#     lsigmas=[None, None, None],
#     lcolors=["r", "b", "k"],
#     lstyles=["-", "-", "--"],
#     labels=["Synthetic", "Real", "Fiducial"],
#     DMmax=2500,
#     Aconts=[0.01],
#     show_FRBs=False,
#     zmax=3,
# )

# 95th ptile
fig_varyF(
    "degeneracy/fig_H0_F_Degeneracy.png",
    Fs=[None, np.log10(0.82)],
    H0s=[None, 55],
    lmeans=[None, None],
    lsigmas=[None, None],
    lcolors=["b", "r"],
    lstyles=["-", "-"],
    labels=[
        r"$H_0$ = 67.66, $\log_{10} F =$ -0.49",
        r"$H_0$ = 55, $\log_{10} F =$ -0.086",
    ],
    DMmax=1800,
    Aconts=[0.025],
    show_FRBs=False,
    zmax=2.3,
    plotMacquart=False,
)
