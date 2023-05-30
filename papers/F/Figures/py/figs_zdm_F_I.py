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


def fig_craco_varyF_zDM(
    outfile,
    zmax=2.3,
    DMmax=1500,
    norm=2,
    other_param="Emax",
    Aconts=[0.05],
    fuss_with_ticks: bool = False,
    suppress_DM_host=False,
    iFRB=0,
    show_FRBS=True,
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
    survey, grid = analy_F_I.craco_mc_survey_grid(iFRB=iFRB)

    fiducial_Emax = grid.state.energy.lEmax
    fiducial_H0 = grid.state.cosmo.H0
    fiducial_lmean = grid.state.host.lmean
    fiducial_lsigma = grid.state.host.lsigma

    plt.figure(dpi=300)
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
    elif other_param == "lmean":
        F_values = [0.01, 0.32, 0.7, 0.32]
        other_values = [0.0, 1.0, 0.0, 3.0]
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
        vparams["logF"] = F

        # Sets the log-normal distribution for DM_host to ~0.
        if suppress_DM_host:
            vparams["lmean"] = 1e-3
            vparams["lsigma"] = 0.1

        if other_param == "Emax":
            vparams["lEmax"] = fiducial_Emax + scl
        elif other_param == "H0":
            vparams["H0"] = scl
        elif other_param == "lmean":
            vparams["lsigma"] = fiducial_lsigma
            vparams["lmean"] = scl

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

        # Sets the x and y ticks
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
                r"$\\log_\{10\} F = $"
                + f"{F}, log "
                + r"$E_{\rm max}$"
                + f"= {vparams['lEmax']}"
            )
        elif other_param == "H0":
            labels.append(r"$\log_\{10\} F = " + f"{F}, H0 = {vparams['H0']}")
        elif other_param == "lmean":
            labels.append(r"$\log_\{10\} F = " + f"{F}, $\mu =$ {vparams['lmean']}")

    ###### gets decent axis labels, down to 1 decimal place #######
    ax = plt.gca()

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
    dms, zeval = figm.average_DM(2.0, cumul=True, cosmo=cosmo)

    l_mqr = ax.plot(f_z(zeval), f_DM(dms), "k--")

    legend_lines.append(l_mqr[0])
    labels.append("Macquart Relation")

    ax.legend(legend_lines, labels, loc="lower right")

    # put the FRBs in

    FRBZ = survey.frbs["Z"]
    FRBDM = survey.DMEGs

    # Cut down grid
    zvals, dmvals, zDMgrid = figures.proc_pgrid(
        full_zDMgrid, zvals, (0, zmax), dmvals, (0, DMmax)
    )
    ddm = dmvals[1] - dmvals[0]
    dz = zvals[1] - zvals[0]
    nz, ndm = zDMgrid.shape

    ##### add FRB host galaxies at some DM/redshift #####
    if (FRBZ is not None) and show_FRBS:
        iDMs = FRBDM / ddm
        iZ = FRBZ / dz
        # Restrict to plot range
        gd = (FRBDM < DMmax) & (FRBZ < zmax)
        ax.plot(iZ[gd], iDMs[gd], "ko", linestyle="", markersize=2.0)

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


###


def fig_varyF(
    outfile,
    zmax=2.3,
    DMmax=1500,
    other_param="lmean",
    Aconts=[0.05],
    F_values=[None],
    other_values=[None],
    lcolors=["b"],
    lstyles=["-"],
    zticks=None,
    ylim=None,
    iFRB=0,
    show_FRBs=True,
):
    survey, grid = analy_F_I.craco_mc_survey_grid(iFRB=iFRB)

    fiducial_F = grid.state.IGM.logF
    fiducial_Emax = grid.state.energy.lEmax
    fiducial_H0 = grid.state.cosmo.H0
    fiducial_lmean = grid.state.host.lmean
    fiducial_lsigma = grid.state.host.lsigma

    fig, ax = plt.subplots(dpi=200)

    ax.set_xlabel("z")
    ax.set_ylabel("${\\rm DM}_{\\rm EG}$")

    legend_lines = []
    labels = []

    for F, other, lstyle, color in zip(F_values, other_values, lstyles, lcolors):
        vparams = {}

        if F is None:
            F = fiducial_F

        vparams["logF"] = F

        if other_param == "H0":
            if other == None:
                other = fiducial_H0
            vparams["H0"] = other
        elif other_param == "Emax":
            if other == None:
                other = fiducial_Emax
            vparams["Emax"] = other
            other = fiducial_Emax
        elif other_param == "lmean":
            if other == None:
                other = fiducial_lmean
            vparams["lmean"] = other
            other = fiducial_lmean

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

        dms, zeval = figm.average_DM(2.0, cumul=True, cosmo=cosmo)

        l_mqr = ax.plot(f_z(zeval), f_DM(dms), ls="--", c=color, alpha=0.5)

        #### TEST END

        if other_param == "Emax":
            labels.append(
                r"$\\log_\{10\} F = $"
                + f"{F}, log "
                + r"$E_{\rm max}$"
                + f"= {vparams['lEmax']}"
            )
        elif other_param == "H0":
            labels.append(r"$\log_{10} F = $" + f"{F}, H0 = {vparams['H0']}")
        elif other_param == "lmean":
            labels.append(r"$\log_{10} F = $" + f"{F}, $\mu =$ {vparams['lmean']}")

    # # Interpolators
    # f_DM = interp1d(
    #     dmvals, np.arange(dmvals.size), fill_value="extrapolate", bounds_error=False
    # )
    # f_z = interp1d(
    #     zvals, np.arange(zvals.size), fill_value="extrapolate", bounds_error=False
    # )

    # cosmo = FlatLambdaCDM(
    #     H0=grid.state.cosmo.H0,
    #     Ob0=grid.state.cosmo.Omega_b,
    #     Om0=grid.state.cosmo.Omega_m,
    # )

    # dms, zeval = figm.average_DM(2.0, cumul=True, cosmo=cosmo)

    # l_mqr = ax.plot(f_z(zeval), f_DM(dms), "k--")

    # legend_lines.append(l_mqr[0])
    # labels.append("Macquart Relation")

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


###


def fig_craco_fiducial_F(
    outfile="fig_craco_fiducial_F.png",
    zmax=2.5,
    DMmax=2500,
    show_Macquart=False,
    log=True,
    label="$\\log_{10} \; p(DM_{\\rm EG},z)$",
    Aconts=[0.01, 0.1, 0.5],
    cmap="jet",
    show=False,
    figsize=None,
    vmnx=(None, None),
    grid=None,
    survey=None,
    F=-0.49,
    H0=None,
    iFRB=0,
    suppress_DM_host=False,
    show_FRBs=True,
):
    """
    Very complicated routine for plotting 2D zdm grids
    Args:
        zDMgrid ([type]): [description]
        zvals ([type]): [description]
        dmvals ([type]): [description]
        zmax (int, optional): [description]. Defaults to 1.
        DMmax (int, optional): [description]. Defaults to 1000.
        norm (int, optional): [description]. Defaults to 0.
        log (bool, optional): [description]. Defaults to True.
        label (str, optional): [description]. Defaults to '$\log_{10}p(DM_{\rm EG},z)$'.
        project (bool, optional): [description]. Defaults to False.
        conts (bool, optional): [description]. Defaults to False.
        FRBZ ([type], optional): [description]. Defaults to None.
        FRBDM ([type], optional): [description]. Defaults to None.
        Aconts (bool, optional): [description]. Defaults to False.
        Macquart (state, optional): state object.  Used to generat the Maquart relation.
            Defaults to None.
        title (str, optional): [description]. Defaults to "Plot".
        H0 ([type], optional): [description]. Defaults to None.
        showplot (bool, optional): [description]. Defaults to False.
    """
    # Generate the grid
    if grid is None or survey is None:
        survey, grid = analy_F_I.craco_mc_survey_grid(iFRB=iFRB)

    fiducial_H0 = grid.state.cosmo.H0

    if H0 is None:
        H0 = fiducial_H0

    vparams = {"H0": H0, "logF": F}

    if suppress_DM_host:
        # Sets the log-normal distribution for DM_host to ~0.
        vparams["lmean"] = 1e-3
        vparams["lsigma"] = 0.1

    grid.update(vparams)

    # Unpack
    full_zDMgrid, zvals, dmvals = grid.rates, grid.zvals, grid.dmvals
    FRBZ = survey.frbs["Z"]
    FRBDM = survey.DMEGs

    ##### imshow of grid #######
    fsize = 14.0
    plt.figure(figsize=figsize)
    ax1 = plt.axes()
    plt.sca(ax1)

    plt.xlabel("z")
    plt.ylabel("${\\rm DM}_{\\rm EG}$")
    # plt.title(title+str(H0))

    # Cut down grid
    zvals, dmvals, zDMgrid = figures.proc_pgrid(
        full_zDMgrid, zvals, (0, zmax), dmvals, (0, DMmax)
    )
    ddm = dmvals[1] - dmvals[0]
    dz = zvals[1] - zvals[0]
    nz, ndm = zDMgrid.shape

    # Contours
    alevels = figures.find_Alevels(full_zDMgrid, Aconts, log=True)

    # Ticks
    tvals, ticks = figures.ticks_pgrid(zvals)  # , fmt='str4')
    plt.xticks(tvals, ticks)
    tvals, ticks = figures.ticks_pgrid(dmvals, fmt="int")  # , fmt='str4')
    plt.yticks(tvals, ticks)

    # Image
    im = plt.imshow(
        zDMgrid.T,
        cmap=cmap,
        origin="lower",
        vmin=vmnx[0],
        vmax=vmnx[1],
        interpolation="None",
        aspect="auto",
    )

    styles = ["--", "-.", ":"]
    ax = plt.gca()
    cs = ax.contour(
        zDMgrid.T, levels=alevels, origin="lower", colors="white", linestyles=styles
    )

    ax = plt.gca()

    ax.set_title(rf"$\log_{{10}} F = {F}$, $H_0$ = {H0}")

    muDMhost = np.log(10**grid.state.host.lmean)
    sigmaDMhost = np.log(10**grid.state.host.lsigma)
    meanHost = np.exp(muDMhost + sigmaDMhost**2 / 2.0)
    medianHost = np.exp(muDMhost)
    print(f"Host: mean={meanHost}, median={medianHost}")
    plt.ylim(0, ndm - 1)
    plt.xlim(0, nz - 1)
    zmax = zvals[-1]
    nz = zvals.size
    # DMbar, zeval = igm.average_DM(zmax, cumul=True, neval=nz+1)
    DM_cosmic = pcosmic.get_mean_DM(zvals, grid.state)

    # idea is that 1 point is 1, hence...
    zeval = zvals / dz
    DMEG_mean = (DM_cosmic + meanHost) / ddm
    DMEG_median = (DM_cosmic + medianHost) / ddm

    # Check median
    f_median = interp1d(zvals, DM_cosmic + medianHost, fill_value="extrapolate")
    eval_DMEG = f_median(FRBZ)
    above = FRBDM > eval_DMEG
    print(f"There are {np.sum(above)/len(FRBZ)} above the median")

    if show_Macquart:
        plt.plot(
            zeval,
            DMEG_mean,
            color="gray",
            linewidth=2,
            label="Macquart relation (mean)",
        )
        # plt.plot(
        #     zeval,
        #     DMEG_median,
        #     color="gray",
        #     linewidth=2,
        #     ls="--",
        #     label="Macquart relation (median)",
        # )
        l = plt.legend(loc="lower right", fontsize=12)
    # l=plt.legend(bbox_to_anchor=(0.2, 0.8),fontsize=8)
    # for text in l.get_texts():
    # 	text.set_color("white")

    # limit to a reasonable range if logscale
    if log and vmnx[0] is None:
        themax = zDMgrid.max()
        themin = int(themax - 4)
        themax = int(themax)
        plt.clim(themin, themax)

    ##### add FRB host galaxies at some DM/redshift #####
    if (FRBZ is not None) and show_FRBs:
        iDMs = FRBDM / ddm
        iZ = FRBZ / dz
        # Restrict to plot range
        gd = (FRBDM < DMmax) & (FRBZ < zmax)
        plt.plot(iZ[gd], iDMs[gd], "ko", linestyle="", markersize=2.0)

    cbar = plt.colorbar(im, fraction=0.046, shrink=1.2, aspect=15, pad=0.05)
    cbar.set_label(label)

    fig_utils.set_fontsize(ax, fsize)

    plt.tight_layout()

    if show:
        plt.show()
    else:
        plt.savefig(outfile, dpi=300)
        print(f"Wrote: {outfile}")
    plt.close()


### tests

# logfs = [-1.5, -1.5, -1.5]
# h0s = [62.5, 64, 67]


# for h0, logF in zip(h0s, logfs):
#     fig_craco_fiducial_F(
#         f"diagnostic/fig_craco_logF_{logF}_H0_{h0}.png",
#         show_Macquart=True,
#         F=logF,
#         H0=h0,
#         suppress_DM_host=False,
#     )

# fig_varyF(
#     "fig_varyF_H0_compare.png",
#     other_param="H0",
#     F_values=[-0.57, -.37],
#     other_values=[69.02, 77.14],
#     lcolors=["r", "b"],
#     lstyles=["-", "-"],
#     DMmax=2500,
#     Aconts=[0.01],
#     show_FRBs=False,
#     zmax=3
# )

fig_craco_fiducial_F(
    f"figs/fiducial_distribution.png",
    show_Macquart=True,
    H0=None,
    suppress_DM_host=False,
    iFRB=100,
    show_FRBs=True,
    Aconts=[0.025],
)

fig_craco_fiducial_F(
    f"figs/high_feedback_efficiency.png",
    show_Macquart=False,
    F=np.round(np.log10(0.01), 3),
    H0=None,
    suppress_DM_host=False,
    iFRB=100,
    show_FRBs=False,
    Aconts=[0.025],
)

fig_craco_fiducial_F(
    f"figs/low_feedback_efficiency.png",
    show_Macquart=False,
    F=np.round(np.log10(0.9), 3),
    H0=None,
    suppress_DM_host=False,
    iFRB=100,
    show_FRBs=False,
    Aconts=[0.025],
)
