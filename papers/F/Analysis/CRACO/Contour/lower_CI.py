from webbrowser import get
import numpy as np
import zdm
import matplotlib.pyplot as plt
from frb.dm import cosmic
from zdm.pcosmic import pcosmic, get_mean_DM
from zdm.parameters import State
import scipy.stats

from IPython import embed

fC0 = cosmic.grab_C0_spline()


def lower_ci(data, conflevel=0.95):
    mu, sigma = np.mean(data), scipy.stats.sem(data)
    k = sigma * scipy.stats.t.ppf((1 + conflevel) / 2.0, len(data) - 1)
    return mu - k


def lowerCI_F(
    Fs,
    H0=None,
    z=0.5,
    deltas=np.linspace(0.01, 5, 200),
    niter_per_F=1000,
    ns_per_F=1000,
):

    z = np.array(z).reshape(1)

    n = len(Fs)

    lower_cis = np.zeros(n)

    state = State()

    if H0 is not None:
        state.update_params({"H0": H0})
    else:
        H0 = state.cosmo.H0

    mean_dm_cosmic = get_mean_DM(z, state)

    for k, F in enumerate(Fs):

        sigma = F / np.sqrt(z)
        C0 = fC0(sigma)
        pdelta = zdm.pcosmic.pcosmic(deltas, z, F, C0)

        # Perform a bootstrap CI for `niter_per_F` iterations
        lower_cis_at_F = np.zeros(niter_per_F)
        for i in range(niter_per_F):
            sample = np.random.choice(
                deltas, p=pdelta / np.sum(pdelta), size=ns_per_F, replace=True
            )
            lower_cis_at_F[i] = lower_ci(sample) * mean_dm_cosmic

        # Get the mean lower 95 pct CI from the bootstrap
        lower_cis[k] = np.mean(lower_cis_at_F)

    return {"F": Fs, "lower.ci": lower_cis, "H0": np.ones(n) * H0}


def lowerCI_H0(
    H0s,
    F=None,
    z=0.5,
    deltas=np.linspace(0.01, 5, 200),
    niter_per_H0=1000,
    ns_per_H0=1000,
):

    z = np.array(z).reshape(1)

    n = len(H0s)

    lower_cis = np.zeros(n)

    state = State()

    if F is not None:
        state.update_params({"F": F})
    else:
        F = state.IGM.F

    sigma = F / np.sqrt(z)
    C0 = fC0(sigma)
    pdelta = zdm.pcosmic.pcosmic(deltas, z, F, C0)

    for k, H0 in enumerate(H0s):

        state.update_params({"H0": H0})

        mean_dm_cosmic = get_mean_DM(z, state)

        # Perform a bootstrap CI for `niter_per_F` iterations
        lower_cis_at_H0 = np.zeros(niter_per_H0)
        for i in range(niter_per_H0):

            sample = np.random.choice(
                deltas, p=pdelta / np.sum(pdelta), size=ns_per_H0, replace=True
            )

            lower_cis_at_H0[i] = lower_ci(sample) * mean_dm_cosmic

        # Get the mean lower 95 pct CI from the bootstrap
        lower_cis[k] = np.mean(lower_cis_at_H0)

    return {"H0": H0s, "lower.ci": lower_cis, "F": np.ones(n) * F}


def make_plots_F(
    Fs,
    H0=None,
    z=0.5,
    deltas=np.linspace(0.01, 5, 200),
    niter_per_F=1000,
    ns_per_F=1000,
    outfile="F_plot.png",
):

    df = lowerCI_F(
        Fs,
        H0=H0,
        z=0.5,
        deltas=np.linspace(0.01, 5, 200),
        niter_per_F=1000,
        ns_per_F=1000,
    )

    H0 = df["H0"][0]

    fig, ax = plt.subplots(dpi=200)
    ax.scatter(df["F"], df["lower.ci"])
    ax.set_title(f"H0 = {H0}, z = {z}")
    ax.set_xlabel(f"$F$")
    ax.set_ylabel(f"$p(\Delta)$")
    plt.savefig(outfile)


def make_plots_H0(
    H0s,
    F=None,
    z=0.5,
    deltas=np.linspace(0.01, 5, 200),
    niter_per_H0=1000,
    ns_per_H0=1000,
    outfile="H0_plot.png",
):

    df = lowerCI_H0(
        H0s,
        F=F,
        z=0.5,
        deltas=np.linspace(0.01, 5, 200),
        niter_per_H0=1000,
        ns_per_H0=1000,
    )

    F = df["F"][0]

    fig, ax = plt.subplots(dpi=200)
    ax.scatter(df["H0"], df["lower.ci"])
    ax.set_title(f"F = {F}, z = {z}")
    ax.set_xlabel(f"$H_0$")
    ax.set_ylabel(f"$p(\Delta)$")
    plt.savefig(outfile)


# make_plots_F(np.linspace(0.1, 1, 20), z=0.5, outfile="F_plot_z_0.5.png")
# make_plots_H0(np.linspace(50, 80, 20), z=0.5, outfile="H0_plot_z_0.5.png")

# make_plots_F(np.linspace(0.1, 1, 20), z=0.25, outfile="F_plot_z_0.25.png")
# make_plots_H0(np.linspace(50, 80, 20), z=0.25, outfile="H0_plot_z_0.25.png")

# make_plots_F(np.linspace(0.1, 1, 20), z=0.1, outfile="F_plot_z_0.1.png")
# make_plots_H0(np.linspace(50, 80, 20), z=0.1, outfile="H0_plot_z_0.1.png")

# make_plots_F(np.linspace(0.1, 1, 20), z=1.5, outfile="F_plot_z_1.5.png")
# make_plots_H0(np.linspace(50, 80, 20), z=1.5, outfile="H0_plot_z_1.5.png")

make_plots_F(np.linspace(0.1, 1, 20), H0=55, z=0.25, outfile="F_plot_z_0.25_alt.png")
make_plots_H0(np.linspace(50, 80, 20), F=0.8, z=0.25, outfile="H0_plot_z_0.25_alt.png")
