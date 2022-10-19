import numpy as np
import zdm
import matplotlib.pyplot as plt
from frb.dm import cosmic
from zdm.pcosmic import pcosmic, get_mean_DM
from zdm.parameters import State
import scipy.stats

fC0 = cosmic.grab_C0_spline()


def makePDeltaPlot_F(deltas, F, z, outfile=None):
    sigma = F / np.sqrt(z)
    C0 = fC0(sigma)
    pdelta = zdm.pcosmic.pcosmic(deltas, z, F, C0)
    fig, ax = plt.subplots(dpi=200)
    ax.plot(deltas, pdelta, c="k")
    if outfile is None:
        outfile = f"pdelta_{F}_{z}.png"
    ax.set_xlabel(r"$\Delta$")
    ax.set_ylabel(r"$p(\Delta)$")
    ax.set_title(f"F={F}, z={z}")
    plt.savefig(outfile)


def test(deltas, Fs, z, colors, outfile=None):

    fig, ax = plt.subplots(figsize=(5, 4), dpi=200)

    for i, F in enumerate(Fs):
        sigma = F / np.sqrt(z)
        C0 = fC0(sigma)
        pdelta = zdm.pcosmic.pcosmic(deltas, z, F, C0)
        ax.plot(deltas, pdelta, c=colors[i], label=f"F = {F}")

    if outfile is None:
        outfile = f"pdelta_test.png"
    ax.set_xlabel(r"$\Delta$")
    ax.set_ylabel(r"$p(\Delta)$")
    ax.set_title(f"z={z}")
    ax.legend()
    plt.savefig(outfile, bbox_inches="tight")


def test2(deltas, Fs, z, colors, outfile=None):

    state = State()

    fig, ax = plt.subplots(figsize=(5, 4), dpi=200)

    for i, F in enumerate(Fs):

        state.update_params({"F": F})

        sigma = F / np.sqrt(z)
        C0 = fC0(sigma)
        pdelta = zdm.pcosmic.pcosmic(deltas, z, F, C0)
        mean_DM = get_mean_DM(np.array(z).reshape(1), state)
        ax.plot(deltas * mean_DM, pdelta, c=colors[i], label=f"F = {F}")

    if outfile is None:
        outfile = f"pdelta_test_2.png"
    ax.set_xlabel(r"$\rm{DM_{EG}}$")
    ax.set_ylabel(r"$p(\rm{DM_{EG}})$")
    ax.set_title(f"z={z}")
    ax.legend()
    plt.savefig(outfile, bbox_inches="tight")


# makePDeltaPlot_F(np.linspace(0.01, 2.5, 100), 0.32, 1)
# makePDeltaPlot_F(np.linspace(0.01, 2.5, 100), 0.01, 1)
# makePDeltaPlot_F(np.linspace(0.01, 2.5, 100), 1, 1)

test2(
    np.linspace(0.01, 2.5, 300), [0.01, 0.9], z=0.5, colors=["r", "orange"],
)

