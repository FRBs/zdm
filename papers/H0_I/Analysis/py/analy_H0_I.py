# imports
import numpy as np
from zdm.craco import loading

from IPython import embed


def generate_sz_grid(ns=1000, outfile='sz_grid.npy'):
    # Load
    survey, grid = loading.survey_and_grid(
        survey_name='CRACO_alpha1_Planck18_Gamma',
        NFRB=100, lum_func=1)

    # Init
    nw, nz, nDM = grid.thresholds.shape
    Emax=10**grid.state.energy.lEmax
    Emin=10**grid.state.energy.lEmin
    gamma=grid.state.energy.gamma

    # Collapse rates on DM
    z_coll = np.sum(grid.rates, axis=1)

    # s values (S/N)
    s_vals = np.linspace(survey.SNRTHRESHs[0], 1000., ns)

    sz_grid = np.zeros((nz, ns))

    # Loop on z
    sEobs = np.zeros((nw, nDM, s_vals.size))
    for zz in np.arange(0, 100, 10):
        print(f"zz={zz}")
        # Build sEobs
        for kk in range(nw):
            sEobs[kk,...] = np.outer(grid.thresholds[kk,zz,:], s_vals)
        # 
        psnr = np.zeros((nDM, ns))
        # This loop is slow!
        for i,b in enumerate(survey.beam_b):
            bEobs = sEobs/b
            temp = grid.array_diff_lf(bEobs,Emin,Emax,gamma) * survey.beam_o[i]
            # Sum on w
            psnr += np.inner(temp.T, grid.eff_weights).T

        # Normalize by collapse
        ps = np.sum(psnr, axis=0)
        norm = np.sum(ps)


        sz_grid[zz,:] = z_coll[zz]*ps/norm

    
    embed(header='50 of analy')    
    from matplotlib import pyplot as plt
    plt.clf()
    ax = plt.gca()
    for zz in np.arange(0, 100, 10):
        ax.plot(s_vals, np.log10(sz_grid[zz,:]), label=f'z={grid.zvals[zz]:0.2f}')
    ax.legend()
    plt.show()

generate_sz_grid()