# imports
import numpy as np
from zdm.MC_sample import loading
from zdm import errors_misc_functions as err

from IPython import embed

fiducial_survey = 'CRACO_std_May2022'

def craco_mc_survey_grid():
    """ Load the defaul MonteCarlo survey+grid for CRACO """
    survey, grid = loading.survey_and_grid(
        survey_name=fiducial_survey,
        NFRB=100, lum_func=2, iFRB=100)
    return survey, grid

def generate_grids(params, ns=100, logsmax=2.5):

    H0_values = [60., 70., 80., 80.]
    lEmax_values = [41.4, 41.4, 41.4, 41.3]
    snrs=np.logspace(0,logsmax,ns)

    # Load
    survey, grid = craco_mc_survey_grid()

    for H0, lEmax, in zip(H0_values, lEmax_values):

        print(f"Working on H0={H0}, lEmax={lEmax}")
        vparams = {}
        vparams['H0'] = H0
        vparams['lEmax'] = lEmax
        grid.update(vparams)

        # Generate sz
        if params == 'sz':
            psnrs,psz=err.get_sc_grid(grid, ns, snrs, calc_psz=True)
        elif params == 'sDM':
            psnrs,psDM=err.get_sc_grid(grid, ns, snrs, calc_psz=False)

        # Outfile
        outfile = f'GridData/p{params}_H0{int(H0)}_Emax{lEmax}.npz'
        if params == 'sz':
            np.savez(outfile, snrs=snrs, zvals=grid.zvals, psz=psz)
        elif params == 'sDM':
            np.savez(outfile, snrs=snrs, dmvals=grid.dmvals, psDM=psDM)
        print(f"Wrote: {outfile}")


def deprecated():
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

        # Collapse in DM
        #  TODO -- Do I need to weight by rates??
        ps = np.sum(psnr, axis=0)
        norm = np.sum(ps)
        # Save, Normalize by collapse in DMz
        sz_grid[zz,:] = z_coll[zz]*ps/norm

    
    embed(header='50 of analy')    
    from matplotlib import pyplot as plt
    plt.clf()
    ax = plt.gca()
    for zz in np.arange(0, 100, 10):
        ax.plot(s_vals, np.log10(sz_grid[zz,:]), label=f'z={grid.zvals[zz]:0.2f}')
    ax.legend()
    plt.show()

if __name__ == '__main__':
    #generate_grids('sz')
    generate_grids('sDM')