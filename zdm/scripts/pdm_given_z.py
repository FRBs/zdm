""" Calculate p(DM|z) for a given DM and survey
"""

# It should be possible to remove all the matplotlib calls from this
# but in the current implementation it is not removed.
import argparse

from IPython import embed

def main(pargs):
    
    import numpy as np
    from matplotlib import pyplot as plt
    
    print("Can't solve local environment")

    from frb import mw
    from frb.figures.utils import set_fontsize 
    
    from zdm import survey
    from zdm import parameters
    from zdm import cosmology as cos
    from zdm import misc_functions
    
    
    
    plot_mean_dm()
    
    limits = (2.5, 97.5)

    # Set parmaeters
    state = parameters.State()
    state.host.lmean = pargs.host_lmean
    state.host.lsigma = pargs.host_lsigma
    state.energy.luminosity_function = 2  # Gamma + spline

    # Cosmology
    cos.set_cosmology(state)
    cos.init_dist_measures()

    # get the grid of p(DM|z)
    zDMgrid, zvals,dmvals = misc_functions.get_zdm_grid(
        state, new=True, plot=False, method='analytic',zmax=1,dmmax=2000)

    # Survey
    isurvey = survey.load_survey(pargs.survey, state, dmvals)

    # Grid
    igrid = misc_functions.initialise_grids(
        [isurvey], zDMgrid, zvals, dmvals, state, wdist=True)[0]
    
    
    PDM_z = igrid.rates # This is p(DM,z) including telescope bias

    # Find z in the grid
    iz = np.argmin(np.abs(zvals - pargs.z))
    PDMz = PDM_z[iz, :] / np.sum(PDM_z[iz, :])
    
    # gets values of IGM only and also unbiased estimates
    PDMz_unbiased = igrid.smear_grid[iz, :] / np.sum(igrid.smear_grid[iz, :])
    PDMz_igm_only = igrid.grid[iz, :] / np.sum(igrid.grid[iz, :])

    # Limits
    cum_sum = np.cumsum(PDMz)
    DM_EG_min = dmvals[np.argmin(np.abs(cum_sum-limits[0]/100.))]
    DM_EG_max = dmvals[np.argmin(np.abs(cum_sum-limits[1]/100.))]
    DM_EG_median = dmvals[np.argmin(np.abs(cum_sum-50./100.))]

    # Stats
    DM_EG_mean = np.sum(dmvals*PDMz) 
    DM_EG_peak = dmvals[np.argmax(PDMz)]

    # Plot
    plt.figure()
    ax = plt.gca()
    ax.plot(dmvals, PDMz,label='p(DM|z)')
    ax.plot(dmvals, PDMz_unbiased,color='orange',label='intrinsic (no bias)')
    ax.plot(dmvals, PDMz_igm_only,color='green',label='DM_IGM only')
    plt.ylim(0,0.006)

    # Limits
    for DM, ls, limit in zip([DM_EG_min, DM_EG_max], ('--', ':'), limits):
        ax.axvline(DM, color='red', ls=ls,
                   label=f'{DM:0.1f} ({limit} c.l.)')

    # Stats
    ax.axvline(DM_EG_median, color='k', ls='--', 
               label=r'median DM$_{\rm EG}$'+ f' = {DM_EG_median:0.1f}')
    ax.axvline(DM_EG_mean, color='k', 
               label=r'mean DM$_{\rm EG}$'+f' = {DM_EG_mean:0.1f}')
    ax.axvline(DM_EG_peak, color='k', ls=':', 
               label=r'peak DM$_{\rm EG}$'+f' = {DM_EG_peak:0.1f}')

    ax.set_xlim(0, DM_EG_max*1.5)

    ax.legend(fontsize=15.)

    ax.set_xlabel(r'DM$_{\rm EG}$')
    ax.set_ylabel('P(DM_EG|z) [Normalized]')
    set_fontsize(ax, 15.)
    
    plt.savefig('other_lines_added.pdf')
        

def parse_args(options=None):
    # test for command-line arguments here
    parser = argparse.ArgumentParser()
    parser.add_argument("z", type=float, help="FRB redshift")
    parser.add_argument('-s','--survey',type=str, default='CRAFT/ICS', help="Name of survey [CRAFT/ICS, PKS/Mb]")
    parser.add_argument("--host_lmean", type=float, default=2.16, help="Log10 mean of DM host contribution (log normal)")
    parser.add_argument("--host_lsigma", type=float, default=0.51, help="Log10 sigma of DM host contribution (log normal)")
    args = parser.parse_args()
    return args

def run():
    pargs = parse_args()
    main(pargs)

def plot_mean_dm():
    from frb.dm import igm
    import numpy as np
    from matplotlib import pyplot as plt
    zmax=2.
    mean_dms,zvals=igm.average_DM(zmax, cumul=True, neval=101)
    for i,dm in enumerate(mean_dms):
        print(zvals[i],dm)
    plt.figure()
    plt.plot(zvals,mean_dms)
    plt.xlabel('z')
    plt.ylabel('Mean DM cosmic')
    plt.savefig('mean_dm.pdf')
    plt.close()

run()
