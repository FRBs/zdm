""" Figures for H0 paper I"""
import os, sys
from typing import IO
import numpy as np
from numpy.lib.function_base import percentile
import scipy
from scipy import stats

import argparse

import matplotlib as mpl
import matplotlib.gridspec as gridspec
from matplotlib import pyplot as plt

from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

mpl.rcParams['font.family'] = 'stixgeneral'

import pandas
import seaborn as sns

import h5py

from zdm.craco import loading
from zdm import pcosmic
from zdm import figures

from IPython import embed

sys.path.append(os.path.abspath("../Analysis/py"))
import analy_H0_I

def fig_craco_fiducial(outfile='fig_craco_fiducial.png',
                zmax=2,DMmax=2000,
                log=True,
                label='$\\log_{10} \; p(DM_{\\rm EG},z)$',
                Aconts=[0.01, 0.1, 0.5],
                cmap='jet', show=False, figsize=None,
                vmnx=(None,None),
                grid=None, survey=None):
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
        survey, grid = loading.survey_and_grid(
            survey_name=analy_H0_I.fiducial_survey,
            NFRB=100, lum_func=1)

    # Unpack
    full_zDMgrid, zvals, dmvals = grid.rates, grid.zvals, grid.dmvals
    FRBZ=survey.frbs['Z']
    FRBDM=survey.DMEGs
    
    ##### imshow of grid #######
    plt.figure(figsize=figsize)
    ax1=plt.axes()
    plt.sca(ax1)
    
    plt.xlabel('z')
    plt.ylabel('${\\rm DM}_{\\rm EG}$')
    #plt.title(title+str(H0))
    
    # Cut down grid
    zvals, dmvals, zDMgrid = figures.proc_pgrid(
        full_zDMgrid, 
        zvals, (0, zmax),
        dmvals, (0, DMmax))
    ddm=dmvals[1]-dmvals[0]
    dz=zvals[1]-zvals[0]
    nz, ndm = zDMgrid.shape

    # Contours
    alevels = figures.find_Alevels(full_zDMgrid, Aconts, log=True)
        
    # Ticks
    tvals, ticks = figures.ticks_pgrid(zvals)# , fmt='str4')
    plt.xticks(tvals, ticks)
    tvals, ticks = figures.ticks_pgrid(dmvals, fmt='int')# , fmt='str4')
    plt.yticks(tvals, ticks)

    # Image 
    im=plt.imshow(zDMgrid.T,cmap=cmap,origin='lower', 
                  vmin=vmnx[0], vmax=vmnx[1],
                  interpolation='None',
                  aspect='auto')
    
    styles=['--','-.',':']
    ax=plt.gca()
    cs=ax.contour(zDMgrid.T,levels=alevels,origin='lower',colors="white",linestyles=styles)

    ax=plt.gca()
    
    muDMhost=np.log(10**grid.state.host.lmean)
    sigmaDMhost=np.log(10**grid.state.host.lsigma)
    meanHost = np.exp(muDMhost + sigmaDMhost**2/2.)
    medianHost = np.exp(muDMhost) 
    print(f"Host: mean={meanHost}, median={medianHost}")
    plt.ylim(0,ndm-1)
    plt.xlim(0,nz-1)
    zmax=zvals[-1]
    nz=zvals.size
    #DMbar, zeval = igm.average_DM(zmax, cumul=True, neval=nz+1)
    DM_cosmic = pcosmic.get_mean_DM(zvals, grid.state)
    
    #idea is that 1 point is 1, hence...
    zeval = zvals/dz
    DMEG_mean = (DM_cosmic+meanHost)/ddm
    DMEG_median = (DM_cosmic+medianHost)/ddm
    plt.plot(zeval,DMEG_mean,color='gray',linewidth=2,
                label='Macquart relation (mean)')
    plt.plot(zeval,DMEG_median,color='gray',
                linewidth=2, ls='--',
                label='Macquart relation (median)')
    l=plt.legend(loc='lower right',fontsize=12)
    #l=plt.legend(bbox_to_anchor=(0.2, 0.8),fontsize=8)
    #for text in l.get_texts():
        #	text.set_color("white")
    
    # limit to a reasonable range if logscale
    if log and vmnx[0] is None:
        themax=zDMgrid.max()
        themin=int(themax-4)
        themax=int(themax)
        plt.clim(themin,themax)
    
    ##### add FRB host galaxies at some DM/redshift #####
    if FRBZ is not None:
        iDMs=FRBDM/ddm
        iZ=FRBZ/dz
        # Restrict to plot range
        gd = (FRBDM < DMmax) & (FRBZ < zmax)
        plt.plot(iZ[gd],iDMs[gd],'ko',linestyle="",markersize=2.)

    cbar=plt.colorbar(im,fraction=0.046, shrink=1.2,aspect=15,pad=0.05)
    cbar.set_label(label)
    plt.tight_layout()
    
    if show:
        plt.show()
    else:
        plt.savefig(outfile, dpi=300)
    plt.close()
    print(f"Wrote: {outfile}")


def fig_craco_varyH0_zDM(outfile,
                zmax=2,DMmax=1500,
                norm=2, other_param='Emax',
                Aconts=[0.05]):
    # Generate the grid
    survey, grid = loading.survey_and_grid(
        survey_name='CRACO_alpha1_Planck18_Gamma',
        NFRB=100, lum_func=1)
    fiducial_Emax = grid.state.energy.lEmax
    fiducial_F = grid.state.IGM.F

    plt.figure()
    ax1=plt.axes()

    plt.sca(ax1)
    
    plt.xlabel('z')
    plt.ylabel('${\\rm DM}_{\\rm EG}$')
    #plt.title(title+str(H0))

    if other_param == 'Emax':
        H0_values = [60., 70., 80., 80.]
        other_values = [0., 0., 0., -0.1]
        lstyles = ['-', '-', '-', ':']
    elif other_param == 'F':
        H0_values = [60., 70., 80., 60.]
        other_values = [fiducial_F, fiducial_F, fiducial_F, 0.5]
        lstyle = '-'

    # Loop on grids
    legend_lines = []
    labels = []
    for H0, scl, lstyle, clr in zip(
                      H0_values,
                      other_values,
                      lstyles,
                      ['b', 'k','r', 'gray']):

        # Update grid
        vparams = {}
        vparams['H0'] = H0
        if other_param == 'Emax':
            vparams['lEmax'] = fiducial_Emax + scl
        elif other_param == 'F':
            vparams['F'] = scl
        grid.update(vparams)

        # Unpack
        full_zDMgrid, zvals, dmvals = grid.rates.copy(), grid.zvals.copy(), grid.dmvals.copy()
    
        # currently this is "per cell" - now to change to "per DM"
        # normalises the grid by the bin width, i.e. probability per bin, not probability density
        
        # checks against zeros for a log-plot

        zvals, dmvals, zDMgrid = figures.proc_pgrid(
            full_zDMgrid, 
            zvals, (0, zmax),
            dmvals, (0, DMmax))

        # Contours
        alevels = figures.find_Alevels(full_zDMgrid, Aconts)
        
        # sets the x and y tics	
        tvals, ticks = figures.ticks_pgrid(zvals)# , fmt='str4')
        plt.xticks(tvals, ticks)
        tvals, ticks = figures.ticks_pgrid(dmvals, fmt='int')# , fmt='str4')
        plt.yticks(tvals, ticks)

        ax=plt.gca()
        cs=ax.contour(zDMgrid.T,levels=alevels,
                      origin='lower',colors=[clr],
                      linestyles=lstyle)
        leg, _ = cs.legend_elements()
        legend_lines.append(leg[0])

        # Label
        if other_param == 'Emax':
            labels.append(r"$H_0 = $"+f"{H0}, log "+r"$E_{\rm max}$"+f"= {vparams['lEmax']}")
        elif other_param == 'F':
            labels.append(r"$H_0 = $"+f"{H0}, F = {vparams['F']}")

    ###### gets decent axis labels, down to 1 decimal place #######
    ax=plt.gca()
    ax.legend(legend_lines, labels, loc='lower right')

    # Ticks
    labels = [item.get_text() for item in ax.get_xticklabels()]
    for i in np.arange(len(labels)):
        labels[i]=labels[i][0:4]
    ax.set_xticklabels(labels)
    labels = [item.get_text() for item in ax.get_yticklabels()]
    for i in np.arange(len(labels)):
        if '.' in labels[i]:
            labels[i]=labels[i].split('.')[0]
    ax.set_yticklabels(labels)
    ax.yaxis.labelpad = 0
        

    # Finish
    plt.tight_layout()
    plt.savefig(outfile, dpi=300)
    plt.close()
    print(f"Wrote: {outfile}")


def fig_craco_varyH0_other(outfile, params,
                zmax=2,DMmax=1500,
                smax=25., 
                other_param='Emax',
                Aconts=[0.05], debug:bool=False):

    if other_param == 'Emax':
        H0_values = [60., 70., 80., 80.]
        other_values = [41.4, 41.4, 41.4, 41.3]
        lstyles = ['-', '-', '-', ':']
    elif other_param == 'F':
        H0_values = [60., 70., 80., 60.]
        other_values = [fiducial_F, fiducial_F, fiducial_F, 0.5]
        lstyle = '-'

    plt.figure()
    ax1=plt.axes()

    plt.sca(ax1)
    
    if params == 'sDM':
        plt.xlabel(r'DM$_{\rm EG}$')
    else:
        plt.xlabel(r'$z$')
    plt.ylabel(r'$s$')

    # Loop on grids
    legend_lines = []
    labels = []
    first = True
    for H0, lEmax, lstyle, clr in zip(
                      H0_values,
                      other_values,
                      lstyles,
                      ['b', 'k','r', 'gray']):

        # Unpack
        grid_file = f'../Analysis/GridData/p{params}_H0{int(H0)}_Emax{lEmax}.npz'
        print(f"Loading: {grid_file}")
        data = np.load(grid_file)
        if params == 'sDM':
            full_pgrid = data['psDM']
            dmvals = data['dmvals']
        else:
            full_pgrid = data['psz']
            zvals = data['zvals']
        snrs = data['snrs']
    
        # Process full grid
        if params == 'sDM':
            snrs, dmvals, cut_pgrid = figures.proc_pgrid(full_pgrid, 
                snrs[0:-1], (0, smax),
                dmvals, (0, DMmax))
        else:
            snrs, zvals, cut_pgrid = figures.proc_pgrid(full_pgrid, 
                snrs[0:-1], (0, smax),
                zvals, (0, zmax))

        # Contours
        alevels = figures.find_Alevels(full_pgrid, Aconts)

        if first:
            if debug:
                im=plt.imshow(cut_pgrid,cmap='jet',origin='lower', 
                    interpolation='None',
                    # extent=[0., 2, 0, 2000.],
                vmin=-30.,
                    aspect='auto')
        
            # sets the x and y tics	
            if params == 'sz':
                tvals, ticks = figures.ticks_pgrid(zvals)# , fmt='str4')
            else:
                tvals, ticks = figures.ticks_pgrid(dmvals, fmt='int')
            plt.xticks(tvals, ticks)
            tvals, ticks = figures.ticks_pgrid(snrs, fmt='str4')
            plt.yticks(tvals, ticks)
            #
            first = False

        ax=plt.gca()
        cs=ax.contour(cut_pgrid,levels=alevels,
                      origin='lower',colors=[clr],
                      linestyles=lstyle)
        leg, _ = cs.legend_elements()
        legend_lines.append(leg[0])

        # Label
        if other_param == 'Emax':
            labels.append(r"$H_0 = $"+f"{H0}, log "+r"$E_{\rm max}$"+f"= {lEmax}")
        elif other_param == 'F':
            labels.append(r"$H_0 = $"+f"{H0}, F = {vparams['F']}")

    ###### gets decent axis labels, down to 1 decimal place #######
    ax=plt.gca()
    ax.legend(legend_lines, labels, loc='upper right')

    # Ticks
    labels = [item.get_text() for item in ax.get_xticklabels()]
    for i in np.arange(len(labels)):
        labels[i]=labels[i][0:4]
    ax.set_xticklabels(labels)
    labels = [item.get_text() for item in ax.get_yticklabels()]
    for i in np.arange(len(labels)):
        if '.' in labels[i]:
            labels[i]=labels[i].split('.')[0]
    ax.set_yticklabels(labels)
    ax.yaxis.labelpad = 0
        

    # Finish
    plt.tight_layout()
    plt.savefig(outfile, dpi=300)
    plt.close()
    print(f"Wrote: {outfile}")

def fig_craco_H0vsEmax(outfile='fig_craco_H0vsEmax.png'):
    # Load the cube
    cube_out = np.load('../Analysis/Cubes/craco_H0_Emax_cube.npz')
    ll = cube_out['ll'] # log10

    # Slurp
    lEmax = cube_out['lEmax']
    H0 = cube_out['H0']
    #
    dE = lEmax[1]-lEmax[0]
    dH = H0[1] - H0[0]
        
    # Normalize
    ll -= ll.max()

    # Plot
    plt.clf()
    ax = plt.gca()

    im=plt.imshow(ll.T,cmap='jet',origin='lower', 
                    interpolation='None', extent=[40.4-dE/2, 43.4+dE/2, 
                                                  60.-dH/2, 80+dH/2],
                aspect='auto', vmin=-4.
                )#aspect=aspect)
    # Color bar
    cbar=plt.colorbar(im,fraction=0.046, shrink=1.2,aspect=15,pad=0.05)
    cbar.set_label(r'$\Delta$ Log10 Likelihood')
    #
    ax.set_xlabel('log Emax')
    ax.set_ylabel('H0 (km/s/Mpc)')
    plt.savefig(outfile, dpi=200)
    print(f"Wrote: {outfile}")


def fig_craco_H0vsF(outfile='fig_craco_H0vsF.png'):
    # Load the cube
    cube_out = np.load('../Analysis/Cubes/craco_H0_F_cube.npz')
    ll = cube_out['ll'] # log10

    # Slurp
    F = cube_out['F']
    H0 = cube_out['H0']
    #
    dF = F[1]-F[0]
    dH = H0[1] - H0[0]
        
    # Normalize
    ll -= ll.max()

    # Plot
    plt.clf()
    ax = plt.gca()

    im=plt.imshow(ll.T,cmap='jet',origin='lower', 
                    interpolation='None', extent=[0.1-dF/2, 0.5+dF/2, 
                                                  60.-dH/2, 80+dH/2],
                aspect='auto', vmin=-4.
                )#aspect=aspect)
    # Color bar
    cbar=plt.colorbar(im,fraction=0.046, shrink=1.2,aspect=15,pad=0.05)
    cbar.set_label(r'$\Delta$ Log10 Likelihood')
    #
    ax.set_xlabel('F')
    ax.set_ylabel('H0 (km/s/Mpc)')
    plt.savefig(outfile, dpi=200)
    print(f"Wrote: {outfile}")

#### ########################## #########################
def main(pargs):

    # Fiducial CRACO
    if pargs.figure == 'fiducial':
        fig_craco_fiducial()

    # Vary H0, Emax
    if pargs.figure == 'varyH0E_zDM':
        fig_craco_varyH0_zDM(outfile='fig_craco_varyH0E_zDM.png',
                         other_param='Emax')
    if pargs.figure == 'varyH0E_sz':
        fig_craco_varyH0_other('fig_craco_varyH0E_sz.png',
            'sz', other_param='Emax')
    if pargs.figure == 'varyH0E_sDM':
        fig_craco_varyH0_other('fig_craco_varyH0E_sDM.png',
            'sDM', other_param='Emax', DMmax=2000.)


    # Vary H0, F
    if pargs.figure == 'varyH0F':
        fig_craco_varyH0_zDM(outfile='fig_craco_varyH0F.png',
                         other_param='F')


    # H0 vs. Emax
    if pargs.figure == 'H0vsEmax':
        fig_craco_H0vsEmax()

    # H0 vs. F
    if pargs.figure == 'H0vsF':
        fig_craco_H0vsF()


def parse_option():
    """
    This is a function used to parse the arguments for figure making
    
    Returns:
        args: (dict) dictionary of the arguments.
    """
    parser = argparse.ArgumentParser("zdm H0 I Figures")
    parser.add_argument("figure", type=str, 
                        help="function to execute: ('fiducial, 'varyH0', 'H0vsEmax')")
    #parser.add_argument('--cmap', type=str, help="Color map")
    #parser.add_argument('--distr', type=str, default='normal',
    #                    help='Distribution to fit [normal, lognorm]')
    args = parser.parse_args()
    
    return args

# Command line execution
if __name__ == '__main__':

    pargs = parse_option()
    main(pargs)


# python py/figs_zdm_H0_I.py fiducial
# python py/figs_zdm_H0_I.py varyH0E_zDM
# python py/figs_zdm_H0_I.py H0vsEmax
# python py/figs_zdm_H0_I.py H0vsF
# python py/figs_zdm_H0_I.py varyH0F
# python py/figs_zdm_H0_I.py varyH0E_sz
# python py/figs_zdm_H0_I.py varyH0E_sDM