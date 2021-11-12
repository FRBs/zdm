""" Run tests with CRACO FRBs """

######
# first run this to generate surveys and parameter sets, by 
# setting NewSurveys=True NewGrids=True
# Then set these to False and run with command line arguments
# to generate *many* outputs
#####

# It should be possible to remove all the matplotlib calls from this
# but in the current implementation it is not removed.
import argparse
import numpy as np
import os
from pkg_resources import resource_filename
import matplotlib
from matplotlib import pyplot as plt

from astropy.cosmology import Planck15, Planck18

from zdm import survey
from zdm import parameters
from zdm import cosmology as cos
from zdm import misc_functions
from zdm import iteration as it
from zdm import io

from IPython import embed

import pickle

matplotlib.rcParams['image.interpolation'] = None

defaultsize=14
ds=4
font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : defaultsize}
matplotlib.rc('font', **font)

#import igm
defaultsize=14
ds=4
font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : defaultsize}
matplotlib.rc('font', **font)

def main(pargs):

    #psetmins,psetmaxes,nvals=misc_functions.process_pfile(Cube[2])
    #input_dict= io.process_jfile(Cube[2])
    #state_dict, cube_dict, vparam_dict = it.parse_input_dict(input_dict)

    state_dict = dict(FRBdemo=dict(alpha_method=1),
                      cosmo=dict(fix_Omega_b_h2=True))

    ############## Initialise parameters ##############
    state = parameters.State()

    # Clancy values
    state.energy.lEmax=41.4 
    state.energy.alpha=0.65 
    state.energy.gamma=-1.01 
    state.FRBdemo.sfr_n=0.73 
    state.host.lmean=2.18
    state.host.lsigma=0.48

    state.energy.luminosity_function = pargs.lum_func

    state.update_param_dict(state_dict)
    
    ############## Initialise cosmology ##############
    if pargs.cosmo == 'Planck18':
        state.set_astropy_cosmo(Planck18)
    elif pargs.cosmo == 'Planck15':
        state.set_astropy_cosmo(Planck15)
    else:
        raise IOError("Bad astropy cosmology")
        
    cos.set_cosmology(state)
    cos.init_dist_measures()
    
    # get the grid of p(DM|z)
    zDMgrid, zvals,dmvals = misc_functions.get_zdm_grid(
        state, new=True, plot=False, method='analytic',
        datdir=resource_filename('zdm', 'GridData'))
    
    ############## Initialise surveys ##############
    
    
    isurvey = survey.load_survey(pargs.survey, state, dmvals,
                                 NFRB=pargs.nFRB)
    surveys = [isurvey]

    # generates zdm grid
    grids = misc_functions.initialise_grids(
        surveys,zDMgrid, zvals, dmvals, state, wdist=True)
    print("Initialised grids")

    pvals = np.linspace(pargs.min, pargs.max, pargs.nstep)
    vparams = {}
    vparams[pargs.param] = None
    vparams['lC'] = -0.9

    lls = []
    for tt, pval in enumerate(pvals):
        vparams[pargs.param] = pval
        C,llC,lltot=it.minimise_const_only(
                    vparams,grids,surveys, Verbose=False)
        vparams['lC']=C
        if tt == 0: # do it again as the initial guess is often junk!
            C,llC,lltot=it.minimise_const_only(
                    vparams,grids,surveys, Verbose=False)
            vparams['lC']=C
        #likeli =  it.cube_likelihoods(grids,surveys, vparam_dict, cube_dict,
        #            run,howmany,opfile, starti=starti,clone=clone)
        lls.append(lltot)
        print(f'{pargs.param}: pval={pval}, C={C}, lltot={lltot}')

    # Max
    imx = np.nanargmax(lls)
    print(f"Max LL at {pargs.param}={pvals[imx]}")

    # Plot
    plt.clf()
    ax = plt.gca()
    ax.plot(pvals, lls, 'o')
    # Nan
    bad = np.isnan(lls)
    nbad = np.sum(bad)
    if nbad > 0:
        ax.plot(pvals[bad], [np.nanmin(lls)]*nbad, 'x', color='r')
    ax.set_xlabel(pargs.param)
    ax.set_ylabel('LL')
    # Max
    ax.axvline(pvals[imx], color='g', ls='--', label=f'max={pvals[imx]}')
    ax.legend()
    # Save?
    if pargs.opfile is not None:
        plt.savefig(pargs.opfile)
    else:
        plt.show()
    plt.close()
        


# command-line arguments here
parser = argparse.ArgumentParser()
parser.add_argument('param',type=str,help="paramter to test on")
parser.add_argument('min',type=float,help="minimum value")
parser.add_argument('max',type=float,help="maximum value")
parser.add_argument('--nstep',type=int,default=10,required=False,help="number of steps")
parser.add_argument('--nFRB',type=int,default=1000,required=False,help="number of FRBs to analyze")
parser.add_argument('-o','--opfile',type=str,required=False,help="Output file for the data")
parser.add_argument('--cosmo',type=str,default='Planck18', required=False,help="Output file for the data")
parser.add_argument('--survey',type=str,default='CRAFT/CRACO_1_5000',
                    required=False,help="Output file for the data")
parser.add_argument('--lum_func',type=int,default=0, required=False,help="Luminosity function (0=power-law, 1=gamma)")
pargs = parser.parse_args()


main(pargs)

'''
python test_with_craco.py sfr_n 0.2 2. --nstep 100 --nFRB 1000 --cosmo Planck15 -o CRACO_1000_sfr_n.png
python test_with_craco.py gamma -1.5 -0.8 --nstep 30 --nFRB 1000 --cosmo Planck15 -o CRACO_1000_gamma.png
python test_with_craco.py alpha 0.0 1.0 --nstep 50 --nFRB 1000 --cosmo Planck15 -o CRACO_1000_alpha.png
python test_with_craco.py lEmax 41. 43. --nstep 50 --nFRB 1000 --cosmo Planck15 -o CRACO_1000_lEmax.png
python test_with_craco.py H0 60. 80. --nstep 50 --nFRB 1000 --cosmo Planck15 -o CRACO_1000_H0.png
python test_with_craco.py lmean 1.9 2.5  --nstep 30 --nFRB 1000 --cosmo Planck15 -o CRACO_1000_lmean.png
'''