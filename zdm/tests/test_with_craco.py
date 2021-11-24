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
from zdm.tests import craco 

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

    isurvey, igrid = craco.load_craco(cosmo=pargs.cosmo,
                                      survey_name=pargs.survey,
                                      NFRB=pargs.nFRB,
                                      lum_func=pargs.lum_func)
    surveys = [isurvey]                                      
    grids = [igrid]

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
        lls_final,alist,expected= it.calc_likelihoods_2D(
                    igrid, isurvey, vparams['lC'],
                    norm=True,psnr=True,dolist=1)
        lls.append(lls_final)
        print(f'{pargs.param}: pval={pval}, C={C}, lltot={lls_final}')

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