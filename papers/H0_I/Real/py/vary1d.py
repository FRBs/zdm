""" Examine output on the Real FRBs """

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
import matplotlib
from matplotlib import pyplot as plt

from zdm import iteration as it
import loading

from IPython import embed

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

    surveys, grids = loading.surveys_and_grids(
        lum_func=pargs.lum_func)

    pvals = np.linspace(pargs.min, pargs.max, 
                        pargs.nstep)
    vparams = {}
    vparams[pargs.param] = None
    vparams['lC'] = -0.9

    # DEBUGGING
    #print("WARNING:  REMOVE THE LINE BELOW WHEN DONE DEBUGGING")
    #vparams['lEmax'] = 40.6

    tot_lls = []
    nterms = []  # LL term related to norm (i.e. rates)
    pvterms = []  # LL term related to norm (i.e. rates)
    pvvals = []  # 
    wzvals = []  # 
    for tt, pval in enumerate(pvals):
        vparams[pargs.param] = pval
        C,llC,lltot=it.minimise_const_only(
                    vparams,grids,surveys, Verbose=False)
        # Set lC
        vparams['lC']=C
        ills = []
        for isurvey, igrid in zip(surveys, grids):
            igrid.state.FRBdemo.lC = C
            # Grab final LL
            if isurvey.nD==1:
                lls, alist,expected = it.calc_likelihoods_1D(
                        igrid, isurvey, norm=True,psnr=True,dolist=1)
            elif isurvey.nD==2:
                lls, nterm, pvterm, lpvals, lwz = it.calc_likelihoods_2D(
                    igrid, isurvey, 
                    norm=True,psnr=True, dolist=4)
            elif isurvey.nD==3:
                # mixture of 1 and 2D samples. NEVER calculate Pn twice!
                llsum1,alist1,expected1 = it.calc_likelihoods_1D(
                    igrid, isurvey, 
                    norm=True,psnr=True,dolist=1)
                llsum2,alist2,expected2 = it.calc_likelihoods_2D(
                    igrid, isurvey, 
                    norm=True,psnr=True,dolist=1,Pn=False)
                lls = llsum1+llsum2
            else:
                embed(header='78 of vary1d')
            if pargs.debug:
                print(pval, isurvey.name, lls)
            ills.append(lls)
        # Hold
        tot_lls.append(np.sum(ills))
        #nterms.append(nterm)
        #pvterms.append(pvterm)
        #pvvals.append(lpvals)
        #wzvals.append(lwz)
        print(f'{pargs.param}: pval={pval}, C={C}, lltot={tot_lls[-1]}')

    # Max
    imx = np.nanargmax(tot_lls)
    print(f"Max LL at {pargs.param}={pvals[imx]}")

    # Plot
    plt.clf()
    ax = plt.gca()
    ax.plot(pvals, tot_lls, 'o')
    # Nan
    bad = np.isnan(tot_lls)
    nbad = np.sum(bad)
    if nbad > 0:
        ax.plot(pvals[bad], [np.nanmin(tot_lls)]*nbad, 'x', color='r')
    ax.set_xlabel(pargs.param)
    ax.set_ylabel('LL')
    # Max
    ax.axvline(pvals[imx], color='g', ls='--', label=f'max={pvals[imx]}')
    ax.legend()
    # Save?
    if pargs.opfile is not None:
        plt.savefig(pargs.opfile)
        print(f"Wrote: {pargs.opfile}")
    else:
        plt.show()
    plt.close()

    '''
    # Plot nterm
    plt.clf()
    ax = plt.gca()
    ax.plot(pvals, nterms, 'o')
    ax.set_xlabel(pargs.param)
    ax.set_ylabel('nterm')
    plt.savefig('nterms.png')
    plt.close()

    # Plot nterm
    plt.clf()
    ax = plt.gca()
    ax.plot(pvals, pvterms, 'o')
    ax.set_xlabel(pargs.param)
    ax.set_ylabel('pvterm')
    plt.savefig('pvterms.png')
    plt.close()
    '''

# command-line arguments here
parser = argparse.ArgumentParser()
parser.add_argument('param',type=str,help="paramter to test on")
parser.add_argument('min',type=float,help="minimum value")
parser.add_argument('max',type=float,help="maximum value")
parser.add_argument('--nstep',type=int,default=10,required=False,help="number of steps")
parser.add_argument('-o','--opfile',type=str,required=False,help="Output file for the data")
parser.add_argument('--lum_func',type=int,default=0, required=False,help="Luminosity function (0=power-law, 1=gamma)")
parser.add_argument('--debug', default=False, action='store_true',
                            help='Debug')
pargs = parser.parse_args()


main(pargs)

'''
# Newest round
python py/vary1d.py H0 60. 80. --nstep 50 -o Plots/Real_H0.png 

'''