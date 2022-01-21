""" Examine LL values for the Real FRBs """

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
import pandas

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

    # Load up
    surveys, grids = loading.surveys_and_grids()
    print(grids[0].state)

    vparams = {}
    #vparams[pargs.param] = None
    vparams['lC'] = -0.9

    ll_df = pandas.DataFrame()
    pandas.set_option("display.max_rows", None, 
                      "display.max_columns", None) 
    pandas.set_option('display.precision', 3)

    C,llC,lltot=it.minimise_const_only(
                vparams,grids,surveys, Verbose=False)
    for isurvey, igrid in zip(surveys, grids):
        # Set lC
        igrid.state.FRBdemo.lC = C
        # Grab final LL
        if isurvey.nD==1:
            llsum,lllist,expected,longlist = \
                it.calc_likelihoods_1D(
                    igrid, isurvey, 
                    norm=True,psnr=True, dolist=2)
        elif isurvey.nD==2:
            llsum,lllist,expected,longlist = \
                it.calc_likelihoods_2D(igrid, isurvey, 
                    norm=True,psnr=True, dolist=2)
        elif isurvey.nD==3:
            # mixture of 1 and 2D samples. NEVER calculate Pn twice!
            llsum1,lllist,expected,longlist1 = \
                it.calc_likelihoods_1D(igrid, isurvey, 
                norm=True,psnr=True,dolist=2)
            llsum2,lllist,expected,longlist2 = \
                it.calc_likelihoods_2D(igrid, isurvey, 
                    norm=True,psnr=True, dolist=2, Pn=False)
            lls = llsum1+llsum2
            longlist = np.concatenate([longlist1, longlist2])
        else:
            embed(header='78 of vary1d')

        # Sub table
        sub_df = pandas.DataFrame()
        sub_df['Survey'] = [isurvey.name]*isurvey.NFRB
        sub_df['ID'] = isurvey.frbs['ID']
        sub_df['DM'] = isurvey.frbs['DM'] 
        sub_df['s'] = isurvey.frbs['SNR'] / isurvey.frbs['SNRTHRESH']
        if isurvey.frbs['Z'] is None:
            sub_df['Z'] = [np.nan]*isurvey.NFRB
        else:
            sub_df['Z'] = isurvey.frbs['Z'] 
        sub_df['LL'] = longlist

        # Append
        ll_df = pandas.concat([ll_df, sub_df], 
                              ignore_index=True)

    print(ll_df)



# command-line arguments here
parser = argparse.ArgumentParser()
parser.add_argument('--params',type=str,help="comma separated parameters to change from defaults")
parser.add_argument('--values',type=str,help="comma separated values of the parameters to change from defaults")
#parser.add_argument('--debug', default=False, action='store_true',
#                            help='Debug')
pargs = parser.parse_args()


main(pargs)

'''
# Newest round
python py/chk_ll.py 

'''