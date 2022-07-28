""" Time profiling """

######
# first run this to generate surveys and parameter sets, by 
# setting NewSurveys=True NewGrids=True
# Then set these to False and run with command line arguments
# to generate *many* outputs
#####

import numpy as np
from matplotlib import pyplot as plt
import mpmath as mp

from zdm import iteration as it
from zdm.craco import loading
from zdm import energetics

from IPython import embed


# Default parameters
pparam = 'H0'
psurvey = 'CRACO_std_May2022'
pnFRB = 100
piFRB = 100
plum_func = 2 # Spline
pval = 72.  # H0

# Load up
isurvey, igrid = loading.survey_and_grid(survey_name=psurvey,
                                    NFRB=pnFRB,
                                    iFRB=piFRB,
                                    lum_func=plum_func)

#pvals = np.linspace(pargs.min, pargs.max, pargs.nstep)
vparams = {}
vparams[pparam] = None
vparams['lC'] = -0.9

vparams[pparam] = pval


# C,llC=it.minimise_const_only(vparams,grids,surveys, Verbose=False)

def run_spline():
    igrid.array_cum_lf=energetics.array_cum_gamma_spline
    igrid.vector_cum_lf=energetics.vector_cum_gamma_spline

    igrid.update(vparams, ALL=True) 
    return igrid.fractions.copy()

# ###################
# Now do linear

def run_linear():
    igrid.array_cum_lf=energetics.array_cum_gamma_linear
    igrid.vector_cum_lf=energetics.vector_cum_gamma_linear

    # An example of how to turn on log10
    # igrid.use_log10 = False
    
    igrid.update(vparams, ALL=True) 
    return igrid.fractions.copy()

def run_linear_log10():
    igrid.array_cum_lf=energetics.array_cum_gamma_linear
    igrid.vector_cum_lf=energetics.vector_cum_gamma_linear

    # An example of how to turn on log10
    igrid.use_log10 = True
    
    igrid.update(vparams, ALL=True) 
    return igrid.fractions.copy()



frac1 = run_spline()
frac2 = run_linear()


relative_acc_array = np.absolute(frac1 - frac2) / frac1
relative_acc_avg = np.average(relative_acc_array)

# doing something wrong bc there's no difference?
# print(relative_acc_array)
# print(relative_acc_avg)
'''
#TIMEPROFILING
python -m cProfile -o time_profile.prof time_profile.py
'''