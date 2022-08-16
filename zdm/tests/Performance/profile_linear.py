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
from sqlalchemy import false

from zdm import iteration as it
from zdm.craco import loading
from zdm import energetics

from IPython import embed

import datetime

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
    igrid.use_log10 = False
    times = []
    for i in range(10):
        now = datetime.datetime.now()
        igrid.update(vparams, ALL=True) 
        done = datetime.datetime.now()
        print(f'Time to normal loop = {done-now}')
        times.append((done-now))

    average = sum(times, datetime.timedelta(0)) / len(times)
    print(average)
    return (average, igrid.fractions.copy())

# ###################
# Now do linear

def run_linear():
    igrid.array_cum_lf=energetics.array_cum_gamma_linear
    igrid.vector_cum_lf=energetics.vector_cum_gamma_linear

    # An example of how to turn on log10
    igrid.use_log10 = False
    
    times = []

    for i in range(10):
        now = datetime.datetime.now()
        igrid.update(vparams, ALL=True) 
        done = datetime.datetime.now()
        print(f'Time to normal loop = {done-now}')
        times.append((done-now))

    
    average = sum(times, datetime.timedelta(0)) / len(times)
    print(average)
    return (average, igrid.fractions.copy())

def run_linear_log10():
    igrid.array_cum_lf=energetics.array_cum_gamma_linear
    igrid.vector_cum_lf=energetics.vector_cum_gamma_linear

    # An example of how to turn on log10
    igrid.use_log10 = True
    times = []

    
    for i in range(10):
        now = datetime.datetime.now()
        igrid.update(vparams, ALL=True) 
        done = datetime.datetime.now()
        print(f'Time to normal loop = {done-now}')
        times.append((done-now))
   
    
    average = sum(times, datetime.timedelta(0)) / len(times)
    print(average)
    return (average, igrid.fractions.copy())


splineAvg, frac1 = run_spline()
linearAvg, frac2 = run_linear()
linearAvgLog, frac3 = run_linear_log10()



relative_acc_array = np.absolute(frac1 - frac2) / frac1
relative_acc_avg = np.average(relative_acc_array)

relative_acc_array2 = np.absolute(frac1 - frac3) / frac1
relative_acc_avg2 = np.average(relative_acc_array2)

diff_array = np.absolute(relative_acc_array - relative_acc_array2)

# embed()

'''
#TIMEPROFILING
python -m cProfile -o time_profile.prof time_profile.py
'''