""" Time profiling """

######
# first run this to generate surveys and parameter sets, by 
# setting NewSurveys=True NewGrids=True
# Then set these to False and run with command line arguments
# to generate *many* outputs
#####

import numpy as np
from matplotlib import pyplot as plt

from zdm import iteration as it
from zdm.craco import loading

from IPython import embed


# Default parameters
pparam = 'H0'
psurvey = 'CRACO_std_May2022'
pnFRB = 100
piFRB = 100
plum_func = 2
pval = 72.  # H0

# Load up
isurvey, igrid = loading.survey_and_grid(survey_name=psurvey,
                                    NFRB=pnFRB,
                                    iFRB=piFRB,
                                    lum_func=plum_func)
surveys = [isurvey]                                      
grids = [igrid]

#pvals = np.linspace(pargs.min, pargs.max, pargs.nstep)
vparams = {}
vparams[pparam] = None
vparams['lC'] = -0.9

vparams[pparam] = pval

# PROFILE HERE PLEASE
C,llC=it.minimise_const_only(vparams,grids,surveys, Verbose=False)