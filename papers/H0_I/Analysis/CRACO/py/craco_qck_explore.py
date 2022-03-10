# imports
from importlib import reload
import numpy as np
import sys, os

from zdm import analyze_cube
from zdm import iteration as it
from zdm import io
from zdm.craco import loading


#sys.path.append(os.path.abspath("../../Figures/py"))

scube = 'mini'
outdir = 'Mini/'

# Load
npdict = np.load(f'Cubes/craco_{scube}_cube.npz')

ll_cube = npdict['ll']
lC_cube = npdict['lC']

# Deal with Nan
ll_cube[np.isnan(ll_cube)] = -1e99
params = npdict['params']

# Cube parameters
############## Load up ##############
pfile = f'Cubes/craco_{scube}_cube.json'
input_dict=io.process_jfile(pfile)

# Deconstruct the input_dict
state_dict, cube_dict, vparam_dict = it.parse_input_dict(input_dict)

# Run Bayes

# Offset by max
ll_cube = ll_cube - np.max(ll_cube)

uvals,vectors,wvectors = analyze_cube.get_bayesian_data(ll_cube)

analyze_cube.do_single_plots(uvals,vectors,wvectors, params, 
                             vparams_dict=vparam_dict, outdir=outdir)

print(f"Wrote figures to {outdir}")