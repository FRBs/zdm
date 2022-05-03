# imports
from importlib import reload
import numpy as np
import sys, os

from zdm import analyze_cube
from zdm import iteration as it
from zdm import io
from zdm.craco import loading


#sys.path.append(os.path.abspath("../../Figures/py"))

def main(pargs):
    jroot = None
    if pargs.run == 'mini':
        scube = 'mini' 
        outdir = 'Mini/'
    elif pargs.run == 'full':
        scube = 'full' 
        outdir = 'Full/'
    elif pargs.run == 'full400':
        scube = '400_full' 
        jroot = 'full' 
        outdir = 'Full400/'

    if jroot is None:
        jroot = scube


    # Load
    npdict = np.load(f'Cubes/craco_{scube}_cube.npz')

    ll_cube = npdict['ll']

    # Deal with Nan
    ll_cube[np.isnan(ll_cube)] = -1e99
    params = npdict['params']

    # Cube parameters
    ############## Load up ##############
    pfile = f'Cubes/craco_{jroot}_cube.json'
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

def parse_option():
    """
    This is a function used to parse the arguments in the training.
    
    Returns:
        args: (dict) dictionary of the arguments.
    """
    import argparse

    parser = argparse.ArgumentParser("Slurping the cubes")
    parser.add_argument("run", type=str, help="Run to slurp")
    #parser.add_argument('--debug', default=False, action='store_true',
    #                    help='Debug?')
    args = parser.parse_args()
    
    return args

# Command line execution
if __name__ == '__main__':

    pargs = parse_option()
    main(pargs)

#  python py/slurp_craco_cubes.py mini