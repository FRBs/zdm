""" Script to build the build script! """

# It should be possible to remove all the matplotlib calls from this
# but in the current implementation it is not removed.
import argparse
import numpy as np
import os

from zdm import survey
from zdm import parameters
from zdm import cosmology as cos
from zdm import misc_functions
from zdm import iteration as it
from zdm import io
from zdm.craco import loading

import analy_H0_I

from IPython import embed

def main(pargs):
    
    ############## Load up ##############
    input_dict=io.process_jfile(pargs.pfile)

    # Deconstruct the input_dict
    state_dict, cube_dict, vparam_dict = it.parse_input_dict(input_dict)

    npoints = np.array([item['n'] for key, item in vparam_dict.items()])
    ntotal = np.prod(np.abs(npoints))

    with open(pargs.opfile) as f:    

# test for command-line arguments here
parser = argparse.ArgumentParser()
parser.add_argument('-n','--ncpu',type=int, required=True,help="Number of CPUs to run on")
parser.add_argument('-p','--pfile',type=str, required=True,help="File defining parameter ranges")
parser.add_argument('-o','--opfile',type=str,required=True,help="Output file for the data")
args = parser.parse_args()


main(args)

'''
# Test
python py/craco_H0_Emax_cube.py -n 1 -m 100 -o tmp.out --clobber

# 
python py/craco_H0_Emax_cube.py -n 1 -m 250 -o Cubes/craco_H0_Emax_cube0.out --clobber

'''