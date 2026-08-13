""" Mini Cube for full CRACO run """

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
from zdm.MC_sample import loading

import analy_H0_I

from IPython import embed

def main(pargs, outdir='Cubes/'):
    
    # Clobber?
    if args.clobber and os.path.isfile(pargs.opfile):
        os.remove(pargs.opfile)

    ############## Load up ##############
    pfile = pargs.pfile if pargs.pfile is not None else 'Cubes/craco_mini_cube.json'
    input_dict=io.process_jfile(pfile)

    # Deconstruct the input_dict
    state_dict, cube_dict, vparam_dict = it.parse_input_dict(input_dict)

    ############## Initialise ##############
    survey, grid = loading.survey_and_grid(
        state_dict=state_dict,
        survey_name=analy_H0_I.fiducial_survey,
        NFRB=100)

    # Write state to disk
    state_file = pfile.replace('cube.json', 'state.json')
    grid.state.write(state_file)
    
    if not os.path.exists(outdir):
        os.mkdir(outdir)

    # Set what portion of the Cube we are generating 
    run=pargs.number
    howmany=pargs.howmany
    opfile=pargs.opfile
    
    # checks to see if the file is already there, and how many iterations have been performed
    starti=it.check_cube_opfile(run,howmany,opfile)
    print("starti is ",starti)
    if starti==howmany:
        print("Done everything!")
        pass
    #
    it.cube_likelihoods([grid],[survey], vparam_dict, cube_dict,
                    run,howmany,opfile, starti=starti)
        

# test for command-line arguments here
parser = argparse.ArgumentParser()
parser.add_argument('-n','--number',type=int,required=False,help="nth iteration, beginning at 0")
parser.add_argument('-m','--howmany',type=int,required=False,help="number m to iterate at once")
parser.add_argument('-p','--pfile',type=str, required=False,help="File defining parameter ranges")
parser.add_argument('-o','--opfile',type=str,required=False,help="Output file for the data")
parser.add_argument('--clobber', default=False, action='store_true',
                    help="Clobber output file?")
args = parser.parse_args()


main(args)

'''
# Test
python py/craco_H0_Emax_cube.py -n 1 -m 100 -o tmp.out --clobber

# 
python py/craco_H0_Emax_cube.py -n 1 -m 250 -o Cubes/craco_H0_Emax_cube0.out --clobber

'''