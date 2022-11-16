""" Build a log-likelihood cube for zdm for Real FRBs! 
"""

# It should be possible to remove all the matplotlib calls from this
# but in the current implementation it is not removed.
import argparse
import numpy as np
import os

from zdm import iteration as it
from zdm import io
from zdm import real_loading

from IPython import embed

def main(pargs):
    

    # Clobber?
    if pargs.clobber and os.path.isfile(pargs.opfile):
        os.remove(pargs.opfile)

    ############## Load up ##############
    input_dict=io.process_jfile(pargs.pfile)

    # Deconstruct the input_dict
    state_dict, cube_dict, vparam_dict = it.parse_input_dict(input_dict)

    # State
    state = real_loading.set_state()
    state.update_param_dict(state_dict)

    ############## Initialise ##############
    surveys, grids = real_loading.surveys_and_grids(init_state=state)

    # Write state to disk
    state_file = pargs.pfile.replace('cube.json', 'state.json')
    state.write(state_file)
    
    # Set what portion of the Cube we are generating 
    run=pargs.number
    howmany=pargs.howmany
    opfile=pargs.opfile
    
    # checks to see if the file is already there, and how many iterations have been performed
    starti=it.check_cube_opfile(run, howmany, opfile)
    print("starti is ",starti)
    if starti==howmany:
        print("Done everything!")
        pass
    #
    it.cube_likelihoods(grids, surveys, vparam_dict, cube_dict,
                    run, howmany, opfile, starti=starti)
        

def parse_args(options=None):
    # test for command-line arguments here
    parser = argparse.ArgumentParser()
    parser.add_argument('-n','--number',type=int,required=True,help="nth iteration, beginning at 0")
    parser.add_argument('-m','--howmany',type=int,required=True,help="number m to iterate at once")
    parser.add_argument('-p','--pfile',type=str, required=True,help="File defining parameter ranges")
    parser.add_argument('-o','--opfile',type=str,required=True,help="Output file for the data")
    parser.add_argument('--clobber', default=False, action='store_true',
                    help="Clobber output file?")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    pargs = parse_args()
    main(pargs)