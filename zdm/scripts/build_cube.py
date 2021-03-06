""" Build a log-likelihood cube for zdm 
  -- ONLY WORKS FOR CRACO MC 
"""

# It should be possible to remove all the matplotlib calls from this
# but in the current implementation it is not removed.
import argparse
import numpy as np
import os

import pandas

from zdm import iteration as it
from zdm import io
from zdm.craco import loading

from IPython import embed

def main(pargs):
    
    # Clobber?
    if pargs.clobber and os.path.isfile(pargs.opfile):
        os.remove(pargs.opfile)

    ############## Load up ##############
    input_dict=io.process_jfile(pargs.pfile)

    # Deconstruct the input_dict
    state_dict, cube_dict, vparam_dict = it.parse_input_dict(input_dict)

    ############## Initialise ##############
    # ONLY FOR CRACO SO FAR
    survey, grid = loading.survey_and_grid(
        state_dict=state_dict,
        survey_name=pargs.survey,
        NFRB=pargs.NFRB,
        iFRB=pargs.iFRB)

    # Write state to disk
    state_file = pargs.pfile.replace('cube.json', 'state.json')
    grid.state.write(state_file)
    
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
    it.cube_likelihoods([grid],[survey], vparam_dict, 
                                  cube_dict,
                    run,howmany,opfile, starti=starti)
    # Write
    #df = pandas.DataFrame(outputs)
    #df.to_csv(opfile, index=False)
    print(f"Wrote: {opfile}")

def parse_args(options=None):
    # test for command-line arguments here
    parser = argparse.ArgumentParser()
    parser.add_argument('-n','--number',type=int,required=True,help="nth iteration, beginning at 0")
    parser.add_argument('-m','--howmany',type=int,required=True,help="number m to iterate at once")
    parser.add_argument('-p','--pfile',type=str, required=True,help="File defining parameter ranges")
    parser.add_argument('-s','--survey',type=str, required=True,help="Name of CRACO MC survey")
    parser.add_argument('-o','--opfile',type=str,required=True,help="Output file for the data")
    parser.add_argument('--NFRB',type=int,required=False,help="Number of FRBs to analzye")
    parser.add_argument('--iFRB',type=int,default=0,help="Initial FRB to run from")
    parser.add_argument('--clobber', default=False, action='store_true',
                    help="Clobber output file?")
    args = parser.parse_args()
    return args

def run():
    pargs = parse_args()
    main(pargs)

'''
# Test
python py/craco_H0_Emax_cube.py -n 1 -m 100 -o tmp.out --clobber

# 
python py/craco_H0_Emax_cube.py -n 1 -m 250 -o Cubes/craco_H0_Emax_cube0.out --clobber

'''