""" Script to build the build script! """

# It should be possible to remove all the matplotlib calls from this
# but in the current implementation it is not removed.
import argparse
import numpy as np
import os

from zdm import iteration as it
from zdm import io

from IPython import embed

def main(pargs):
    
    ############## Load up ##############
    input_dict=io.process_jfile(pargs.pfile)

    # Deconstruct the input_dict
    state_dict, cube_dict, vparam_dict = it.parse_input_dict(input_dict)

    npoints = np.array([item['n'] for key, item in vparam_dict.items()])
    ntotal = int(np.prod(np.abs(npoints)))

    nper_cpu = ntotal // pargs.ncpu
    if int(ntotal/pargs.ncpu) != nper_cpu:
        raise IOError(f"Ncpu={pargs.ncpu} must divide evenly into ntotal={ntotal}")


    with open(pargs.bfile, 'w') as f:    
        for kk in range(pargs.ncpu):
            outfile = pargs.opfile.replace('.out', f'{kk+1}.out')
            line = f'zdm_build_cube -n {kk+1} -m {nper_cpu} -o {outfile} -s CRAFT_CRACO_MC_alpha1_gamma_1000 --clobber -p {pargs.pfile} & \n'
            f.write(line)

# test for command-line arguments here
parser = argparse.ArgumentParser()
parser.add_argument('-n','--ncpu',type=int, required=True,help="Number of CPUs to run on")
parser.add_argument('-p','--pfile',type=str, required=True,help="File defining parameter ranges")
parser.add_argument('-o','--opfile',type=str,required=True,help="Output file for the data")
parser.add_argument('-b','--bfile',type=str,required=True,help="Output file for script")
args = parser.parse_args()


main(args)

'''
Mini CRACO
python py/build_build.py -n 10 -p Cubes/craco_mini_cube.json -o Cubes/craco_mini_cube.out -b build_craco_mini_cube.src
'''