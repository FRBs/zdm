""" Run a Nautilus test """

# It should be possible to remove all the matplotlib calls from this
# but in the current implementation it is not removed.
import argparse
import numpy as np
import os, sys

from concurrent.futures import ProcessPoolExecutor
import subprocess

from zdm import iteration as it
from zdm import io

from IPython import embed

# Local
sys.path.append(os.path.abspath("../Analysis/py"))

def main(pargs, pfile:str, oproot:str, NFRB:int=None, iFRB:int=0,
         outdir:str='Output'):

    # Generate the folder?
    if not os.path.isdir(outdir):
        os.mkdir(outdir)
    
    ############## Load up ##############
    input_dict=io.process_jfile(pfile)

    # Deconstruct the input_dict
    state_dict, cube_dict, vparam_dict = it.parse_input_dict(input_dict)

    npoints = np.array([item['n'] for key, item in vparam_dict.items()])
    ntotal = int(np.prod(np.abs(npoints)))

    nper_cpu = ntotal // pargs.ncpu
    if int(ntotal/pargs.ncpu) != nper_cpu:
        raise IOError(f"Ncpu={pargs.ncpu} must divide evenly into ntotal={ntotal}")

    commands = []
    for kk in range(pargs.ncpu):
        line = []
        outfile = os.path.join(outdir, oproot.replace('.out', f'{kk+1}.out'))
        line = ['zdm_build_cube', '-n', f'{kk+1}',
                '-m', f'{nper_cpu}', '-o', f'{outfile}',
                '-s', f'CRACO_alpha1_Planck18_Gamma', '--clobber',
                '-p', f'{pfile}']
        # NFRB?
        if NFRB is not None:
            line += [f'--NFRB', f'{NFRB}']
        # iFRB?
        if iFRB > 0:
            line += [f'--iFRB', f'{iFRB}']
        # Finish
        #line += ' & \n'
        commands.append(line)

    # Launch em!
    processes = []
    for command in commands:
        # Popen
        pw = subprocess.Popen(command)
        processes.append(pw)

    # Wait on em!
    for pw in processes:
        pw.wait()

    print("All done!")

def parse_option():
    # test for command-line arguments here
    parser = argparse.ArgumentParser()
    parser.add_argument('-n','--ncpu',type=int, required=True,help="Number of CPUs to run on")
    #parser.add_argument('--NFRB',type=int,required=False,help="Number of FRBs to analzye")
    #parser.add_argument('--iFRB',type=int,default=0,help="Initial FRB to run from")
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    # get the argument of training.
    pfile = '../Cubes/craco_alpha_Emax_cube.json'
    oproot = 'craco_nautilus_test.out' 
    pargs = parse_option()
    main(pargs, pfile, oproot, NFRB=100, iFRB=100)

'''
alpha vs Emax
python py/build_build.py -n 10 -p Cubes/craco_alpha_Emax_cube.json -o Cubes/craco_alpha_Emax_cube.out -b build_craco_alpha_Emax_cube.src --NFRB 100 --iFRB 100

sfr vs Emax
python py/build_build.py -n 10 -p Cubes/craco_sfr_Emax_cube.json -o Cubes/craco_sfr_Emax_cube.out -b build_craco_sfr_Emax_cube.src --NFRB 100 --iFRB 100

SubMini CRACO
python py/build_build.py -n 10 -p Cubes/craco_submini_cube.json -o Cubes/craco_submini_cube.out -b build_craco_submini_cube.src --NFRB 100 --iFRB 100

Mini CRACO
python py/build_build.py -n 15 -p Cubes/craco_mini_cube.json -o Cubes/craco_mini_cube.out -b build_craco_mini_cube.src --NFRB 100 --iFRB 100

Nautilus test
python py/build_build.py -n 5 -p Cubes/craco_alpha_Emax_cube.json -o TstCubes/craco_nautilus_test.out -b build_nautilus_test.src --NFRB 100 --iFRB 100
'''