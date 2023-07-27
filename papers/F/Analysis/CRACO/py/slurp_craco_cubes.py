"""
Script to intake the `.csv` files from the CRACO runs and convert them to a single `.npz` file.

The only argument in running the file corresponds to a hard-coded location of the `.csv` files and the cube `.json` file. 
"""

from zdm import analyze_cube


def main(pargs):

    if pargs.run == "logF":
        # 2D cube run with H0 and logF
        input_file = "Cubes/craco_H0_logF_cube.json"
        prefix = "Cloud/Output_logF_test/craco_H0_logF"
        nsurveys = 1

        # Run it
        analyze_cube.slurp_cube(
            input_file, prefix, "Cubes/craco_H0_logF_cube.npz", nsurveys
        )

    elif pargs.run == "logF_full":
        # Full CRACO likelihood cube
        input_file = "Cubes/craco_full_cube.json"
        prefix = "Cloud/OutputFull/craco_full"
        nsurveys = 1

        # Run it
        analyze_cube.slurp_cube(
            input_file, prefix, "Cubes/craco_full_cube.npz", nsurveys
        )

def parse_option():
    """
    This is a function used to parse the arguments in the training.
    
    Returns:
        args: (dict) dictionary of the arguments.
    """
    import argparse

    parser = argparse.ArgumentParser("Slurping the cubes")
    parser.add_argument("run", type=str, help="Run to slurp")
    # parser.add_argument('--debug', default=False, action='store_true',
    #                    help='Debug?')
    args = parser.parse_args()

    return args


# Command line execution
if __name__ == "__main__":

    pargs = parse_option()
    main(pargs)

#  python py/slurp_craco_cubes.py logF_full
