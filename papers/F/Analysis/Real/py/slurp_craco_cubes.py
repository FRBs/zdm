""" Simple script to slurp """

from zdm import analyze_cube


def main(pargs):

    if pargs.run == "Emax":
        # Emax
        input_file = "Cubes/craco_H0_Emax_cube.json"
        prefix = "Cubes/tmp"
        nsurveys = 1

        # Run it
        analyze_cube.slurp_cube(
            input_file, prefix, "Cubes/craco_H0_Emax_cube.npz", nsurveys
        )

    elif pargs.run == "F":
        # Emax
        input_file = "Cubes/craco_H0_F_cube.json"
        prefix = "Cloud/Output/craco_H0_F"
        nsurveys = 1

        # Run it
        analyze_cube.slurp_cube(
            input_file, prefix, "Cubes/craco_H0_F_cube.npz", nsurveys
        )

    elif pargs.run == "logF":
        # Emax
        input_file = "Cubes/craco_H0_logF_cube.json"
        prefix = "Cloud/Output_logF_test/craco_H0_logF"
        nsurveys = 1

        # Run it
        analyze_cube.slurp_cube(
            input_file, prefix, "Cubes/craco_H0_logF_cube.npz", nsurveys
        )

    elif pargs.run == "logF_full":
        # Emax
        input_file = "Cubes/craco_full_cube.json"
        prefix = "Cloud/OutputFull/craco_full"
        nsurveys = 1

        # Run it
        analyze_cube.slurp_cube(
            input_file, prefix, "Cubes/craco_full_cube.npz", nsurveys
        )

    elif pargs.run == "lmF":
        # Emax
        input_file = "Cubes/craco_lm_F_cube.json"
        prefix = "Cloud/Output/craco_lm_F"
        nsurveys = 1

        # Run it
        analyze_cube.slurp_cube(
            input_file, prefix, "Cubes/craco_lm_F_cube.npz", nsurveys
        )

    elif pargs.run == "mini":
        # Emax
        input_file = "Cubes/craco_mini_cube.json"
        # prefix = 'Cubes/craco_mini'
        prefix = "Cloud/OutputMini/craco_mini"
        nsurveys = 1

        # Run it
        analyze_cube.slurp_cube(
            input_file, prefix, "Cubes/craco_mini_cube.npz", nsurveys
        )
    elif pargs.run == "submini":
        # Emax
        input_file = "Cubes/craco_submini_cube.json"
        prefix = "Cubes/craco_submini_cube"
        nsurveys = 1

        # Run it
        analyze_cube.slurp_cube(
            input_file, prefix, "Cubes/craco_submini_cube.npz", nsurveys
        )

    elif pargs.run == "sfrEmax":
        # Emax
        input_file = "Cubes/craco_sfr_Emax_cube.json"
        prefix = "Cubes/craco_sfr_Emax_cube"
        nsurveys = 1

        # Run it
        analyze_cube.slurp_cube(
            input_file, prefix, "Cubes/craco_sfr_Emax_cube.npz", nsurveys
        )

    elif pargs.run == "alphaEmax":
        # Emax
        input_file = "Cubes/craco_alpha_Emax_cube.json"
        prefix = "Cubes/craco_alpha_Emax_cube"
        nsurveys = 1

        # Run it
        analyze_cube.slurp_cube(
            input_file, prefix, "Cubes/craco_alpha_Emax_cube.npz", nsurveys
        )
    elif pargs.run == "full":
        # Emax
        input_file = "Cubes/craco_full_cube.json"
        prefix = "Cubes/craco_full"
        nsurveys = 1

        # Run it
        analyze_cube.slurp_cube(
            input_file, prefix, "Cubes/craco_full_cube.npz", nsurveys
        )
    elif pargs.run == "another_full":
        # Emax
        input_file = "Cubes/craco_full_cube.json"
        prefix = "Cubes/craco_400_full"
        nsurveys = 1

        # Run it
        analyze_cube.slurp_cube(
            input_file, prefix, "Cubes/craco_400_full_cube.npz", nsurveys
        )
    elif pargs.run == "third_full":
        # Emax
        input_file = "Cubes/craco_full_cube.json"
        prefix = "Cubes/craco_3rd_full"
        nsurveys = 1

        # Run it
        analyze_cube.slurp_cube(
            input_file, prefix, "Cubes/craco_3rd_full_cube.npz", nsurveys
        )
    elif pargs.run == "real":
        # Emax
        input_file = "Cubes/craco_real_cube.json"
        prefix = "Cloud/Output/craco_real"
        nsurveys = 1

        # Run it
        analyze_cube.slurp_cube(
            input_file, prefix, "Cubes/craco_real_cube.npz", nsurveys
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

#  python py/slurp_craco_cubes.py mini
#  python py/slurp_craco_cubes.py another_full

#  python py/slurp_craco_cubes.py F
