""" Simple script to slurp """

from zdm import analyze_cube


Emax = False
if Emax:
    # Emax
    input_file = 'Cubes/craco_H0_Emax_cube.json'
    prefix = 'Cubes/craco_H0_Emax_cube'
    nsurveys = 1

    # Run it
    analyze_cube.slurp_cube(input_file, prefix, 'Cubes/craco_H0_Emax_cube.npz',
                            nsurveys)

F = True
if F:
    # Emax
    input_file = 'Cubes/craco_H0_F_cube.json'
    prefix = 'Cubes/craco_H0_F_cube'
    nsurveys = 1

    # Run it
    analyze_cube.slurp_cube(input_file, prefix, 
                            'Cubes/craco_H0_F_cube.npz',
                            nsurveys)