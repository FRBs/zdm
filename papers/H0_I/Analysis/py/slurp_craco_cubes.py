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

F = False
if F:
    # Emax
    input_file = 'Cubes/craco_H0_F_cube.json'
    prefix = 'Cubes/craco_H0_F_cube'
    nsurveys = 1

    # Run it
    analyze_cube.slurp_cube(input_file, prefix, 
                            'Cubes/craco_H0_F_cube.npz',
                            nsurveys)

mini = False
if mini:
    # Emax
    input_file = 'Cubes/craco_mini_cube.json'
    prefix = 'Cubes/craco_mini_cube'
    nsurveys = 1

    # Run it
    analyze_cube.slurp_cube(input_file, prefix, 
                            'Cubes/craco_mini_cube.npz',
                            nsurveys)

submini = False
if submini:
    # Emax
    input_file = 'Cubes/craco_submini_cube.json'
    prefix = 'Cubes/craco_submini_cube'
    nsurveys = 1

    # Run it
    analyze_cube.slurp_cube(input_file, prefix, 
                            'Cubes/craco_submini_cube.npz',
                            nsurveys)

sfrEmax = True
if sfrEmax:
    # Emax
    input_file = 'Cubes/craco_sfr_Emax_cube.json'
    prefix = 'Cubes/craco_sfr_Emax_cube'
    nsurveys = 1

    # Run it
    analyze_cube.slurp_cube(input_file, prefix, 
                            'Cubes/craco_sfr_Emax_cube.npz',
                            nsurveys)