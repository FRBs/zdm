""" Simple script to slurp real data cubes """

from zdm import analyze_cube


mini = True
if mini:
    # Emax
    input_file = 'Cubes/real_mini_cube.json'
    prefix = 'Cubes/real_mini'
    nsurveys = 5

    # Run it
    analyze_cube.slurp_cube(input_file, prefix, 
                            'Cubes/real_mini_cube.npz',
                            nsurveys, suffix='.out')
    
super_mini = False
if super_mini:
    # Emax
    input_file = 'Cubes/real_super_mini_cube.json'
    prefix = 'Cubes/real_super_mini_cube'
    nsurveys = 5

    # Run it
    analyze_cube.slurp_cube(input_file, prefix, 
                            'Cubes/real_super_mini_cube.npz',
                            nsurveys)
