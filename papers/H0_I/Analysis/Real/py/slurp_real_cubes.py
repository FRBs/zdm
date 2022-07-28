""" Simple script to slurp real data cubes """

from zdm import analyze_cube


mini = False
if mini:
    # Emax
    input_file = 'Cubes/real_mini_cube.json'
    prefix = 'Cubes/real_mini'
    nsurveys = 5

    # Run it
    analyze_cube.slurp_cube(input_file, prefix, 
                            'Cubes/real_mini_cube.npz',
                            nsurveys)
    
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

full = False
if full:
    # Emax
    input_file = 'Cubes/real_full_cube.json'
    prefix = 'Cubes/real_full'
    nsurveys = 5

    # Run it
    analyze_cube.slurp_cube(input_file, prefix, 
                            'Cubes/real_full_cube.npz',
                            nsurveys)
    
down_full = True
if down_full:
    # Emax
    input_file = 'Cubes/downgraded_real_full_cube.json'
    prefix = 'Cubes/real_down_full'
    nsurveys = 5

    # Run it
    analyze_cube.slurp_cube(input_file, prefix, 
                            'Cubes/real_down_full_cube.npz',
                            nsurveys)
    