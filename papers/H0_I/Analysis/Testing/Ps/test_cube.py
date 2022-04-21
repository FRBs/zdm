"""
This file is intended to test the computing performance
when updating grids for a cube evaluation.

It tackles the following questions:
- How long does it take to update a grid for parameter X?
  (how much longer for additional grids beyond the first?)
- What is the optimal parameter order for cubing?
  (is this changed by the number of grids used in the cube?)
- Is the grid update method accurate?
  (and is it still accurate when using the many-cube shortcut?)

"""

from zdm import real_loading
from zdm import iteration as it
from zdm import io

from IPython import embed

# this based off the file 
def main(likelihoods=True,detail=0,verbose=True):
    
    ############## Load up ##############
    pfile = 'gamma_hnot_slice.json'
    input_dict=io.process_jfile(pfile)

    # Deconstruct the input_dict
    state_dict, cube_dict, vparam_dict = it.parse_input_dict(input_dict)
    
    
    # State
    state = real_loading.set_state()
    state.update_param_dict(state_dict)
    
    ############## Initialise ##############
    surveys, grids = real_loading.surveys_and_grids(init_state=state)
    embed(header='38 of test_cube')

    # Exploring
    # grids[-1]
    # These are identical!
    #
    # DIFFERENT
    # In [7]: grids[-1].rates[0][0]
    # Out[7]: 1.0728471869760847e-12
    # In [10]: grids[-1].pdv[0][0]
    # Out[10]: 3.2549376652150038e-06
    # Dialing back further
    # IDENTICAL
    # In [13]: grids[-1].dV[0]
    # Out[13]: 85602.6085381501
    # In [15]: grids[-1].smear[0][0]
    # Out[15]: 0.0007586531824186714
    # In [17]: grids[-1].smear_grid[0][0]
    # Out[17]: 3.3853247269297123e-07
    # In [19]: grids[-1].thresholds[0,0,0]
    # Out[19]: 4.0384026613336495e+35
    # After fixing the randomness in weights
    # In [1]: self.b_fractions[0,0,0]
    # Out[1]: 2.223242035802155e-16





    
    # does EVERYTHING
    run=1
    opfile="local_cube_test.dat"
    starti=0
    howmany=176
    it.cube_likelihoods(grids, surveys, vparam_dict, cube_dict,
                    run, howmany, opfile, starti=starti)
main()        
