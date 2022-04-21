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
    
    # does EVERYTHING
    run=1
    opfile="local_cube_test.dat"
    starti=0
    howmany=176
    it.cube_likelihoods(grids, surveys, vparam_dict, cube_dict,
                    run, howmany, opfile, starti=starti)
main()        
