"""
File to show how to calculate joint likelihoods of PATH and zDM
"""

from zdm import optical_numerics as on
from zdm import optical as opt
from zdm import optical_params as op
from zdm import iteration as it
from zdm import cosmology as cos
from zdm import loading
from zdm import states
from zdm import figures

def main():
    """
    Demonstrates how to calculate joint FRB/Path probabilities
    
    """
    
    ######## Part 1: Initialise zDM grid ############
    # Initlisation of zDM grid
    REPS='d'
    REPS=None
    state = states.load_state("HoffmannHalo25",scat=None,rep=REPS)
    
    cos.set_cosmology(state)
    cos.init_dist_measures()
    
    # loads zDM grids
    names=['CRAFT_average_ICS']#,'CRAFT_ICS_1300','CRAFT_ICS_1632']
    ss,gs = loading.surveys_and_grids(survey_names=names,init_state=state,
                                        repeaters = False)
    s = ss[0]
    g = gs[0]
    
    # loads in Loudas model wrapper
    opstate = op.OpticalState()
    model = opt.loudas_model(opstate)
    # scales with star-formation
    model.init_args([1.])
    wrapper = opt.model_wrapper(model,g.zvals)
    
    llsum = it.get_joint_path_zdm_likelihoods(g,s,wrapper)
    print("llsum is ",llsum)
    
    
    
    
    
    
    
    
    
    
    
    
main()
