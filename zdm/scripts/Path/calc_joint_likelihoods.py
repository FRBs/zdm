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

import numpy as np
from matplotlib import pyplot as plt 

def main():
    """
    Demonstrates how to calculate joint FRB/Path probabilities
    
    """
    
    ######## Part 1: Initialise zDM grid ############
    # Initlisation of zDM grid
    REPS='d'
    #REPS=None
    state = states.load_state("HoffmannHalo25",scat=None,rep=REPS)
    
    cos.set_cosmology(state)
    cos.init_dist_measures()
    
    names=['CRAFT_average_ICS']#,'CRAFT_ICS_1300','CRAFT_ICS_1632']
    
    # scan in H0
    NH=6
    lls = np.zeros([NH])
    H0s = np.linspace(50,90,NH)
    
    for i,H0 in enumerate(H0s):
        state.cosmo.H0 = H0
        lls[i] = load_survey_calc_likelihood(state,names)
    
    plt.figure()
    plt.plot(H0s,lls)
    plt.xlabel("$H_0$ [km s$^{-1}$ Mpc$^{-1}$")
    plt.ylabel("$\\log_{10} \\mathcal{L}$")
    plt.tight_layout()
    plt.savefig("h0scan.png")
    plt.close()
    
def load_survey_calc_likelihood(state,names):
    """
    constructs grids, and calculates joint likelihoods, for survey
    names using state state
    """
    # loads zDM grids
    
    ss,gs = loading.surveys_and_grids(survey_names=names,init_state=state,
                                        repeaters = True)
    s = ss[0]
    g = gs[0]
    
    # loads in Loudas model wrapper
    opstate = op.OpticalState()
    model = opt.loudas_model(opstate)
    # scales with star-formation
    model.init_args([1.])
    wrapper = opt.model_wrapper(model,g.zvals)
    
    
    llsum = it.get_joint_path_zdm_likelihoods(g,s,wrapper,Pn=True,pwb=True,psnr=True)
    return llsum
    
    
    
    
main()
