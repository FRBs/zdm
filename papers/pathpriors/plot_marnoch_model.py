"""
This file compares predictions and outputs for the model of Marnoch et al
"""


#standard Python imports
import os
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import minimize

# imports from the "FRB" series
from zdm import optical as opt
from zdm import optical_params as op
from zdm import loading
from zdm import cosmology as cos
from zdm import parameters
from zdm import loading
from zdm import optical_numerics as on
from zdm import states

# other FRB library imports
import astropath.priors as pathpriors

import matplotlib

defaultsize=14
ds=4
font = {'family' : 'Helvetica',
        'weight' : 'normal',
        'size'   : defaultsize}
matplotlib.rc('font', **font)


def main():
    """
    Main function
    Contains outer loop to iterate over parameters
    
    """
    
    ######### List of all ICS FRBs for which we can run PATH #######
    # hard-coded list of FRBs with PATH data in ice paper
    frblist=opt.frblist
    frblist.remove('FRB20230731A') # too reddened
    
    # Initlisation of zDM grid
    # Eventually, this should be part of the loop, i.e. host IDs should
    # be re-fed into FRB surveys. However, it will be difficult to do this
    # with very limited redshift estimates. That might require posterior
    # estimates of redshift given the observed galaxies. Maybe.
    state = states.load_state("HoffmannHalo25",scat=None,rep=None)
    #state = parameters.State()
    cos.set_cosmology(state)
    cos.init_dist_measures()
    names=['CRAFT_ICS_892','CRAFT_ICS_1300','CRAFT_ICS_1632']
    ss,gs = loading.surveys_and_grids(survey_names=names,init_state=state)
    
    
    ######## Determnine which statistic to use in optimisation ########
    # setting istat=0 means using a ks statistic to fit p(m_r)
    # setting istat=1 means using a maximum likelihood estimator
    istat=1
    # determines which model to use
    modelname = "marnoch"
    
    opdir = modelname+"_output/"
    POxcut = None # set to e.g. 0.9 to reject FRBs with lower posteriors when doing model comparisons
    
    if not os.path.exists(opdir):
        os.mkdir(opdir)
    
    model = opt.marnoch_model()
    
    outfile = opdir+"marnoch_best_fit_apparent_magnitudes.png"
    wrappers = on.make_wrappers(model,gs)
    NFRB,AppMags,AppMagPriors,ObsMags,ObsPriors,ObsPosteriors,PUprior,PUobs,sumPUprior,sumPUobs,frbs,dms = on.calc_path_priors(frblist,ss,gs,wrappers,verbose=False)
    llstat = on.calculate_likelihood_statistic(NFRB,AppMags,AppMagPriors,ObsMags,ObsPosteriors,PUobs,PUprior,plotfile=outfile)
    
    ksstat = on.calculate_ks_statistic(NFRB,AppMags,AppMagPriors,ObsMags,ObsPosteriors,sumPUobs,
                                    sumPUprior,plotfile=outfile,abc="(a)",tag="Marnoch23: ")
    
    print("Best-fit stats of the Marnoch model are ll=",llstat," ks = ",ksstat)
    


main()
