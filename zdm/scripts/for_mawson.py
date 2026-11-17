""" 
This script creates zdm grids and plots localised FRBs

It can also generate a summed histogram from all CRAFT data

"""
import os

from zdm import cosmology as cos
from zdm import misc_functions
from zdm import parameters
from zdm import survey
from zdm import pcosmic
from zdm import iteration as it
from zdm.craco import loading
from zdm import io

import numpy as np
from zdm import survey
from matplotlib import pyplot as plt


def main():

    # in case you wish to switch to another output directory
    #opdir = "Localised_FRBs/"
    opdir = "CHORD/"
    
    if not os.path.exists(opdir):
        os.mkdir(opdir)

    # Initialise surveys and grids

    # The below is for private, unpublished FRBs. You will NOT see this in the repository!
    sdir = "../data/Surveys/"
    name = 'CHORD'
    
    # specifies state, updates variables according to H0
    # best fit, but with Emax extended as per Ryder et al
#    state = parameters.State()
#    state.energy.lEmax = 41.63
#    state.energy.gamma = -0.948
#    state.energy.alpha = -1.03
#    state.FRBdemo.sfr_n = 1.15
#    state.host.lsigma = 0.57
#    state.host.lmean = 2.22
#    state.FRBdemo.lC = 1.443
    state = parameters.State()
    state.energy.lEmax = 41.38
    state.energy.gamma = -1.3
    state.energy.alpha = -1.39
    state.FRBdemo.sfr_n = 1.0
    state.host.lsigma = 0.57
    state.host.lmean = 2.22
    state.FRBdemo.lC = 4.86
    state.energy.luminosity_function=0
    
    
    s,g = loading.survey_and_grid(survey_name=name,
        NFRB=None,sdir=sdir,init_state=state)

    np.save('zvals', g.zvals)
    np.save('dmvals', g.dmvals)

    np.save('ratesUnlensed', g.rates)
    
    FRB_rate_per_day = np.sum(g.rates) * 10**g.state.FRBdemo.lC
    print("Rate of FRBs per day is ",FRB_rate_per_day)
    FRB_rate_per_day = np.sum(g.rates[g.zvals>1.0,:]) * 10**g.state.FRBdemo.lC
    print("Rate of FRBs per day with z > 1.0 is ",FRB_rate_per_day)
 
    misc_functions.plot_grid_2(
            g.rates,
            g.zvals,
            g.dmvals,
            name=opdir + name + ".pdf",
            norm=3,
            log=True,
            label="$\\log_{10} p({\\rm DM}_{\\rm EG},z)$  [a.u.]",
            project=False,
            zmax=4,
            Aconts=[0.01, 0.1, 0.5],
            DMmax=4000
        ) #


main()
