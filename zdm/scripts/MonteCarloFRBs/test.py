import os

from astropy.cosmology import Planck18
#from zdm import cosmology as cos
from zdm import misc_functions
from zdm import parameters
#from zdm import survey
#from zdm import pcosmic
#from zdm import iteration as it
from zdm import loading
#from zdm import io
import matplotlib.pyplot as plt

import numpy as np
#from matplotlib import pyplot as plt
from pkg_resources import resource_filename
#import time

from frb.dm import igm
#params 
NMC = 10 
nu = 1. # GHz
t_samp = 1.182  # ms
bandwidth = 1.0 # MHz
snr_thresh = 4. # SNR threshold
mean_DM_MW = 80. # pc cm^-3
disp_DM_MW = 50. # pc cm^-3



def creategrid():

    # in case you wish to switch to another output directory
    #opdir = "../ScatSim/"
    #if not os.path.exists(opdir):
    #    os.mkdir(opdir)
    
    # Initialise surveys and grids
    #sdir = os.path.join(resource_filename('zdm', 'data'), 'Surveys')
    sdir = '/Users/cjames/CRAFT/Git/zdm/zdm/data/Surveys'
    
    names = ["CRAFT_ICS_1300"]
    
    # essentially turns off DM host and sets all FRB widths to ~0 (or close enough)
    param_dict = {'lmean': 0.01, 'lsigma': 0.4, 'Wlogmean': -1,'WNbins': 1,
        'Wlogsigma': 0.1, 'Slogmean': -2,'Slogsigma': 0.1}
    state = parameters.State()
    state.set_astropy_cosmo(Planck18)
    state.update_params(param_dict)
    
    
    surveys, grids = loading.surveys_and_grids(survey_names = names,
        repeaters=False, sdir=sdir)
    
    # plots it
    #misc_functions.plot_grid_2(g.rates,g.zvals,g.dmvals,
    #        name=os.path.join(opdir,'tau_dm_grid.png'),norm=3,log=True,
    #        label='$\\log_{10} p({\\rm DM}_{\\rm EG}|z)$ [a.u.]',
    #        project=False,
    #        zmax=2.5,DMmax=3000,cmap="Oranges",Aconts=[0.01,0.1,0.5])#,
    #            pdmgz=[0.05,0.5,0.95])

    return grids[0]
    
    


# get the grid
g = creategrid()





# create DM_EG by sampling the grid
frbs = g.GenMCSample(NMC)
frbs = np.array(frbs)
zs = frbs[:,0]
DMcos = frbs[:,1]
snrs = frbs[:,3]
#bs = frbs[:,2]
#ws = frbs[:,4]

print (snrs)
