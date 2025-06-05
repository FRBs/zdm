"""
This script creates a 3D distribution. For each values
of width w, it determines the relative contribution of 
each scattering and width value to that w.

This then allows FRB w and tau values to be fit directly,
rather than indirectly via a total effective width.

This is not a 5D problem (z,DM,w,scat,tau) because
p(w|z,DM) is independent of the tau and w that
contributed to it.

Nonetheless, p(tau,w) is z-dependent, hence we need a p

"""

import os

from zdm import cosmology as cos
from zdm import figures
from zdm import parameters
from zdm import survey
from zdm import pcosmic
from zdm import iteration as it
from zdm import loading
from zdm import io
from pkg_resources import resource_filename
import numpy as np
from matplotlib import pyplot as plt

import matplotlib

defaultsize=14
ds=4
font = {'family' : 'Helvetica',
        'weight' : 'normal',
        'size'   : defaultsize}
matplotlib.rc('font', **font)


def main():
    """
    
    """
    
    # in case you wish to switch to another output directory
    opdir = "Plots/"
    if not os.path.exists(opdir):
        os.mkdir(opdir)
    
    # directory where the survey files are located. The below is the default - 
    # you can leave this out, or change it for a different survey file location.
    sdir = os.path.join(resource_filename('zdm', 'data'), 'Surveys')
    
    survey_name = "CRAFT_average_ICS"
    
    # make this into a list to initialise multiple surveys art once
    names = ["CRAFT_ICS_892","CRAFT_ICS_1300","CRAFT_ICS_1632"]
    
    repeaters=False
    # sets plotting limits
    zmax = 2.
    dmmax = 2000
    
    survey_dict = {"WMETHOD": 3}
    state_dict = {}
    state_dict["scat"] = {}
    state_dict["scat"]["Sbackproject"] = True # turns on backprojection of tau and width for our model
    state_dict["scat"]["Sbackproject"] = True
    state_dict["width"] = {}
    state_dict["width"]["WNInternalBins"] = 100 # sets it to a small quantity
    
    surveys, grids = loading.surveys_and_grids(survey_names = names,\
                        repeaters=repeaters, sdir=sdir,nz=70,ndm=140,
                        survey_dict = survey_dict, state_dict = state_dict)

    # gets log-likelihoods including tau,w
    s=surveys[0]
    g=grids[0]
    ll1 = it.calc_likelihoods_1D(g,s,Pn=True,pNreps=True,ptauw=True,dolist=0)
    ll2 = it.calc_likelihoods_2D(g,s,Pn=True,pNreps=True,psnr=True,ptauw=True,dolist=0)
    print(ll1,ll2)
main()
