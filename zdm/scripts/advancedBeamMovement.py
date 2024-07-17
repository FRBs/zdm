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
from magnificationMapper import normalisedLensFuncsAcrossBeam
from astropy.io import fits
from astropy import wcs
import astropy
from astropy import units as u
from astropy import constants as const

import numpy as np
from zdm import survey
from matplotlib import pyplot as plt


def main():

    # in case you wish to switch to another output directory
    #opdir = "Localised_FRBs/"
    opdir = "CHORD/ClusterLensed/hlsp_frontier_model_macs0717_cats_v4/z0.545/"
    
    if not os.path.exists(opdir):
        os.mkdir(opdir)

    # Initialise surveys and grids

    # The below is for private, unpublished FRBs. You will NOT see this in the repository!
    sdir = "../data/Surveys/"
    
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
    state.energy.gamma = -0.948
    state.energy.alpha = 1.03
    state.FRBdemo.sfr_n = 1.15
    state.host.lsigma = 0.57
    state.host.lmean = 2.22
    state.FRBdemo.lC = 1.443
    state.energy.luminosity_function=4
    state.FRBdemo.alpha_method=0

    clusterRedshift = 0.545

    cluster=True
    lensing =True
    
    #relBeamPositions = np.load('relBeamPos.npy') #relative to magni
    relBeamPositions = np.array([[0,0]])
    ratesArr=np.zeros(len(relBeamPositions[:,0]))
    
    
    for i in range(len(relBeamPositions[:,0])):
        print('---Beam Pos:', i)
        formatted_number = "{:02d}".format(i)
        surveyName = 'CHORD_BeamPos_'+str(formatted_number)
        bPosNum = "{:02d}".format(i)
        s,g = loading.survey_and_grid(survey_name=surveyName, opdir=opdir, bPosNum=bPosNum,
            NFRB=None,sdir=sdir,init_state=state, cluster=cluster, 
            clusterRedshift=clusterRedshift, lensing=lensing)
    
        np.save(opdir+'ratesUnlensed_BP_'+str(formatted_number), g.rates)
        
        
        FRB_rate_per_day = np.sum(g.rates) * 10**g.state.FRBdemo.lC
        print("Rate of FRBs per day is ",FRB_rate_per_day)
        FRB_rate_per_day = np.sum(g.rates[g.zvals>1.0,:]) * 10**g.state.FRBdemo.lC
        print("Rate of FRBs per day with z > 1.0 is ",FRB_rate_per_day)
     
        ratesArr[i] = FRB_rate_per_day
        np.save(opdir+surveyName+'RatesArr', ratesArr)

        misc_functions.plot_grid_2(
                g.rates,
                g.zvals,
                g.dmvals,
                name=opdir + surveyName + ".pdf",
                norm=3,
                log=True,
                label="$\\log_{10} p({\\rm DM}_{\\rm EG},z)$  [a.u.]",
                project=False,
                zmax=4,
                Aconts=[0.01, 0.1, 0.5],
                DMmax=4000
            ) #
    

main()
