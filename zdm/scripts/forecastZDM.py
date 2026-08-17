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
from zdm import loading
from zdm import io
from magnificationMapper import normalisedLensFuncsAcrossBeam
from astropy.io import fits
from astropy import wcs
import astropy
from astropy import units as u
from astropy import constants as const

import numpy as np
from zdm import survey, figures
from matplotlib import pyplot as plt


def pullTrigger(clusterRedshift, name, gamma, n):

    formatted_cluster_redshift = "{:03.2f}".format(clusterRedshift)
    formatted_energy_index = "{:03.2f}".format(gamma)
    opdir = "/arc/projects/chime_frb/msammons/clusterLensing/testZDM/"+name+'/z'+formatted_cluster_redshift+'/'+formatted_energy_index+'/'
    print(opdir)
    #opdir = "/arc/projects/chime_frb/msammons/CHIME/ClusterLensed/"+name+'/z'+formatted_cluster_redshift+'/'

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
    state.energy.lEmax = 41.63
    state.energy.lEmin = 30.0
    state.energy.gamma = gamma
    state.energy.alpha = 1.03
    state.FRBdemo.sfr_n = n
    state.host.lsigma = 0.57
    state.host.lmean = 2.22
    state.FRBdemo.lC = 2.3-9
    state.energy.luminosity_function=4
    state.FRBdemo.alpha_method=1
    #state.FRBdemo.source_evolution=0
#    state = parameters.State()
#    state.energy.lEmax = 41.42
#    state.energy.gamma = -1.16
#    state.energy.alpha = 0.92
#    state.FRBdemo.sfr_n = 0.91
#    state.host.lsigma = 0.46
#    state.host.lmean = 2.02
#    state.FRBdemo.lC = 2.0
#    state.energy.luminosity_function=2
#    state.FRBdemo.alpha_method=1


    cluster=True
    lensing =True
    
    #relBeamPositions = np.load('relBeamPos.npy') #relative to magni
    relBeamPositions = np.array([[0,0]])
    ratesArr=np.zeros([len(relBeamPositions[:,0]), 2])
    
    
    for i in range(len(relBeamPositions[:,0])):
        print('---Beam Pos:', i)
        formatted_number = "{:02d}".format(i)
        surveyName = 'CHIME_SynthBeam'
        bPosNum = "{:02d}".format(i)
        s,gSet = loading.surveys_and_grids(survey_names=[surveyName], opdir=opdir, bPosNum=bPosNum,
            NFRB=None,sdir=sdir,init_state=state, cluster=cluster, 
            clusterRedshift=clusterRedshift, lensing=lensing)
    
        print('gSet lengths:', len(gSet))
        g = gSet[0]
        np.save(opdir+'rates_BP_'+str(formatted_number)+'SourceFunc1thresh5NEWZDMSFRn'+str("{:.2f}".format(n)), g.rates*10**g.state.FRBdemo.lC)
        
        
        FRB_rate_per_day = np.sum(g.rates) * 10**g.state.FRBdemo.lC
        print("Rate of FRBs per day is ",FRB_rate_per_day)
        ratesArr[i,0] = FRB_rate_per_day
        FRB_rate_per_day = np.sum(g.rates[g.zvals>1.0,:]) * 10**g.state.FRBdemo.lC
        print("Rate of FRBs per day with z > 1.0 is ",FRB_rate_per_day)
     
        ratesArr[i,1] = FRB_rate_per_day
        np.save(opdir+surveyName+'RatesArrSourceFunc1thresh5NEWZDMSFRn'+str("{:.2f}".format(n)), ratesArr)

        print('SAVING TO', opdir + surveyName + 'SourceFunc1thresh5NEWZDMSFRn'+str("{:.2f}".format(n)))
        figures.plot_grid(
                g.rates,
                g.zvals,
                g.dmvals,
                name=opdir + surveyName + 'SourceFunc1thresh5NEWZDMSFRn'+str("{:.2f}".format(n))+'.pdf',
                norm=3,
                log=True,
                label="$\\log_{10} p({\\rm DM}_{\\rm EG},z)$  [a.u.]",
                project=False,
                zmax=4,
                Aconts=[0.01, 0.1, 0.5],
                DMmax=4000
            ) #
     

