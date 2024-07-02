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
    opdir = "CHORD/"
    
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
    state.FRBdemo.alpha_method=1

    clusterDMFile = 'Thermo_MACSJ0717_N.fits'
    #clusterDMFile = 'allOnes.fits'
    clusterRedshift = 0.545

    magni = fits.getdata('hlsp_frontier_model_macs0717_bradac_v1_z01-magnifMAX100.fits')
    info = fits.getheader('hlsp_frontier_model_macs0717_bradac_v1_z01-magnifMAX100.fits')
    proj = wcs.WCS(info)
    xMagni = np.meshgrid(np.arange(0,len(magni[:,0]),1), np.arange(0,len(magni[0,:]),1))
    tempCoords = proj.array_index_to_world_values(xMagni[0], xMagni[1])

    dms = fits.getdata(clusterDMFile)
    infoDM = fits.getheader(clusterDMFile)
    projDM = wcs.WCS(infoDM)
    cluster=True
    lensing =True
    
    #relBeamPositions = np.load('relBeamPos.npy') #relative to magni
    relBeamPositions = np.array([[0,0]])
    ratesArr=np.zeros(len(relBeamPositions[:,0]))
    
    
    for i in range(len(relBeamPositions[:,0])):
        print('---Beam Pos:', i)
        formatted_number = "{:02d}".format(i)
        surveyName = 'CHORD_BeamPos_'+str(formatted_number)
        mux, pmux, wideMagni, wideX = normalisedLensFuncsAcrossBeam(48*u.m, 900*u.MHz, 1e-3, 100, np.array([np.mean(tempCoords[0])+relBeamPositions[i,0], np.mean(tempCoords[1])+relBeamPositions[i,1]]), proj, magni, 'CHORD/'+surveyName)
        rawWeights = 1/wideMagni*(1/wideMagni)**(state.energy.gamma)
        np.save('mux_BP_'+str(formatted_number), np.log10(mux))
        np.save('pmux_BP_'+str(formatted_number), pmux)
        np.save(str(surveyName)+'mus', np.log10(mux))
        np.save(str(surveyName)+'pmus', pmux)
        del(pmux)
        del(mux)


        s,g = loading.survey_and_grid(survey_name=surveyName,
            NFRB=None,sdir=sdir,init_state=state, cluster=cluster, 
            clusterNeFile=clusterDMFile, clusterRedshift=clusterRedshift, 
            bPos=np.array([np.mean(tempCoords[0])+relBeamPositions[i,0], np.mean(tempCoords[1])+relBeamPositions[i,1]]), 
            lensing=lensing, rawWeights=rawWeights, weightsProj=proj, xWeights=wideX)
    
        np.save('ratesUnlensed_BP_'+str(formatted_number), g.rates)
        
        
        FRB_rate_per_day = np.sum(g.rates) * 10**g.state.FRBdemo.lC
        print("Rate of FRBs per day is ",FRB_rate_per_day)
        FRB_rate_per_day = np.sum(g.rates[g.zvals>1.0,:]) * 10**g.state.FRBdemo.lC
        print("Rate of FRBs per day with z > 1.0 is ",FRB_rate_per_day)
     
        ratesArr[i] = FRB_rate_per_day
        np.save('CHORD/'+surveyName+'RatesArr', ratesArr)

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
