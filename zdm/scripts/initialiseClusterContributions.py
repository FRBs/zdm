""" 
This script creates zdm grids and plots localised FRBs

It can also generate a summed histogram from all CRAFT data

"""
import os
from magnificationMapper import normalisedLensFuncsAcrossBeam, clusterDMFuncAcrossBeam, mapRescaler
from astropy.io import fits
from astropy import wcs
import astropy
from astropy import units as u
from astropy import constants as const
from astropy.cosmology import Planck18 as cosmo

import numpy as np
from matplotlib import pyplot as plt


def runner(clusterRedshift, name, clusterNeFile, trueClusterRedshift, sourceMagniArr=False):

    # in case you wish to switch to another output directory
    #opdir = "Localised_FRBs/"
    formatted_cluster_redshift = "{:03.2f}".format(clusterRedshift)
    opdir = "/arc/projects/chime_frb/msammons/CHIME/ClusterLensed/"+name+'/z'+formatted_cluster_redshift+'/'
    print(opdir)
    
    if not os.path.exists(opdir):
        os.makedirs(opdir)

    dishDiam = 80*u.m
    fbar = 600*u.MHz
    bThresh = 1e-3
    bbins = 100
    energyIndex = -0.948

    fileList = [name+'_kappa.fits', name+'_gamma.fits', clusterNeFile]
    mapRescaler(opdir, fileList, trueClusterRedshift, clusterRedshift)

    kappa = fits.getdata(opdir+fileList[0])
    gamma = fits.getdata(opdir+fileList[1])
    info = fits.getheader(opdir+fileList[0])
    proj = wcs.WCS(info)
    xMagni = np.meshgrid(np.arange(0,len(kappa[:,0]),1), np.arange(0,len(kappa[0,:]),1))
    tempCoords = proj.array_index_to_world_values(xMagni[0], xMagni[1])

    infoNe = fits.getheader(opdir+fileList[2])
    projNe = wcs.WCS(infoNe)
    ne = fits.getdata(opdir+fileList[2])

    cluster=True
    lensing =True
    
    #relBeamPositions = np.load('relBeamPos.npy') #relative to magni
    relBeamPositions = np.array([[0,0]])
    ratesArr=np.zeros(len(relBeamPositions[:,0]))
    zvals = np.load('zvals.npy')
    mux = 10**(np.arange(-3,2,0.02)+0.05)
    np.save(opdir+'mux', np.log10(mux))
    scatThresh = 10**np.arange(-4,3,0.02)
    xProbScat = scatThresh[:-1]*10**(np.diff(np.log10(scatThresh))[0]/2)
    np.save(opdir+'xProbScat', (xProbScat))
    DMThresh = np.arange(0,15000,200)
    np.save(opdir+'DMThresh', DMThresh)

    
    for i in range(len(relBeamPositions[:,0])):
        bPos = np.array([np.mean(tempCoords[0])+relBeamPositions[i,0], np.mean(tempCoords[1])+relBeamPositions[i,1]])
        for j in range(len(zvals)):
            print('---Beam Pos:', i, '---Redshift:', j)
            formatted_number = "{:02d}".format(i)
            formatted_redshift = "{:03.2f}".format(zvals[j])
            surveyName = 'CHIME_BeamPos_'+str(formatted_number)
            
            if zvals[j] > clusterRedshift:
                if sourceMagniArr:
                    pixCoordsArr= np.load(opdir+("{:03.2f}".format(zvals[j]))+'_sourceCoordsArr.npy') 
                    magni = np.load(opdir+("{:03.2f}".format(zvals[j]))+'_sourceCoordsArr.npy') 

                else:
                    rescaleFactor = (cosmo.angular_diameter_distance_z1z2(clusterRedshift, zvals[j])*cosmo.angular_diameter_distance(trueClusterRedshift)/cosmo.angular_diameter_distance(zvals[j])/cosmo.angular_diameter_distance(clusterRedshift)).value
                    pixCoordsArr = xMagni
                    magni = 1/np.abs((1-kappa*rescaleFactor)**2-(gamma*rescaleFactor)**2)
                    magni[magni>=100] = 100
                

                pmux, magni, xMagni = normalisedLensFuncsAcrossBeam(dishDiam, fbar, bThresh, bbins, bPos, proj, xMagni, magni, pixCoordsArr, opdir+surveyName, sourceMagniArr=sourceMagniArr, muThresh = mux)

                np.save(opdir+'pmux_BP_'+str(formatted_number)+str(formatted_redshift), pmux)
                del(pmux)

                if sourceMagniArr:
                    rawWeights = (1/magni)**(energyIndex)
                else:
                    rawWeights = 1/magni*(1/magni)**(energyIndex)
    
                tempCoords = proj.array_index_to_world_values(xMagni[0], xMagni[1])

                probScat, fractionUnscattered, pdms = clusterDMFuncAcrossBeam(
                    D = dishDiam,
                    freq = fbar,
                    thresh = bThresh,
                    nbins = bbins,
                    bPos = bPos, 
                    proj = projNe, 
                    clusterRedshift = clusterRedshift,
                    z = zvals[j],
                    ne = ne,
                    name = opdir+'DM_BP_'+str(formatted_number),
                    weights = rawWeights,
                    imageProj = proj,
                    imageCoords = tempCoords,
                    DMThresh = DMThresh,
                    scatThresh = scatThresh
                )   
                np.save(opdir+'probScat_BP_'+str(formatted_number)+str(formatted_redshift), probScat) 
                np.save(opdir+'fractionUnscattered_BP_'+str(formatted_number)+str(formatted_redshift), fractionUnscattered) 
                np.save(opdir+'pdms_BP_'+str(formatted_number)+str(formatted_redshift),pdms)
            else:
                # fill in based on other else conditions
                np.save(opdir+'probScat_BP_'+str(formatted_number)+str(formatted_redshift), np.zeros([len(xProbScat),bbins]))
                np.save(opdir+'fractionUnscattered_BP_'+str(formatted_number)+str(formatted_redshift), np.ones(bbins))
                np.save(opdir+'pdms_BP_'+str(formatted_number)+str(formatted_redshift),np.zeros([len(DMThresh[:-1]),bbins]))

