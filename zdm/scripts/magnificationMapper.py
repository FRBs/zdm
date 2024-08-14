import scipy.signal
import scipy.interpolate
import scipy.integrate
import matplotlib.pyplot as plt 
import numpy as np
from astropy.cosmology import Planck18 as cosmo
from astropy.cosmology import LambdaCDM
from astropy import units as u
from astropy import constants as const
import matplotlib.pyplot as plt
import numpy as np
#import dynspec
#import pygedm
import astropy.coordinates as c
from astropy.io import fits
from astropy import wcs
import astropy
from astropy.convolution import Gaussian2DKernel

def offSetBeamGains(bPos, imageCoords, beamSigma):
    xOffset = imageCoords[0] - bPos[0]
    yOffset = imageCoords[1] - bPos[1]
    radiusOffset = (xOffset**2+yOffset**2)**0.5
    gains = BFG(radiusOffset, beamSigma)
    return gains

def BFG(x, sigma):
    return np.exp(-1/2*(x**2/(sigma**2)))


def logSpaceIntegrand(logmu, func, funcArgs, base):
    """Useful for evaluating integrals of func in log space"""
    return func(base**logmu, funcArgs)*base**logmu*np.log(base)

def unnormalisedLensFuncAtSubBeam(log10b, dlog10b, OmegaB, bGains, pixRes, magni, muThresh):
    #OmegaB in arcminutes^2, same as pixRes
    print(log10b, dlog10b)
    inBeam = np.abs(np.log10(bGains)-log10b)<np.abs(dlog10b/2)
    if np.sum(inBeam)>0:
        gtrMu = np.zeros(len(muThresh))
        for i in range(len(muThresh)):
            gtrMu[i] = np.sum(magni[inBeam]>=muThresh[i]) #inside the sum multiply by the
                                                          #image number matrix, then handle
                                                          #normalisation later, check image set
                                                          # for regions of zero images
                                                          # if it's only the outside edges
                                                          # set them to one then that helps sort
                                                          # sort out the issue of outside array 
                                                          # deflections, beam is generally larger
                                                          # than the image array so should be fine
        modelledArea = np.sum(inBeam)*(pixRes[0]*pixRes[1])
        numUnmodelledCells = (OmegaB - modelledArea)/(pixRes[0]*pixRes[1])
        print('num in beam', np.sum(inBeam))
        extra1SWhere = muThresh<1
        gtrMu[extra1SWhere] = gtrMu[extra1SWhere]+numUnmodelledCells
        probUN = (-1*np.diff((gtrMu))/np.diff(muThresh)/muThresh[:-1])
        # smoothingKernel = scipy.signal.windows.gaussian(len(muThresh[:-1]),0.05/(np.mean(np.diff(np.log10(muThresh)))))
        # probUNSmooth= np.convolve(probUN,smoothingKernel, mode='same')/np.sum(smoothingKernel)
        interpFunc = scipy.interpolate.interp1d(np.log10(muThresh[:-1])+np.diff(np.log10(muThresh))[0], probUN, bounds_error=False, fill_value=0)
    else: 
        interpFunc = None
    return interpFunc

def mapRescaler(opdir, fileList, zTrue, zNew):
    scale_factor = (cosmo.angular_diameter_distance(zNew)/cosmo.angular_diameter_distance(zTrue)).value
    for i in range(len(fileList)):
        hdulist = fits.open(fileList[i])
        header = hdulist[0].header
        header['CDELT1'] *= scale_factor
        header['CDELT2'] *= scale_factor
        fits.writeto(opdir+fileList[i],hdulist[0].data, header, overwrite=True)
        hdulist.close()
    

def mapWidener(magni, wideningFrac):
    smoothKernel = Gaussian2DKernel(int(magni.shape[0]/100))
    smoothMagni = scipy.signal.fftconvolve(np.log10(magni), smoothKernel, mode='valid')
    croppedEachSide = np.floor((np.asarray(magni.shape) - np.asarray(smoothMagni.shape))/2)
    interpFunc = scipy.interpolate.RegularGridInterpolator((np.arange(int(croppedEachSide[0]), (magni.shape[0] - int(croppedEachSide[0])),1), np.arange(int(croppedEachSide[1]), (magni.shape[1] - int(croppedEachSide[1])),1)), (smoothMagni), bounds_error=False, fill_value=None)
    wA = int(magni.shape[0]*wideningFrac/2)
    x = np.meshgrid(np.arange(-wA,magni.shape[0]+wA,1), np.arange(-wA,magni.shape[1]+wA,1))
    expandedMagni = interpFunc((x[1].flatten(), x[0].flatten()))
    finalMagni = (expandedMagni.reshape(np.asarray(magni.shape)+wA*2))
    finalMagni[finalMagni<0] = 0
    finalMagni[:wA,:wA] = np.mean(finalMagni[wA:-wA,:wA])
    finalMagni[-wA:, :wA] = np.mean(finalMagni[-wA:, wA:-wA])
    finalMagni[-wA:,-wA:] = np.mean(finalMagni[wA:-wA, -wA:])
    finalMagni[:wA, -wA:] = np.mean(finalMagni[:wA, wA:-wA])
    completeMagni = finalMagni
    completeMagni[wA:-wA,wA:-wA] = np.log10(magni)
    return 10**completeMagni, x


def normalisedLensFuncsAcrossBeam(D, freq, thresh, nbins, bPos, proj, magni, name, muThresh = 10**(np.arange(-3,6,0.02)+0.05)):
    FWHM = 1.22*(const.c/(freq))/D
    beamSigma=(FWHM/2.)*(2*np.log(2))**-0.5
    dlnb=-np.log(thresh)/nbins
    log10min=np.log10(thresh)
    dlog10b=log10min/nbins
    log10b=(np.arange(nbins)+0.5)*dlog10b
    OmegaB= (2*np.pi*dlnb*(beamSigma*180/np.pi*60)**2).decompose().value
    pixRes = np.abs(np.diag(proj.pixel_scale_matrix*60))
    dataEdge = np.mean(np.concatenate((magni[:,0], magni[:,-1], magni[0,:], magni[-1,:])))
    count = 0
    x = np.meshgrid(np.arange(magni.shape[0]), np.arange(magni.shape[1]))
    tempMagni = magni.copy()
    while (dataEdge -1) > 0.1:                                        # couldnt find anything so do this downsampling so just do it yourself
        # here we want to form a 2D interpolation of the magni map based on a downsampled version, then extrapolate using that model
        # to higher separations and infill the higher res map with extrapolated values. 
        tempMagni,x = mapWidener(magni, 0.5+0.1*count)
        dataEdge = np.mean(np.concatenate((tempMagni[:,0], tempMagni[:,-1], tempMagni[0,:], tempMagni[-1,:])))
        count = count+1
        print('trapped forever', count)
    magni = tempMagni
    imageCoords = proj.array_index_to_world_values(x[0], x[1])
    bGains = offSetBeamGains(bPos, imageCoords, beamSigma.decompose().value*180/np.pi)
#    fig = plt.figure()
#    ax = plt.subplot(111, projection=proj)
#    tower = np.zeros(bGains.shape)
#    for i in range(len(log10b)):
#        gainLevel = np.abs(np.log10(bGains)-log10b[i])<np.abs(dlog10b/2)
#        tower[gainLevel] = i
#    plt.imshow(tower, extent=[0,len(x[0][:,0]),0,len(x[0][0,:])], cmap='tab10', vmin=0, vmax=(len(log10b)-1))
#    ax.imshow(np.log10(magni).T, aspect='auto', extent=[0,len(x[0][:,0]),0,len(x[0][0,:])], alpha=0.7)
#    ax.imshow(bGains, alpha=0.5,extent=[0,len(x[0][:,0]),0,len(x[0][0,:])], cmap='Greys')
        
#    plt.xlabel(r'RA')
#    plt.ylabel(r'Dec')
#    overlay = ax.get_coords_overlay('icrs')
#    overlay.grid(color='white', ls='dotted')
#    fig.savefig(str(name))
#    plt.close()
    pmus = np.zeros([len(muThresh),len(log10b)])
    probMags = np.log10(muThresh[:-1])
    for i in range(len(log10b)):
        print('beaming like crazy right now', i)
        interpFunc = unnormalisedLensFuncAtSubBeam(log10b[i], dlog10b, OmegaB, bGains, pixRes, magni, muThresh)
        if interpFunc != None:
            muTwo = np.arange(-2,10,0.01)
            pmus[:,i] = ((1/(np.nansum(10**muTwo*interpFunc(muTwo))*np.diff(muTwo)[0]*np.log(10))*interpFunc(np.log10(muThresh))))
        else: 
            pmus[:,i] = np.nan
    return pmus, magni, x



def clusterDMFuncAcrossBeam(D, freq, thresh, nbins, bPos, proj, clusterRedshift, z, ne, name, lensing, rawWeights, weightsProj, xWeights, DMThresh = np.arange(0,15000,100), scatThresh = 10**np.arange(-4,3,0.01)):
    # assumed that scatThresh is uniformly spaced in log10, if the base is otherwise need to revise integration step evaluations
    FWHM = 1.22*(const.c/(freq))/D
    beamSigma=(FWHM/2.)*(2*np.log(2))**-0.5
    dlnb=-np.log(thresh)/nbins
    log10min=np.log10(thresh)
    dlog10b=log10min/nbins
    log10b=(np.arange(nbins)+0.5)*dlog10b
    OmegaB= (2*np.pi*dlnb*(beamSigma*180/np.pi*60)**2).decompose().value
    pixRes = np.abs(np.diag(proj.pixel_scale_matrix*60))
    x = np.meshgrid(np.arange(ne.shape[0]), np.arange(ne.shape[1]))
    imageCoords = proj.array_index_to_world_values(x[0], x[1])
    bGains = offSetBeamGains(bPos, imageCoords, beamSigma.decompose().value*180/np.pi)
    if lensing:
        weightCoords = weightsProj.array_index_to_world_values(xWeights[0], xWeights[1])
        bGainsWeights = offSetBeamGains(bPos, weightCoords, beamSigma.decompose().value*180/np.pi)
        pixResWeights = np.abs(np.diag(weightsProj.pixel_scale_matrix*60))
        downSampleFactor = (pixRes/pixResWeights)
        print('downSampleFactor:', downSampleFactor)
        weightsFunc = regridInterpolator(rawWeights, weightCoords, downSampleFactor[0])
    else:
        weightsFunc = lambda x: np.nan

    #fig = plt.figure()
    #ax = plt.subplot(111, projection=proj)
    #tower = np.zeros(bGains.shape)
    #for i in range(len(log10b)):
    #    gainLevel = np.abs(np.log10(bGains)-log10b[i])<np.abs(dlog10b/2)
    #    tower[gainLevel] = i
    #plt.imshow(tower, extent=[0,len(x[0][:,0]),0,len(x[0][0,:])], cmap='tab10', vmin=0, vmax=(len(log10b)-1))
    #ax.imshow((ne*1e6/(1+clusterRedshift)).T, aspect='auto', extent=[0,len(x[0][:,0]),0,len(x[0][0,:])], alpha=0.7)
    #ax.imshow(bGains, alpha=0.5,extent=[0,len(x[0][:,0]),0,len(x[0][0,:])], cmap='Greys')
        
    #plt.xlabel(r'RA')
    #plt.ylabel(r'Dec')
    #overlay = ax.get_coords_overlay('icrs')
    #overlay.grid(color='white', ls='dotted')
    #fig.savefig(str(name)+'DMMap.pdf')
    #plt.close()
    pdms = np.zeros([len(DMThresh[:-1]),len(log10b)])
    probScat = np.zeros([len(scatThresh[:-1]), len(log10b)])
    probMags = (DMThresh[:-1])
    fractionUnscattered = np.zeros(len(log10b))

    for i in range(len(log10b)):
        print('beaming like crazy right now', i)
        if lensing:
            pdms[:,i], probScat[:,i], fractionUnscattered[i] = clusterDMFuncAtSubBeam(log10b[i], dlog10b, OmegaB, freq, bGains, imageCoords, pixResWeights, clusterRedshift, z, scatThresh, ne, DMThresh, lensing, weightsFunc, weightCoords, rawWeights, bGainsWeights)
        else:
            pdms[:,i], probScat[:,i], fractionUnscattered[i] = clusterDMFuncAtSubBeam(log10b[i], dlog10b, OmegaB, freq, bGains, imageCoords, pixRes, clusterRedshift, z, scatThresh, ne, DMThresh, lensing, weightsFunc, np.nan, np.nan, np.nan)

    return probScat, fractionUnscattered, pdms

def regridInterpolator(weights, weightCoords, downSampleFactor):
    #assuming a regular grid for map
    if int(downSampleFactor)==0:
        print('WARNING DM grid finer than magni grid')
        return False
    if int(downSampleFactor)==downSampleFactor:
        smoothingKernel1D = np.ones(int(downSampleFactor))
    else:
        smoothingKernel1D = np.zeros(int(downSampleFactor)+2)
        smoothingKernel1D[1:-1] = 1
        smoothingKernel1D[0] = (downSampleFactor % int(downSampleFactor))/2
        smoothingKernel1D[-1] = (downSampleFactor % int(downSampleFactor))/2
    temp = np.repeat(np.expand_dims(smoothingKernel1D,axis=1),len(smoothingKernel1D),axis=1)
    smoothingKernel = temp*temp.T
    smoothedWeights = scipy.signal.fftconvolve(weights, smoothingKernel, mode='same')
    interpFunc = scipy.interpolate.RegularGridInterpolator((weightCoords[0][:,0], weightCoords[1][0,:]), smoothedWeights)
    return interpFunc 
    

def clusterDMFuncAtSubBeam(log10b, dlog10b, OmegaB, freq, bGains, imageCoords, pixRes, clusterRedshift, z, scatThresh, ne, DMThresh, lensing, weightsFunc, weightCoords, rawWeights, bGainsWeights):
    #OmegaB in arcminutes^2, same as pixRes
    print(log10b, dlog10b)
    inBeam = np.abs(np.log10(bGains)-log10b)<np.abs(dlog10b/2)
    imageRA = imageCoords[0][:,0]
    imageDec = imageCoords[1][0,:]
    imageStep = np.diff(imageRA)[0]    
    lam = (const.c/(freq)).decompose().value

    if lensing:
        inBeam_2 = np.abs(np.log10(bGainsWeights)-log10b)<np.abs(dlog10b/2)
        weights = weightsFunc((imageRA,imageDec))
        DMLessWeights = np.sum(rawWeights*(inBeam_2)*np.abs((weightCoords[0]<np.amax(imageRA))*(weightCoords[1]<np.amax(imageDec))*(weightCoords[0]>np.amin(imageRA))*(weightCoords[1]>np.amin(imageDec))-1).astype(bool))
    else:
        weights = 1.0
        DMLessWeights = 0
 
    

    if np.sum(inBeam)>0:
        gtrDM = np.zeros(len(DMThresh))
        gtrScat = np.zeros([len(scatThresh)])
        probScat = np.zeros([len(scatThresh)-1])
        for i in range(len(DMThresh)):
            gtrDM[i] = np.sum(((ne*1e6/(1+clusterRedshift)*inBeam)>=DMThresh[i])*weights)
        for i in range(len(scatThresh)):
            if z>clusterRedshift:
                scat = (4.1e-5/(1+clusterRedshift)*(lam/1)**4*((cosmo.angular_diameter_distance(clusterRedshift)*cosmo.angular_diameter_distance_z1z2(clusterRedshift,z)/cosmo.angular_diameter_distance(z)).value/1e3)*(8.4e-13*(ne/1e-4)**2*3.08567758e+22/((1+clusterRedshift)**2)/1e12)*(2.06264806e+08)**(1/3)*1e3)
                if(np.amin(scat)<np.amin(scatThresh)):
                    print('WARNING: Scattering outside threshold')
                    print('z = ', z, np.amin(scat), np.amin(scatThresh))
                    break
                gtrScat[i] = np.sum((scat>=scatThresh[i])*weights)
                if i==0:
                    gtrScat[0]=np.sum((scat>=0)*weights)
                    
                probScat[:] = (-1*np.diff(gtrScat[:])/(gtrScat[0]))
            else:
                probScat[:] = 0

        if lensing:
            modelledArea = np.sum(inBeam_2)*(pixRes[0]*pixRes[1])
            print('num in beam', np.sum(inBeam_2))
            print('fraction modelled', np.sum(inBeam_2)*pixRes[0]*pixRes[1]/OmegaB)
        else:
            modelledArea = np.sum(inBeam)*(pixRes[0]*pixRes[1])
            print('num in beam', np.sum(inBeam))
            print('fraction modelled', np.sum(inBeam)*pixRes[0]*pixRes[1]/OmegaB)
               
        
        numUnmodelledCells = (OmegaB - modelledArea)/(pixRes[0]*pixRes[1])
        if numUnmodelledCells < 0:
            numUnmodelledCells = 0
                
        print(OmegaB, pixRes, modelledArea, numUnmodelledCells, DMLessWeights, np.amax(gtrScat))
        fractionUnscattered = (numUnmodelledCells+DMLessWeights)/(np.sum(rawWeights*inBeam_2)+numUnmodelledCells)
        print('fraction unscattered', fractionUnscattered)
        gtrDM[0] = gtrDM[0]+numUnmodelledCells+DMLessWeights
        probUN = (-1*np.diff((gtrDM))/np.diff(DMThresh))
        #interpFunc = scipy.interpolate.interp1d((DMThresh[:-1]), probUN, bounds_error=False, fill_value=0)
    else: 
        gtrScat = np.ones(len(scatThresh[:-1]))*np.nan
        numUnmodelledCells = np.nan
        #interpFunc = None
        probUN = np.ones(len(DMThresh[:-1]))*np.nan
        probScat = np.zeros([len(scatThresh[:-1])])
        fractionUnscattered = 1
    return probUN, probScat, fractionUnscattered

