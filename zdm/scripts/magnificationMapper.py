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

def unnormalisedLensFuncAtSubBeam(log10b, dlog10b, OmegaB, imagePlaneBGains, bGains, pixRes, magniArr, sourceMagniArr, muThresh):
    #OmegaB in arcminutes^2, same as pixRes
    inBeam = np.abs(np.log10(bGains)-log10b)<np.abs(dlog10b/2)
    planeInBeam = np.abs(np.log10(imagePlaneBGains)-log10b)<np.abs(dlog10b/2)
    if not sourceMagniArr:
        if np.sum(planeInBeam*1 - inBeam*1)>0:
            print(log10b, dlog10b, 'ALERT: somethings dead wrong !!!!!!!!!!!!!!!!!!')
            return np.nan
    if np.sum(inBeam)>0:
        gtrMu = np.zeros(len(muThresh))
        for i in range(len(muThresh)):
            gtrMu[i] = np.sum(magniArr[inBeam]>=muThresh[i]) 
        modelledArea = np.sum(planeInBeam)*(pixRes[0]*pixRes[1])
        numUnmodelledCells = (OmegaB - modelledArea)/(pixRes[0]*pixRes[1])
        print('num in beam', np.sum(inBeam))
        extra1SWhere = muThresh<1
        gtrMu[extra1SWhere] = gtrMu[extra1SWhere]+numUnmodelledCells
        if sourceMagniArr:
            probUN = (-1*np.diff((gtrMu))/np.diff(muThresh))
        else:
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
        if 'CD1_1' in header:
            header['CD1_1'] *= scale_factor
            header['CD1_2'] *= scale_factor
            header['CD2_1'] *= scale_factor
            header['CD2_2'] *= scale_factor
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

def normalisedLensFuncsAcrossBeam(D, freq, thresh, nbins, bPos, proj, x, magniArr, pixCoordsArr, name, sourceMagniArr=False, muThresh = 10**(np.arange(-3,2,0.02)+0.05)):
    FWHM = 1.22*(const.c/(freq))/D
    beamSigma=(FWHM/2.)*(2*np.log(2))**-0.5
    dlnb=-np.log(thresh)/nbins
    log10min=np.log10(thresh)
    dlog10b=log10min/nbins
    log10b=(np.arange(nbins)+0.5)*dlog10b
    OmegaB= (2*np.pi*dlnb*(beamSigma*180/np.pi*60)**2).decompose().value
    pixRes = np.abs(np.diag(proj.pixel_scale_matrix*60))

    if sourceMagniArr:
        pixCoords = []
        pixCoords.append(pixCoordsArr[:,0])
        pixCoords.append(pixCoordsArr[:,1])
    else:
        dataEdge = np.mean(np.concatenate((magniArr[:,0], magniArr[:,-1], magniArr[0,:], magniArr[-1,:])))
        count = 0
        tempMagni = magniArr.copy()
        #while (dataEdge -1) > 0.1: 
        #    tempMagni,x = mapWidener(magniArr, 0.5+0.1*count)
        #    dataEdge = np.mean(np.concatenate((tempMagni[:,0], tempMagni[:,-1], tempMagni[0,:], tempMagni[-1,:])))
        #    count = count+1
        #    print('trapped forever', count)
        magniArr = tempMagni
        pixCoords = x
    
    imageCoords = proj.array_index_to_world_values(pixCoords[0], pixCoords[1])
    bGains = offSetBeamGains(bPos, imageCoords, beamSigma.decompose().value*180/np.pi)
    imagePlaneCoords = proj.array_index_to_world_values(x[0], x[1])
    imagePlaneBGains = offSetBeamGains(bPos, imagePlaneCoords, beamSigma.decompose().value*180/np.pi)



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
        interpFunc = unnormalisedLensFuncAtSubBeam(log10b[i], dlog10b, OmegaB, imagePlaneBGains, bGains, pixRes, magniArr, sourceMagniArr, muThresh)
        if interpFunc != None:
            muTwo = np.arange(-2,10,0.01)
            pmus[:,i] = ((1/(np.nansum(10**muTwo*interpFunc(muTwo))*np.diff(muTwo)[0]*np.log(10))*interpFunc(np.log10(muThresh))))
        else: 
            pmus[:,i] = np.nan
    return pmus, magniArr, x



def clusterDMFuncAcrossBeam(D, freq, thresh, nbins, bPos, proj, clusterRedshift, z, ne, name, weights, imageProj, imageCoords, DMThresh = np.arange(0,15000,200), scatThresh = 10**np.arange(-4,3,0.02)):
    # assumed that scatThresh is uniformly spaced in log10, if the base is otherwise need to revise integration step evaluations
    FWHM = 1.22*(const.c/(freq))/D
    beamSigma=(FWHM/2.)*(2*np.log(2))**-0.5
    dlnb=-np.log(thresh)/nbins
    log10min=np.log10(thresh)
    dlog10b=log10min/nbins
    log10b=(np.arange(nbins)+0.5)*dlog10b
    OmegaB= (2*np.pi*dlnb*(beamSigma*180/np.pi*60)**2).decompose().value

    pixRes = np.abs(np.diag(proj.pixel_scale_matrix*60))
    neX = np.meshgrid(np.arange(ne.shape[1]), np.arange(ne.shape[0]))
    neCoords = proj.array_index_to_world_values(neX[0], neX[1])
    neBGains = offSetBeamGains(bPos, neCoords, beamSigma.decompose().value*180/np.pi)
    imageBGains = offSetBeamGains(bPos, imageCoords, beamSigma.decompose().value*180/np.pi)
    pixResWeights = np.abs(np.diag(imageProj.pixel_scale_matrix*60))

    pdms = np.zeros([len(DMThresh[:-1]),len(log10b)])
    probScat = np.zeros([len(scatThresh[:-1]), len(log10b)])
    probMags = (DMThresh[:-1])
    fractionUnscattered = np.zeros(len(log10b))

    #tempX = np.flip(neCoords[0][:,0])
    tempX = (neCoords[0][:,0])
    
    binRangeX = np.append(tempX-np.mean(np.diff(tempX))/2, np.amax(tempX)+np.mean(np.diff(tempX))/2)
    binRangeY = np.append(neCoords[1][0,:]-np.mean(np.diff(neCoords[1][0,:]))/2, np.amax(neCoords[1][0,:])+np.mean(np.diff(neCoords[1][0,:]))/2)

    neWeightedHist = np.histogram2d(imageCoords[0].flatten(), imageCoords[1].flatten(), bins=[binRangeX,binRangeY], weights=weights.flatten())

    for i in range(len(log10b)):
        pdms[:,i], probScat[:,i], fractionUnscattered[i] = clusterDMFuncAtSubBeam(log10b[i], dlog10b, OmegaB, freq, neBGains, neWeightedHist, pixResWeights, clusterRedshift, z, scatThresh, ne, DMThresh, imageBGains, weights)


    return probScat, fractionUnscattered, pdms

def clusterDMFuncAtSubBeam(log10b, dlog10b, OmegaB, freq, neBGains, neWeightedHist, pixRes, clusterRedshift, z, scatThresh, ne, DMThresh, imageBGains, weights):
    #OmegaB in arcminutes^2, same as pixRes
    inBeam = np.abs(np.log10(neBGains)-log10b)<np.abs(dlog10b/2)
    lam = (const.c/(freq)).decompose().value

    inBeam_2 = np.abs(np.log10(imageBGains)-log10b)<np.abs(dlog10b/2)
    DMLessWeights = np.sum(weights*(inBeam_2))-np.sum(neWeightedHist[0][inBeam])
 

    if np.sum(inBeam)>0:
        gtrDM = np.zeros(len(DMThresh))
        gtrScat = np.zeros([len(scatThresh)])
        probScat = np.zeros([len(scatThresh)-1])
        for i in range(len(DMThresh)):
            gtrDM[i] = np.sum((neWeightedHist[0]*inBeam)*((1e6/(1+clusterRedshift)*ne)>=DMThresh[i]))
        for i in range(len(scatThresh)):
            if z>clusterRedshift:
                scat = (4.1e-5/(1+clusterRedshift)*(lam/1)**4*((cosmo.angular_diameter_distance(clusterRedshift)*cosmo.angular_diameter_distance_z1z2(clusterRedshift,z)/cosmo.angular_diameter_distance(z)).value/1e3)*(8.4e-13*(ne/1e-4)**2*3.08567758e+22/((1+clusterRedshift)**2)/1e12)*(2.06264806e+9)**(1/3)*1e3)
                if(np.amin(scat)<np.amin(scatThresh) and np.amin(scat)>0):
                    print('WARNING: Scattering outside threshold')
                    print('z = ', z, np.amin(scat), np.amin(scatThresh))
                    break
                gtrScat[i] = np.sum((scat>=scatThresh[i])*neWeightedHist[0]*inBeam)
                if i==0:
                    gtrScat[0]=np.sum((scat>=0)*neWeightedHist[0]*inBeam)
                    
                probScat[:] = (-1*np.diff(gtrScat[:])/(gtrScat[0]))
            else:
                probScat[:] = 0

        modelledArea = np.sum(inBeam_2)*(pixRes[0]*pixRes[1])
               
        
        numUnmodelledCells = (OmegaB - modelledArea)/(pixRes[0]*pixRes[1])
        if numUnmodelledCells < 0:
            numUnmodelledCells = 0
                
        fractionUnscattered = (numUnmodelledCells+DMLessWeights)/(np.sum(weights*inBeam_2)+numUnmodelledCells)
        print(log10b, dlog10b, 'fraction unscattered', fractionUnscattered, 'fraction modelled', modelledArea/OmegaB, OmegaB, pixRes, np.sum(inBeam_2), modelledArea, numUnmodelledCells, DMLessWeights, np.amax(gtrScat))
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

