"""
This file has been updated and adapted from original code produced
by Ziteng (Andy) Wang. The original is part of the
CRAFT code repository, and can be found here:
https://github.com/askap-craco/craco-beam
"""

import numpy as np
import os
import logging
from matplotlib import pyplot as plt

logging.basicConfig(
    level = logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)

logger = logging.getLogger(__name__)


def Gauss(r, sigma):
    """
    simple Gaussian function
    """
    return np.exp(-0.5 * (r / sigma) ** 2)

def hist_craco_beams(footprint, pitch, freq, gsize, gpix,plot=False):
    """
    Generates a beam histogram for the specified info
    """
    
    infile = f"./craco_{footprint}_p{pitch:.2f}_f{freq:.1f}MHz_f{gsize:.1f}d_npix{gpix}.npy"
    beamshape = np.load(infile)
    
    envelope = np.max(beamshape,axis=0)
    gs = 10.
    npix = 2560
    rads = np.linspace(-gs/2.,gs/2.,npix) * np.pi/180.
    xs = np.repeat(rads,npix)
    xs = xs.reshape([npix,npix])
    ys = np.copy(xs)
    ys = ys.T
    
    ddeg = rads[1]-rads[0]
    dsolid = ddeg**2
    
    # calculate solid angle per grid point
    solids = np.cos(xs**2 + ys**2) #**0.5
    solids *= dsolid 
    
    bmin = -3
    bmax = 0
    nbins = 300
    db = (bmax - bmin)/nbins
    binedges = np.logspace(bmin,bmax,nbins+1)
    h,b = np.histogram(envelope.flatten(),weights = solids.flatten(),bins=binedges)
    
    bcs = 10**np.linspace(bmin + db/2.,bmax - db/2.,nbins)
    
    basename = f"./hist_craco_{footprint}_p{pitch:.2f}_f{freq:.1f}MHz_f{gsize:.1f}d_npix{gpix}.npy"
    
    np.save(basename,h)
    np.save("craco_histogram_bins.npy",binedges)
    
    if plot:
        
        x1 = np.load("../data/BeamData/ASKAP_1300_bins.npy")
        y1 = np.load("../data/BeamData/ASKAP_1300_hist.npy")
        dx1 = x1[1]/x1[0]
        xbar1 = x1[:-1] * dx1**0.5
        
        plt.figure()
        plt.plot(bcs,h,label="CRACO square")
        plt.plot(xbar1,y1,label="ICS closepack 1300 MHz")
        
        plt.xlabel("$B$")
        plt.ylabel("$\\Omega(B)$")
        plt.xscale('log')
        plt.yscale('log')
        plt.show()

class AskapBeam:
    def __init__(
        self, footprint="square", pitch=0.9, freq=920.5,
    ):
        self.footprint = footprint # askap footprint, including square and closepack
        self.pitch = pitch # beam separation in degrees
        self.freq = freq # frequency in MHz

        # self.get_aval = self._load_aval_func()

        self.tsys = self._load_tsys()
        self.beamscentre = self._gen_askap_beamcentre()
        self.beamwidth = self._get_primary_beamwidth()
        self.avals = self._load_aval()
        
    def _load_tsys(self):
        coeffs = np.load("askap_tsys_polyval.npy")
        return np.polyval(coeffs, self.freq)
    
    def _load_aval(self):
        data = np.loadtxt("avals.dat")
        x = data[:,0]
        y = data[:,1]
        newx = np.concatenate([x,-x[::-1]])
        newy = np.concatenate([y,y[::-1]])

        dists = np.sqrt(self.beamscentre[0] ** 2 + self.beamscentre[1] ** 2)
        return np.interp(dists, newx, newy)
        # return get_aval
    
    def _gen_askap_beamcentre(self):
        """
        generate beam centres
        """
        if self.footprint == "square":
            x=np.linspace(0,5.*self.pitch,6)
            x=np.repeat(x,6)
            x = x.reshape([6,6])
            y=np.copy(x)
            y=y.T
            x -= self.pitch*2.5
            y -= self.pitch*2.5

        elif self.footprint == "closepack":
            x = np.zeros([6,6])
            y = np.zeros([6,6])
            for iy in np.arange(6):
                if iy % 2 == 0: x0 = 0.
                else: x0 = 0.5 * self.pitch
                x[iy,:] = np.arange(6) * self.pitch + x0
                y[iy,:] = iy*self.pitch*3**0.5/2.
            x -= np.sum(x)/36.
            y -= np.sum(y)/36.

        return x.flatten(), y.flatten()
    
    def _get_primary_beamwidth(self):
        D = 12.
        c_light = 299792458.
        HPBW=1.09*(c_light/(self.freq * 1e6))/D # from RACS
        sigma=(HPBW/2.)*(2*np.log(2))**-0.5
        deg_sigma = sigma * 180./np.pi

        return deg_sigma # in the unit of degree
    
    def primary_response(self, xpoints, ypoints, beamx, beamy, beamw=None):
        offsets = (xpoints[None, ...] - beamx) ** 2 + (ypoints[..., None] - beamy) ** 2
        offsets = np.sqrt(offsets)
        if beamw is None: beamw = self.beamwidth
        return Gauss(offsets, beamw)


class CracoBeam:

    def __init__(
        self, fov=1.1, npix=256, sbeam=40./3600.
    ):
        self.fov = fov # single field of view in degree
        self.npix = npix # number of pixel in image domain
        self.sbeam = sbeam # size of the synthesized beam in degree

        self.uvcellsize = 1 / np.deg2rad(self.fov)
        self.imgcellsize = self.fov / self.npix # cell size in degree


    def grid_response(self, xpoints, ypoints, beamx=0., beamy=0.):
        """
        pillbox gridding response i.e., sinc function

        xpoints, ypoints: numpy.ndarray
            coordinates in the unit of degree offset from the phase centre
        """
        # this might be something to do with the FoV
        xpoints = (xpoints - beamx).copy()
        ypoints = (ypoints - beamy).copy()

        ### outside the FoV, make copies...
        # xpoints = (xpoints - self.fov / 2) % self.fov - self.fov / 2
        # ypoints = (ypoints - self.fov / 2) % self.fov - self.fov / 2

        xpoints = np.deg2rad(xpoints)
        ypoints = np.deg2rad(ypoints)

        xpoints, ypoints = np.meshgrid(xpoints, ypoints)

        sin = np.sin; pi = np.pi

        p1 = sin(pi*self.uvcellsize*xpoints) / (pi*self.uvcellsize*xpoints)
        p2 = sin(pi*self.uvcellsize*ypoints) / (pi*self.uvcellsize*ypoints)

        return np.abs(p1 * p2)
    
    def sample_response(self, xpoints, ypoints, beamx=0., beamy=0.):
        """
        response due to insufficient sampling i.e., 4% loss

        xpoints, ypoints: numpy.ndarray
            coordinates in the unit of degree offset from the phase centre
        """
        xpoints = (xpoints - beamx).copy()
        ypoints = (ypoints - beamy).copy()

        xoff = xpoints % self.imgcellsize - self.imgcellsize / 2
        yoff = ypoints % self.imgcellsize - self.imgcellsize / 2

        xoff, yoff = np.meshgrid(xoff, yoff)
        off = np.sqrt(xoff ** 2 + yoff ** 2)
        return Gauss(off, self.sbeam)
    

def _make_response_grid(gsize=10., gpix=2560.):
    """
    given the size of the grid (in degree, diameter), and the number of pixels on each axis,
    return two numpy arrays containing grid coordinates
    """
    xpoints = np.linspace(-gsize/2, gsize/2, gpix)
    ypoints = np.linspace(-gsize/2, gsize/2, gpix)
    return xpoints, ypoints

def _sim_one_beam(ibeam, xpoints, ypoints, askapbeam, cracobeam, ):
    logger.info(f"start simulating craco response for beam{ibeam}...")
    beamx = askapbeam.beamscentre[0][ibeam]
    beamy = askapbeam.beamscentre[1][ibeam]

    ### aval value
    beamaval = askapbeam.avals[ibeam]
    ### primary beam response
    logger.info(f"get primary beam response for beam{ibeam}...")
    primresponse = askapbeam.primary_response(xpoints, ypoints, beamx, beamy, )
    ### craco related response
    logger.info(f"get primary beam response for beam{ibeam}...")
    gridresponse = cracobeam.grid_response(xpoints, ypoints, beamx, beamy, )
    logger.info(f"get sample response for beam{ibeam}...")
    sampresponse = cracobeam.sample_response(xpoints, ypoints, beamx, beamy, )

    beamresponse = beamaval * primresponse * gridresponse * sampresponse
    return beamresponse

def get_craco_allbeams(
        footprint="square", pitch=0.9, freq=920.5, # askap footprint related,
        fov=1.1, npix=256, sbeam=40./3600., # craco related
        gsize=10., gpix=2560 # response gridding params
    ):
        xpoints, ypoints = _make_response_grid(gsize=gsize, gpix=gpix)
        askapbeam = AskapBeam(footprint=footprint, pitch=pitch, freq=freq)
        cracobeam = CracoBeam(fov=fov, npix=npix, sbeam=sbeam)
        
        beamsresponse = np.zeros((36, gpix, gpix))
        for ibeam in range(36): # 36 beams
            beamsresponse[ibeam] = _sim_one_beam(ibeam, xpoints, ypoints, askapbeam, cracobeam, )

        savefname = f"./craco_{footprint}_p{pitch:.2f}_f{freq:.1f}MHz_f{gsize:.1f}d_npix{gpix}.npy"
        np.save(savefname, beamsresponse)
        logger.info(f"save data to {savefname}...")

if __name__ == "__main__":
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    parser = ArgumentParser(description="simulate craco response", formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("-fp", "--footprint", help="askap beam footprint", type=str, default="square")
    parser.add_argument("-p", "--pitch", help="beam pitch", type=float, default=0.9)
    parser.add_argument("-f", "--freq", help="frequency of the observation", type=float, default=920.5)
    parser.add_argument("-fv", "--fov", help="craco image field-of-view", type=float, default=1.1)
    parser.add_argument("-npix", "--npix", help="number of pixel in craco image", type=int, default=256)
    parser.add_argument("-sb", "--sbeam", help="size of the synthesized beam", type=float, default=40./3600.)
    parser.add_argument("-gs", "--gridsize", help="grid size in the unit of degree", type=float, default=10.)
    parser.add_argument("-gn", "--gridnpix", help="number of pixel in the grid", type=int, default=2560)
    values = parser.parse_args()
    
    footprint=values.footprint
    pitch=values.pitch
    freq=values.freq
    fov=values.fov
    npix=values.npix
    sbeam=values.sbeam
    gsize=values.gridsize
    gpix=values.gridnpix
            
    opfile = f"./craco_{footprint}_p{pitch:.2f}_f{freq:.1f}MHz_f{gsize:.1f}d_npix{gpix}.npy"
    if not os.path.exists(opfile):
        get_craco_allbeams(
            footprint=footprint, pitch=pitch, freq=freq,
            fov=values.fov, npix=values.npix, sbeam=sbeam, gsize=gsize,
            gpix=gpix
        )
    else:
        hist_craco_beams(footprint, pitch, freq, gsize, gpix,plot=True)


        
