"""
This library contains routines that interact with
the FRB/astropath module and (optical) FRB host galaxy
information.

It includes the class "host_model" for describing the
intrinsic FRB host galaxy distribution, associated functions,
and the approximate fraction of
detectable FRBs from Marnoch et al (https://doi.org/10.1093/mnras/stad2353)
"""


import numpy as np
from matplotlib import pyplot as plt
from zdm import cosmology as cos
import optical_params as op


class host_model:
    """
    A class to hold information about the intrinsic properties of FRB
    host galaxies. Eventually, this should be expanded to be a
    meta-class with different internal models. But for now, it's
    just a simple one
    
    It has one model for describing the intrinsic distribution
    of host galaxies, and another model for converting
    galaxy intrinsic magnitude to an apparent r-band magnitude
    
    Note that while this class describes the intrinsic "magnitudes",
    really magnitude here is a proxy for whatever parameter is used
    to intrinsically describe FRBs. However, only 1D descriptions are
    currently implemented. Future descriptions will include redshift
    evolution, and 2D descriptions (e.g. mass, SFR) at any given redshift.
    
    """
    def __init__(self,opstate=None,verbose=False):
        """
        Class constructor
        
        Args:
            model_type(int): the method of modelling the host
            args: list of args specific to that model
        
        """
        if opstate is None:
            opstate = op.Hosts()
        
        
        if opstate.AppModelID == 0:
            if verbose:
                print("Initialising simple luminosity function")
            self.CalcApparentMags = SimpleApparentMags
        else:
            raise ValueError("Model ",opstate.AppModelID," not implemented")
        
        if opstate.AbsModelID == 0:
            if verbose:
                print("Describing absolute mags with N independent bins")
        else:
            raise ValueError("Model ",opstate.AbsModelID," not implemented")
        
        
        self.AppModelID = opstate.AppModelID
        self.AbsModelID = opstate.AbsModelID
        
        self.opstate = opstate
        
        self.init_abs_bins()
        self.init_app_bins()
        self.init_model_bins()
        self.init_abs_prior()
        
         
    
    def calc_magnitude_priors(self,zlist:np.ndarray,pzlist:np.ndarray,magbins:np.ndarray):
        """
        Calculates priors as a function of magnitude for
        a given redshift distribution.
        
        Args:
            zlist: list of redshifts
            pz: list of probabilities of the FRB occurring at each of those redshifts
            magbins: list of rbandmags for which to calculate priors. These correspond
                    to
        
        # returns probability-weighted magnitude distribution, as a function of
        # self.AppBins
        
        """
        # we integrate over the host luminsity distribution, parameterised by a histogram
        # as a function of luminosity
        
        # checks that pz is normalised
        pzlist /= np.sum(pzlist)
        
        for i,lum in enumerate(self.luminosities):
            plum=self.plums[i]
            mags = lum_to_mag(zlist,lum)
            temp = np.histogram(mags,weights=pzlist*plum,bins=self.AppBins)
            if i==0:
                pmags = temp
            else:
                pmags += temp
        
        return pmags
      
    def init_abs_prior(self):
        """
        Initialises prior on absolute magnitude of galaxies according to the method.
        
        Args:
            method [int]: method for initialising the prior
                0: uniform prior in log space of absolute magnitude
                1: distribution of galaxy luminosities from XXXX
        """
        
        
        if self.opstate.AbsPriorMeth==0:
            Absprior = np.full([self.NModelBins],1./self.NAbsBins)
        else:
            raise ValueError("Luminosity prior method ",self.opstate.AbsPriorMeth," not implemented")
        
        # checks normalisation
        Absprior /= np.sum(Absprior)
        self.AbsPrior = Absprior
        
    def init_app_bins(self):
        """
        Initialises bins in apparent magnitude
        It uses these to calculate priors for any given
        host galaxy magnitude
        """
        
        self.Appmin = self.opstate.Appmin
        self.Appmax = self.opstate.Appmax
        self.NAppBins = self.opstate.NAppBins
        
        # this creates the bin edges
        self.AppBins = np.linspace(self.Appmin,self.Appmax,self.NAppBins+1)
        dAppBin = self.AppBins[1] - self.AppBins[0]
        self.AppMags = self.AppBins[:-1] + dAppBin/2.
        
    def init_abs_bins(self):
        """
        Initialises internal array of absolute magnitudes
        """  
        # shortcuts
        Absmin = self.opstate.Absmin
        Absmax = self.opstate.Absmax
        NAbsBins = self.opstate.NAbsBins
        
        
        self.Absmin = Absmin
        self.Absmax = Absmax
        self.NAbsBins = NAbsBins  
        
        # generally large number of abs magnitude bins
        MagBins = np.linspace(Absmin,Absmax,NAbsBins+1)
        dMag = MagBins[1]-MagBins[0]
        AbsMags = MagBins[:-1]+dMag/2. # bin centres
        
        self.MagBins = MagBins
        self.dMag = dMag
        self.AbsMags = AbsMags
    
    def init_model_bins(self):
        """
        Initialises bins for the simple model of an absolute
        magnitude prior.
        
        The base unit here is assumed to be absolute r-band
        magnitude M, but in principle this works for whatever\
        absolute unit is being used by the model.
        """
        NModelBins = self.opstate.NModelBins
        self.NModelBins = NModelBins 
        
        # generally small numbr of model bins
        ModelBins = np.linspace(self.Absmin,self.Absmax,NModelBins)
        
        self.ModelBins = ModelBins
        self.dModel = ModelBins[1]-ModelBins[0]
    
    def get_weights(self):
        """
        Assigns a weight to each of the absolute magnitudes
        in the internal array (which generally is very large)
        in terms of the absolute magnitude parameterisation
        (generally, only a few parameters)
        """
        
        if self.AbsModelID==0:
            # describes absolute magnitudes via NModelBins
            # between AbsMin and AbsMax
            
            # gives mapping from model bins to mag bins
            self.imags = ((self.AbsMags - self.Absmin)/self.dModel).astype('int')
            
            weights = self.AbsPrior[self.imags]
        else:
            raise ValueError("This weighting scheme not yet implemented")
        return weights
    
    def init_zmapping(self,zvals):
        """
        For a set of redshifts, initialise mapping
        between intrinsic magnitudes and apparent magnitudes
        
        This routine only needs to be called once, since the model
        to convert absolute to apparent magnitudes is fixed
        """
        # mapping of apparent to absolute magnitude
        self.zmap = self.CalcApparentMags(self.AbsMags,zvals)
        
        # this maps the weights from the parameter file to the absoluate magnitudes use
        # internally within the program
        self.wmap = self.get_weights()
        # renormalises the weights, so all internal apparent mags sum to unit
        self.wmap /= np.sum(self.wmap)
        
        self.NZ = zvals.size
        
        # for current model, calculate weighted histogram of apparent magnitude
        # for each redshift. Done by converting intrinsic to apparent for each z,
        # then suming up the associated weights
        maghist = np.zeros([self.NAppBins,self.NZ])
        for i in np.arange(self.NZ):
            # creates weighted histogram of apparent magnitudes,
            # using model weights from wmap (which are fixed for all z)
            hist,bins = np.histogram(self.zmap[:,i],weights=self.wmap,bins=self.AppBins)
            # We may want to renormalise this or not. Not 100% sure yet. I think not.
            #print("sum of hist is ",np.sum(hist))
            #hist /= np.sum(hist)
            maghist[:,i] = hist
            #print("For z of ",zvals[i]," sums are phist ",np.sum(hist),np.sum(self.wmap))
        self.maghist = maghist
        
        
    def get_posterior(self, grid, DM):
        """
        Returns posterior redshift distributiuon for a given grid and DM
        magnitude distribution for FRBs of DM given
        a grid object
        """
        # Step 1: get prior on z
        pz = get_pz_prior(grid,DM)
        
        ### STEP 2: get apparent magnitude distribution ###
        if hasattr(DM,"__len__"):
            papps = np.dot(self.maghist,pz)
            
            #print("Shaps of papps is ",papps.shape,self.maghist.shape,pz.shape)
            #for i,z in enumerate(DM):
            #    print(DM[i],np.sum(pz[:,i]),np.sum(self.maghist[:,0]))
        else:
            papps = self.maghist*pz
        #exit()
        
        return papps,pz



def get_pz_prior(grid, DM):
    """
    Returns posterior redshift distributiuon for a given grid and DM
    magnitude distribution for FRBs of DM given
    a grid object
    """
    
    ### STEP 1: get PZ distribution ###
    # the below should work for vectors
    dmvals = grid.dmvals
    ddm = dmvals[1]-dmvals[0]
    idm1 = (np.floor(DM/ddm)).astype('int')
    idm2 = idm1+1
    dm1 = dmvals[idm1]
    dm2 = dmvals[idm2]
    kdm2 = (DM - dm1)/ddm
    kdm1 = 1.-kdm2
    
    pz = kdm1 * grid.rates[:,idm1] + kdm2 * grid.rates[:,idm2]
    pz = pz/np.sum(pz,axis=0)
    return pz
        
def SimpleApparentMags(Abs,zs):
    """
    Function to convert galaxy luminosities to magnitudes
    Luninosities are in units of Lstar
    Magnitudes are r-band magnitudes
    
    Args:
        Abs (float or array of floats): intrinsic galaxy luminosities 
        zs (float or array of floats): redshifts of galaxies
    
    Returns:
        NL x NZ array of magnitudes
    """
    
    # calculates luminosity distances (Mpc)
    lds = cos.dl(zs)
    
    # finds distance relative to absolute magnitude distance
    dabs = 1e-5 # in units of Mpc
    
    # relative magnitude
    dMag = 2.5*np.log10((lds/dabs)**2)
    
    
    if np.isscalar(zs) or np.isscalar(Abs):
        # just return the product, be it scalar x scalar,
        # scalar x array, or array x scalar
        # this also ensures that the dimensions are as expected
        
        ApparentMags = Abs + dMag
        return ApparentMags
    else:
        # Convert to multiplication so we can use
        # numpy.outer
        temp1 = 10**Abs
        temp2 = 10**dMag
        ApparentMags = np.outer(temp1,temp2)
        ApparentMags = np.log10(ApparentMags)
        return ApparentMags
    


def p_unseen(zvals,plot=False):
    """
    Returns probability of a hist being unseen in typical VLT
    observations.
    
    Inputs:
        zvals [float, array]: array of redshifts
    
    Returns:
        fitv [float, array]: p(Unseen) for redshift zvals
    
    """
    # approx digitisation of Figure 3 p(U|z)
    # from Marnoch et al.
    # https://doi.org/10.1093/mnras/stad2353
    rawz=[0.,0.7,0.8,0.9,0.91,0.98,1.15,1.17,1.25,
        1.5,1.7,1.77,1.85,1.95,2.05,2.4,2.6,2.63,
        2.73,3.05,3.75,4.5,4.9,5]
    rawz=np.array(rawz)
    prawz = np.linspace(0,1.,rawz.size)
    
    pz = np.interp(zvals, rawz, prawz)
    
    
    coeffs=np.polyfit(rawz[1:],prawz[1:],deg=3)
    fitv=np.polyval(coeffs,zvals)
    
    if plot:
        plt.figure()
        plt.xlim(0,5.)
        plt.xlabel("$z$")
        plt.ylim(0,1.)
        plt.ylabel('$p_{\\rm U}(z)$')
        plt.plot(rawz,prawz,label='Marnoch et al.')
        plt.plot(zvals,pz,label='interpolation')
        plt.plot(zvals,fitv,label='quadratic')
        plt.legend()
        plt.tight_layout()
        plt.savefig('p_unseen.pdf')
        plt.close()
    
    # checks against values which are too low
    toolow = np.where(fitv<0.)[0]
    fitv[toolow]=0.
    
    # checks against values which are too high
    toohigh = np.where(fitv>1.)[0]
    fitv[toohigh]=1.
    
    return fitv
