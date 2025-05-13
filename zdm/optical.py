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
from zdm import optical_params as op
from scipy.interpolate import CubicSpline

class host_model:
    """
    A class to hold information about the intrinsic properties of FRB
    host galaxies. Eventually, this should be expanded to be a
    meta-class with different internal models. But for now, it's
    just a simple one
    
    Ingredients are:
        A model for describing the intrinsic distribution
            of host galaxies. This model must be described 
            by some set of parameters, and be able to return a
            prior as a function of intrinsic host galaxy magnitude.
            This model is initialised via opstate.AbsModelID
            
        A model for converting absolute to apparent host magnitudes.
            This is by defult an apparent r-band magnitude, though
            in theory a user can work in whatever band they wish.
            
    Internally, this class initialises:
        An array of absolute magnitudes, which get weighted according
        to the host model.
        Internal variables associated with this are prefaced "Model"
        
        An array of apparent magnitudes, which is used to compare with
        host galaxy candidates
        Internal variables associated with this are prefaced "App"
        
        Arrays mapping intrinsic to absolute magnitude as a function
        of redshift, to allow quick estimation of p(apparent_mag | DM)
        for a given FRB survey with many FRBs
        Internal variables associated with this are prefaced "Abs"
    
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
            # must take arguments of (absoluteMag,z)
            self.CalcApparentMags = SimpleApparentMags
        else:
            raise ValueError("Model ",opstate.AppModelID," not implemented")
        
        if opstate.AbsModelID == 0:
            if verbose:
                print("Describing absolute mags with N independent bins")
        elif opstate.AbsModelID == 1:
            if verbose:
                print("Describing absolute mags with spline interpoilation of N points")
        else:
            raise ValueError("Model ",opstate.AbsModelID," not implemented")
        
        
        self.AppModelID = opstate.AppModelID
        self.AbsModelID = opstate.AbsModelID
        
        self.opstate = opstate
        
        self.init_abs_bins()
        self.init_model_bins()
        self.init_app_bins()
        self.init_abs_prior()
        
        self.ZMAP = False # records that we need to initialise this
        
    #############################################################
    ################## Initialisation Functions #################
    #############################################################
    
    def init_abs_prior(self):
        """
        Initialises prior on absolute magnitude of galaxies according to the method.
        
        Args:
            method [int]: method for initialising the prior
                0: uniform prior in log space of absolute magnitude
                1: distribution of galaxy luminosities from XXXX
        """
        
        if self.opstate.AbsPriorMeth==0:
            Absprior = np.full([self.ModelNBins],1./self.NAbsBins)
        else:
            raise ValueError("Luminosity prior method ",self.opstate.AbsPriorMeth," not implemented")
        
        # checks normalisation
        Absprior /= np.sum(Absprior)
        self.AbsPrior = Absprior
        
        
        # this maps the weights from the parameter file to the absoluate magnitudes use
        # internally within the program. We now initialise this during an "init"
        self.AbsMagWeights = self.init_abs_mag_weights()
        
        # renormalises the weights, so all internal apparent mags sum to unit
        # include this step in the init routine perhaps?
        self.AbsMagWeights /= np.sum(self.AbsMagWeights)
        
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
        self.dAppmag = dAppBin
        
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
        ModelNBins = self.opstate.NModelBins
        self.ModelNBins = ModelNBins
        
        if self.AbsModelID == 0:
            # bins are centres
            dbin = (self.Absmax - self.Absmin)/ModelNBins
            ModelBins = np.linspace(self.Absmin+dbin/2.,self.Absmax-dbin/2.,ModelNBins)
            
        elif self.AbsModelID == 1:
            # bins on edges
            ModelBins = np.linspace(self.Absmin,self.Absmax,ModelNBins)
        
        self.ModelBins = ModelBins
        self.dModel = ModelBins[1]-ModelBins[0]
    
    def init_zmapping(self,zvals):
        """
        For a set of redshifts, initialise mapping
        between intrinsic magnitudes and apparent magnitudes
        
        This routine only needs to be called once, since the model
        to convert absolute to apparent magnitudes is fixed
        
        It is not set automatically however, and needs to be called
        with a set of z values.
        """
        
        # records that this has been initialised
        self.ZMAP = True
        
        # mapping of apparent to absolute magnitude
        self.zmap = self.CalcApparentMags(self.AbsMags,zvals)
        self.zvals = zvals
        self.NZ = self.zvals.size
        
        self.init_maghist()
    
    def init_maghist(self):
        """
        Initialises the array mapping redshifts and absolute magnitudes
        to redshift and apparent magnitude
        """
        
        # for current model, calculate weighted histogram of apparent magnitude
        # for each redshift. Done by converting intrinsic to apparent for each z,
        # then suming up the associated weights
        maghist = np.zeros([self.NAppBins,self.NZ])
        for i in np.arange(self.NZ):
            # creates weighted histogram of apparent magnitudes,
            # using model weights from wmap (which are fixed for all z)
            hist,bins = np.histogram(self.zmap[:,i],weights=self.AbsMagWeights,bins=self.AppBins)
            
            # # NOTE: these should NOT be re-normalised, since the normalisation reflects
            # true magnitudes which fall off the apparent magnitude histogram.
            maghist[:,i] = hist
            
            
        self.maghist = maghist
        
    def reinit_model(self):
        """
        Re-initialises all internal info which depends on the optical
        param model. It assumes that the changes have been implemented in
        self.AbsPrior
        """
        
        # this maps the weights from the parameter file to the absoluate magnitudes use
        # internally within the program. We now initialise this during an "init"
        self.AbsMagWeights = self.init_abs_mag_weights()
        
        # renormalises the weights, so all internal apparent mags sum to unity
        # include this step in the init routine perhaps?
        self.AbsMagWeights /= np.sum(self.AbsMagWeights)
        
        self.init_maghist()
    
    def init_abs_mag_weights(self):
        """
        Assigns a weight to each of the absolute magnitudes
        in the internal array (which generally is very large)
        in terms of the absolute magnitude parameterisation
        (generally, only a few parameters)
        """
        
        if self.AbsModelID==0:
            # describes absolute magnitudes via ModelNBins
            # between AbsMin and AbsMax
            # coefficients at centre of bins
            
            # gives mapping from model bins to mag bins
            self.imags = ((self.AbsMags - self.Absmin)/self.dModel).astype('int')
            
            #rounding errors
            toohigh = np.where(self.imags == self.ModelNBins)
            self.imags[toohigh] = self.ModelNBins-1
            
            weights = self.AbsPrior[self.imags]
        elif self.AbsModelID == 1:
            # As above, but with spline interpolation of model.
            # coefficients span full range
            cs = CubicSpline(self.ModelBins,self.AbsPrior)
            weights = cs(self.AbsMags)
            toolow = np.where(weights < 0.)
            weights[toolow] = 0.
        else:
            raise ValueError("This weighting scheme not yet implemented")
        return weights
    
    
    #############################################################
    ##################    Path Calculations    #################
    #############################################################
    
    def estimate_unseen_prior(self,mag_limit):
        """
        Calculates PU, the prior that an FRB host galaxy of a
        particular DM is unseen in the optical image
        
        This requires initialisation of init_path_raw_prior_Oi
        
        NOTE: The total normalisation of priors in the magnitude range
            may be less than unity. This is because some probability
            may fall outside of the magnitude range being examined.
            Hence, the correct normalisation is found by summing
            the visible magnitudes, and subtracting them from unity. 
        
        """
        
        visible = np.where(self.AppMags < mag_limit)[0]
        
        PSeen = np.sum(self.priors[visible])
        PU = 1.-PSeen
        return PU
    
    def path_raw_prior_Oi(self,mags,ang_sizes,Sigma_ms):
        """
        Function to pass to astropath module
        for calculating a raw FRB prior.
        
        Args:
            mags: tuple of host galaxy r-band magnitude
            ang_sizes: tuple of host galaxy angular size
            Sigma_ms: tuple of galaxy densities on the sky
        
        
        NOTE: as of all recent PATH iterations, the galaxy angular
            size should NOT be included in the calculation, since
            this gets integrated over in estimating the offset error.
            Nonetheless, this function *must* accept an ang_size
            argument.
        
        NOTE2: Before using this function, the call "init_path_raw_prior_Oi"
            must be called. This is because the full prior requires a zDM
            grid and an FRB DM, yet this cannot be passed to raw_prior_Oi
            within PATH.
        """
        
        ngals = len(mags)
        Ois = []
        for i,mag in enumerate(mags):
            
            #print(mag)
            # calculate the bins in apparent magnitude prior
            kmag2 = (mag - self.Appmin)/self.dAppmag
            imag1 = int(np.floor(kmag2))
            
            # careful with interpolation - priors are for magnitude bins
            # with bin edges give by Appmin + N dAppmag.
            # We probably want to smooth this eventually due to minor
            # numerical tweaks
            
            #kmag2 -= imag1
            #kmag1 = 1.-kmag2
            #imag2 = imag1+1
            #prior = kmag1*self.priors[imag1] + kmag2*self.priors[imag2]
            
            # very simple - just gives probability for bin it's in
            Oi = self.priors[imag1]
            Oi /= Sigma_ms[i] # normalise by host counts
            Ois.append(Oi)
        
        Ois = np.array(Ois)
        return Ois
    
    def init_path_raw_prior_Oi(self,DM,grid):
        """
        Initialises the priors for a particlar DM.
        This performs a function very similar to
        "get_posterior" except that it expicitly
        only operates on a single DM, and saves the
        information intrnally so that
        path_raw_prior_Oi can be called for numerous
        host galaxy candidates.
        
        It returns the priors distribution
        
        """
        
        # we start by getting the posterior distribution p(z)
        # for an FRB with DM DM seen by the 'grid'
        pz = get_pz_prior(grid,DM)
        
        # we now calculate the list of priors - for the array
        # defined by self.AppBins with bin centres at self.AppMags
        priors = self.calc_magnitude_priors(grid.zvals,pz)
        
        # stores knowledge of the DM used to calculate the priors
        self.prior_DM = DM
        self.priors = priors
        
        return priors
    
    
    def calc_magnitude_priors(self,zlist:np.ndarray,pzlist:np.ndarray):
        """
        Calculates priors as a function of magnitude for
        a given redshift distribution.
        
        Args:
            zlist: list of redshifts
            pz: list of probabilities of the FRB occurring at each of those redshifts
        
        # returns probability-weighted magnitude distribution, as a function of
        # self.AppBins
        
        """
        # we integrate over the host absolute magnitude distribution
        
        # checks that pz is normalised
        pzlist /= np.sum(pzlist)
        
        for i,absmag in enumerate(self.AbsMags):
            plum = self.AbsMagWeights[i]
            mags = self.CalcApparentMags(absmag,zlist)
            temp,bins = np.histogram(mags,weights=pzlist*plum,bins=self.AppBins)
            if i==0:
                pmags = temp
            else:
                pmags += temp
        
        return pmags
    
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
    Function to convert galaxy absolue to apparent magnitudes.
    
    Nomically, magnitudes are r-band magnitudes, but this function
    is so simple it doesn't matter.
    
    Just applies a distance correction - no k-correction.
    
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
    

def p_unseen_Marnoch(zvals,plot=False):
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
