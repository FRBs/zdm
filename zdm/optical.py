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
import os
from importlib import resources
import pandas


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
            opstate (class: Hosts, optional): class defining parameters
                of optical state model
            verbose (bool, optional): to be verbose y/n
        
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
        
        """
        
        if self.opstate.AbsPriorMeth==0:
            # uniform prior in log space of absolute magnitude
            Absprior = np.full([self.ModelNBins],1./self.NAbsBins)
        else:
            # other methods to be added as required
            raise ValueError("Luminosity prior method ",self.opstate.AbsPriorMeth," not implemented")
        
        # enforces normalisation of the prior to unity
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
        It uses these to calculate priors for any given host galaxy magnitude.
        This is a very simple set of uniformly log-spaced bins in magnitude space,
        and linear interpolation is used between them.
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
        This is a simple set of uniformly log-spaced bins in terms
        of absolute magnitude, which the absolute magnitude model gets
        projected onto
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
        magnitude prior. This is much sparser than the app or
        abs bins, since these model bins correspond to 
        parameters which may get adjusted by e.g. an MCMC
        
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
        with a set of z values. This is all for speedup purposes.
        
        Args:
            zvals (np.ndarray, float): array of redshifts over which
                to map absolute to apparent magnitudes.
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
        
        Calculates the internal maghist array, of size self.NAppBins X self.NZ
        
        No return value.
            
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
        
        args:
            mag_limit (float): maximum observable magnitude of host galaxies
        
        returns:
            PU (float): probability PU of true hist being unseen in the optical
                            image.
        
        """
        
        invisible = np.where(self.AppMags > mag_limit)[0]
        
        PU = np.sum(self.priors[invisible])
        
        return PU
    
    def path_raw_prior_Oi(self,mags,ang_sizes,Sigma_ms):
        """
        Function to pass to astropath module
        for calculating a raw FRB prior.
        
        
        NOTE: as of all recent PATH iterations, the galaxy angular
            size should NOT be included in the calculation, since
            this gets integrated over in estimating the offset error.
            Nonetheless, this function *must* accept an ang_size
            argument.
        
        NOTE2: Before using this function, the call "init_path_raw_prior_Oi"
            must be called. This is because the full prior requires a zDM
            grid and an FRB DM, yet this cannot be passed to raw_prior_Oi
            within PATH.
        
        Args:
            mags (tuple, float): host galaxy r-band magnitudes
            ang_sizes (tuple, float): host galaxy angular sizes (arcsec I believe)
            Sigma_ms (tuple, float): galaxy densities on the sky 
        
        Returns:
            Ois (tuple): priors on host galaxy magnitdues mags
        
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
        information internally so that
        path_raw_prior_Oi can be called for numerous
        host galaxy candidates.
        
        It returns the priors distribution.
        
        Args:
            DM [float]: dispersion measure of an FRB (pc cm-3)
            grid (class grid): initialised grid object from which
                                to calculate priors
        
        Returns:
            priors (float): vector of priors on host galaxy apparent magnitude
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
            zlist (np.ndarray, float): array of redshifts
            pz (np.ndarray, float): array of probabilities of the FRB
                            occurring at each of those redshifts
        
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
        Returns posterior redshift distributiuon for a given grid, and DM
        magnitude distribution, for FRBs of DM given a grid object.
        Note: this calculates a prior for PATH, but is a posterior
            from zDM's point of view.
        
        Args:
            grid (class grid object): grid object defining p(z,DM)
            DM (float, np.ndarray OR scalar): FRB DM(s)
        
        Returns:
            papps (np.ndarray, floats): probability distribution of apparent magnitudes given DM
            pz  (np.ndarray, floats): probability distribution of redshift given DM
        """
        # Step 1: get prior on z
        pz = get_pz_prior(grid,DM)
        
        ### STEP 2: get apparent magnitude distribution ###
        if hasattr(DM,"__len__"):
            papps = np.dot(self.maghist,pz)
        else:
            papps = self.maghist*pz
        
        
        return papps,pz

def get_pz_prior(grid, DM):
    """
    Returns posterior redshift distributiuon for a given grid and DM
    magnitude distribution for FRBs of DM given a grid object
    
    Args:
        grid (class grid object): grid object modelling p(z,DM)
        DM (float or np.ndarray of floats): FRB dispersion measure, pc cm^-3
    
    Returns:
        pz (np.ndarray): probability distribution in redshift space
    """
    
    ### get PZ distribution ###
    # Get the coefficients for linear interpolation between DM bins
    # of the grid
    dmvals = grid.dmvals
    ddm = dmvals[1]-dmvals[0]
    # get the grid DM values that this DM site between
    idm1 = (np.floor(DM/ddm)).astype('int')
    idm2 = idm1+1
    dm1 = dmvals[idm1]
    dm2 = dmvals[idm2]
    # get the coefficients of dm1 and dm2
    kdm2 = (DM - dm1)/ddm
    kdm1 = 1.-kdm2
    
    # calculate p(z) based on interpolating grid.rates
    pz = kdm1 * grid.rates[:,idm1] + kdm2 * grid.rates[:,idm2]
    pz = pz/np.sum(pz,axis=0)
    return pz

def SimpleApparentMags(Abs,zs):
    """
    Function to convert galaxy absolue to apparent magnitudes.
    
    Nominally, magnitudes are r-band magnitudes, but this function
    is so simple it doesn't matter.
    
    Just applies a distance correction - no k-correction.
    
    Args:
        Abs (float or array of floats): intrinsic galaxy luminosities 
        zs (float or array of floats): redshifts of galaxies
    
    Returns:
        ApparentMags: NAbs x NZ array of magnitudes, where these
                        are the dimensions of the inputs
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



def simplify_name(TNSname):
    """
    Simplifies an FRB name to basics
    """
    # reduces all FRBs to six integers
    
    if TNSname[0:3] == "FRB":
        TNSname = TNSname[3:]
    
    if len(TNSname) == 9:
        name = TNSname[2:-1]
    elif len(TNSname) == 8:
        name = TNSname[2:]
    elif len(TNSname) == 7:
        name = TNSname[:-1]
    elif len(TNSname) == 6:
        name = TNSname
    else:
        print("Do not know how to process ",TNSname)
    return name

def matchFRB(TNSname,survey):
    """
    Gets the FRB id from the survey list
    Returns None if not in the survey
    Used to match properties between a survey
    and other FRB libraries
    """
    
    name = simplify_name(TNSname)
    match = None
    for i,frb in enumerate(survey.frbs["TNS"]):
        if name == simplify_name(frb):
            match = i
            break
    return match


# this defines the ICS FRBs for which we have PATH info
frblist=['FRB20180924B','FRB20181112A','FRB20190102C','FRB20190608B',
        'FRB20190611B','FRB20190711A','FRB20190714A','FRB20191001A',
        'FRB20191228A','FRB20200430A','FRB20200906A','FRB20210117A',
        'FRB20210320C','FRB20210807D','FRB20211127I','FRB20211203C',
        'FRB20211212A','FRB20220105A','FRB20220501C',
        'FRB20220610A','FRB20220725A','FRB20220918A',
        'FRB20221106A','FRB20230526A','FRB20230708A', 
        'FRB20230731A','FRB20230902A','FRB20231226A','FRB20240201A',
        'FRB20240210A','FRB20240304A','FRB20240310A']



def run_path(name,model,PU=0.1,usemodel = False, sort = False):
    """
    evaluates PATH on an FRB
    
    absolute [bool]: if True, treats rel_error as an absolute value
        in arcseconds
    """
    from frb.frb import FRB
    from astropath.priors import load_std_priors
    from astropath.path import PATH
    
    ######### Loads FRB, and modifes properties #########
    my_frb = FRB.by_name(name)
    
    # do we even still need this? I guess not, but will keep it here just in case
    my_frb.set_ee(my_frb.sig_a,my_frb.sig_b,my_frb.eellipse['theta'],
                my_frb.eellipse['cl'],True)
    
    # reads in galaxy info
    ppath = os.path.join(resources.files('frb'), 'data', 'Galaxies', 'PATH')
    pfile = os.path.join(ppath, f'{my_frb.frb_name}_PATH.csv')
    ptbl = pandas.read_csv(pfile)
    
    # Load prior
    priors = load_std_priors()
    prior = priors['adopted'] # Default
    
    theta_new = dict(method='exp', 
                    max=priors['adopted']['theta']['max'], 
                    scale=0.5)
    prior['theta'] = theta_new
    
    # change this to something depending on the FRB DM
    prior['U']=PU
    
    candidates = ptbl[['ang_size', 'mag', 'ra', 'dec', 'separation']]
    
    this_path = PATH()
    this_path.init_candidates(candidates.ra.values,
                         candidates.dec.values,
                         candidates.ang_size.values,
                         mag=candidates.mag.values)
    this_path.frb = my_frb
    
    frb_eellipse = dict(a=my_frb.sig_a,
                    b=my_frb.sig_b,
                    theta=my_frb.eellipse['theta'])
    
    this_path.init_localization('eellipse', 
                            center_coord=this_path.frb.coord,
                            eellipse=frb_eellipse)
    
    # this results in a prior which is uniform in log space
    # when summed over all galaxies with the same magnitude
    if usemodel:
        this_path.init_cand_prior('user', P_U=prior['U'])
    else:
        this_path.init_cand_prior('inverse', P_U=prior['U'])
        
    # this is for the offset
    this_path.init_theta_prior(prior['theta']['method'], 
                            prior['theta']['max'],
                            prior['theta']['scale'])
    
    P_O=this_path.calc_priors() 
    
    # Calculate p(O_i|x)
    debug = True
    P_Ox,P_U = this_path.calc_posteriors('fixed', 
                         box_hwidth=10., 
                         max_radius=10., 
                         debug=debug)
    mags = candidates['mag']
    
    if sort:
        indices = np.argsort(P_Ox)
        P_O = P_O[indices]
        P_Ox = P_Ox[indices]
        mags = mags[indices]
    
    return P_O,P_Ox,P_U,mags




def plot_frb(name,ralist,declist,plist,opfile):
    """
    does an frb
    
    absolute [bool]: if True, treats rel_error as an absolute value
        in arcseconds
        
    clist: list of astropy coordinates
    plist: list of p(O|x) for candidates hosts
    """
    
    from frb.frb import FRB
    from astropath.priors import load_std_priors
    from astropath.path import PATH
    
    ######### Loads FRB, and modifes properties #########
    my_frb = FRB.by_name(name)
    
    ppath = os.path.join(resources.files('frb'), 'data', 'Galaxies', 'PATH')
    pfile = os.path.join(ppath, f'{my_frb.frb_name}_PATH.csv')
    ptbl = pandas.read_csv(pfile)
    
    candidates = ptbl[['ang_size', 'mag', 'ra', 'dec', 'separation']]
    
    #raoff=199. + 2910/3600 # -139./3600
    #decoff=-18.8 -139./3600 #+2910/3600
    
    raoff = my_frb.coord.ra.deg
    decoff = my_frb.coord.dec.deg
    
    cosfactor = np.cos(my_frb.coord.dec.rad)
    
    plt.figure()
    plt.xlabel('ra [arcsec] - relative')
    plt.ylabel('dec  [arcsec]  - relative')
    
    
    plt.scatter([(ralist-raoff)*3600*cosfactor],[(declist-decoff)*3600],marker='+',
        c=plist, vmin=0.,vmax=1.,label="Deviated FRB")
    
    
    plt.scatter((candidates['ra']-raoff)*3600*cosfactor,(candidates['dec']-decoff)*3600,
            s=candidates['ang_size']*300, facecolors='none', edgecolors='r',
            label="Candidate host galaxies")
    
    
    # orig scatter plot command
    sc = plt.scatter([(my_frb.coord.ra.deg-raoff)*3600*cosfactor],[(my_frb.coord.dec.deg-decoff)*3600],
        marker='x',label="True FRB")
    plt.colorbar(sc)
    
    for i, ra in enumerate(candidates['ra']):
        ra=(ra-raoff)*3600*cosfactor
        dec=(candidates['dec'][i]-decoff)*3600
        plt.text(ra,dec,str(candidates['ang_size'][i])[0:4])
    plt.legend()
    plt.tight_layout()
    plt.savefig(opfile)
    plt.tight_layout()
    plt.close()
