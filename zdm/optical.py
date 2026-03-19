"""
Optical FRB host galaxy models and PATH interface for zdm.

This module connects zdm redshift-DM grids with the PATH
(Probabilistic Association of Transients to their Hosts) algorithm by
providing physically motivated priors on FRB host galaxy apparent
magnitudes, p(m_r | DM_EG).

Architecture
------------
The module is built around a two-layer design:

**Host magnitude models** — each describes the intrinsic absolute
magnitude distribution of FRB host galaxies, p(M_r), and can convert
it to an apparent magnitude distribution p(m_r | z) at a given
redshift. Three models are provided:

- ``simple_host_model``: parametric histogram of p(M_r) with N free
  amplitudes (default 10), interpolated linearly or via spline. An
  optional power-law k-correction can be included. N (or N+1)
  free parameters.

- ``loudas_model``: precomputed p(m_r | z) tables from Nick Loudas,
  constructed by weighting galaxies by stellar mass or star-formation
  rate. Interpolated between tabulated redshift bins with a
  luminosity-distance shift. Single free parameter ``fSFR`` sets the
  SFR/mass mixing fraction.

- ``marnoch_model``: zero-parameter model. Fits a Gaussian to the
  r-band magnitude distribution of known CRAFT ICS host galaxies from
  Marnoch et al. 2023 (MNRAS 525, 994), using cubic splines for the
  redshift-dependent mean and standard deviation.

**``model_wrapper``** — a survey-independent wrapper around any host
model. Given a zdm ``Grid`` and an observed DM_EG, it convolves
p(m_r | z) with the zdm p(z | DM_EG) posterior to produce a
DM-informed apparent magnitude prior for PATH. It also estimates
P_U, the prior probability that the true host is below the survey
detection limit.

Typical usage
-------------
::

    model = opt.marnoch_model()
    wrapper = opt.model_wrapper(model, grid.zvals)
    wrapper.init_path_raw_prior_Oi(DMEG, grid)   # DM-specific initialisation
    PU = wrapper.estimate_unseen_prior()
    # pathpriors.USR_raw_prior_Oi is now set automatically by init_path_raw_prior_Oi

Module-level data
-----------------
``frblist`` : list of str
    TNS names of CRAFT ICS FRBs for which PATH optical data are
    available (used by the scripts in ``zdm/scripts/Path/``).
"""


import numpy as np
from matplotlib import pyplot as plt
from zdm import cosmology as cos
from zdm import optical_params as op
from scipy.interpolate import CubicSpline
from scipy.interpolate import make_interp_spline
from scipy.stats import norm
import os
from importlib import resources
import pandas
import h5py

import astropath.priors as pathpriors


###################################################################
############ Routines associated with Nick's model ################
###################################################################


class marnoch_model:
    """
    Class initiates a model based on Lachlan Marnoch's predictions
    for FRB host galaxy visibility in
    https://ui.adsabs.harvard.edu/abs/2023MNRAS.525..994M/abstract
    Here, we assume that host galaxy magnitudes have a normal
    distribution, with mean and standard deviation given by
    L. Marnoch's data.
    """
    
    def __init__(self,OpticalState=None):
        """
        Initialises the model. There are no variables here.
        
        Args:
            OpticalState: allows the model to refer to an optical state.
                    However, the model is independent of that state.
        """
        
        # uses the "simple hosts" descriptor
        if OpticalState is None:
            OpticalState = op.OpticalState()
        self.OpticalState = OpticalState
        self.opstate = None
        
        # loads the dataset
        self.load_data()
        
        # extracts subic splines for mean and std dev
        self.process_rbands()
    
    
    def load_data(self):
        """
        Loads the Marnoch et al data on r-band magnitudes from FRB hosts
        """
        from astropy.table import Table
        datafile="magnitudes_and_probabilities_vlt-fors2_R-SPECIAL.ecsv"
        infile =  os.path.join(resources.files('zdm'), 'data', 'optical', datafile)
        table = Table.read(infile, format='ascii.ecsv')
        self.table = table

    def process_rbands(self):
        """
        Build cubic spline fits to the mean and rms of p(m_r) as a function of z.

        Reads the per-FRB r-band magnitude columns from ``self.table``,
        computes their mean (``Rbar``) and sample standard deviation (``Rrms``)
        across all FRBs at each tabulated redshift, then fits two cubic splines:

        - ``self.sbar``: CubicSpline interpolating the mean apparent magnitude
          as a function of redshift.
        - ``self.srms``: CubicSpline interpolating the rms scatter as a
          function of redshift.

        These splines are subsequently used by ``get_pmr_gz`` to evaluate
        the Gaussian p(m_r | z) at arbitrary redshifts.
        """
        
        table = self.table
        colnames = table.colnames
        # gets FRBs
        frblist=[]
        for name in colnames:
            if name[0:3]=="FRB":
                frblist.append(name)
        zlist = table["z"]
        nz = zlist.size
        nfrb = len(frblist)
        Rmags = np.zeros([nfrb,nz])

        for i,frb in enumerate(frblist):

            Rmags[i,:] = table[frb]

        # gets mean and rms
        Rbar = np.average(Rmags,axis=0)
        Rrms = (np.sum((Rmags - Rbar)**2,axis=0)/(nfrb-1))**0.5
        
        # creates cubic spline fits to mean and rms of m_r as a function of z
        self.sbar = CubicSpline(zlist,Rbar)
        self.srms = CubicSpline(zlist,Rrms)
        
        #return Rbar,Rrms,zlist,sbar,srms
    
    def get_pmr_gz(self,mrbins,z):
        """
        Return the apparent magnitude probability distribution p(m_r | z).

        Evaluates a Gaussian distribution whose mean and standard deviation
        are obtained from the cubic splines fit in ``process_rbands``,
        and integrates it over the provided magnitude bins.

        This model has no free parameters; the Gaussian moments are fully
        determined by the Marnoch et al. 2023 host galaxy data.

        Parameters
        ----------
        mrbins : array-like of float, length N+1
            Edges of the apparent magnitude bins over which to compute the
            probability. The output has length N (one value per bin).
        z : float
            Redshift at which to evaluate the magnitude distribution.

        Returns
        -------
        pmr : np.ndarray, length N
            Probability in each magnitude bin (sums to ≤ 1; may be less
            than unity if the Gaussian extends beyond the bin range).
        """

        mean = self.sbar(z)
        rms = self.srms(z)
        
        deviates = (mrbins-mean)/rms
        cprobs = norm.cdf(deviates)
        pmr = cprobs[1:] - cprobs[:-1]
        
        return pmr
    
    
    
class loudas_model:
    """
    This class initiates a model based on Nick Loudas's model of
    galaxy magnitudes as a function of redshift. The underlying
    model is a description of galaxies as a function of
    stellar mass and star-formation rate as a function of redshift.
    
    """
    
    def __init__(self,OpticalState=None,fname='p_mr_distributions_dz0.01_z_in_0_1.2.h5',data_dir=None,verbose=False):
        """
        Initialise the Loudas model, loading precomputed p(m_r | z) tables.

        Args:
            OpticalState (OpticalState, optional): optical parameter state. A
                default ``OpticalState`` is created if not provided.
            fname (str): HDF5 filename containing the Loudas p(m_r | z) tables.
                Defaults to ``'p_mr_distributions_dz0.01_z_in_0_1.2.h5'``.
            data_dir (str, optional): directory containing ``fname``. Defaults
                to the package data directory ``zdm/data/optical/``.
            verbose (bool): if True, print progress messages. Defaults to False.
        """
        
        # uses the "simple hosts" descriptor
        if OpticalState is None:
            OpticalState = op.OpticalState() 
        self.OpticalState = OpticalState
        
        #extract the correct optical substate from the opstate
        self.opstate = self.OpticalState.loudas
        
        self.fsfr = self.opstate.fSFR
        
        
        # checks that cosmology is initialised
        if not cos.INIT:
            cos.init_dist_measures()
        
        # gets base input directory. In future, this may be expanded
        if data_dir is None:
            data_dir = os.path.join(resources.files('zdm'), 'data', 'optical')
        
        # load data and its properties
        self.init_pmr(fname,data_dir)
        
        # initialises cubic splines for faster speedups
        self.init_cubics()
        
    def init_pmr(self,fname,data_dir):
        """
        Loads p(mr|z) distributions from Nick Loudas. Note - these are
        actually distributions in apparent magnitude mr.
        
        Mostly, this wraps around Nick's code "load_p_mr_distributions".
        I've kept them separate to distinguish between his code and mine -CWJ.
        
        """
        ####### loading p(mr) distributions ##########
        zbins, rmag_centres, p_mr_sfr, p_mr_mass = self.load_p_mr_distributions(
                                        data_dir, fname = fname)
        
        # zbins represent ranges. We also calculate z-bin centres
        self.drmag = rmag_centres[1] - rmag_centres[0]
        self.zbins = zbins
        self.nzbins = zbins.size-1
        self.czbins = 0.5*(self.zbins[1:] + self.zbins[:-1])
        self.logzbins = np.log10(zbins)
        self.clogzbins = 0.5*(self.logzbins[1:] + self.logzbins[:-1])
        self.rmags = rmag_centres # centres of rmag bins
        self.p_mr_sfr = p_mr_sfr # sfr-weighted p_mr
        self.p_mr_mass = p_mr_mass # mass-weighted p_mr
        
        
        # we have now all the data we need!
    
    def init_cubics(self):
        """
        initialises cubic splines that interpolate in mr. For later use (speedup!)
        """
        
        sfr_splines = []
        mass_splines = []
        for i in np.arange(self.nzbins):
            sfr_spline = make_interp_spline(self.rmags,self.p_mr_sfr[i],k=1)
            sfr_splines.append(sfr_spline)
            
            mass_spline = make_interp_spline(self.rmags,self.p_mr_mass[i],k=1)
            mass_splines.append(mass_spline)
        
        self.mass_splines = mass_splines
        self.sfr_splines = sfr_splines
    
    def get_pmr_gz(self,mrbins,z):
        """
        Return the apparent magnitude probability distribution p(m_r | z).

        Interpolates between the two nearest tabulated redshift bins (in
        log-z space), applying a luminosity-distance shift to each tabulated
        p(m_r) before combining them. The mass- and SFR-weighted distributions
        are mixed according to ``self.fsfr`` (set via ``init_args``).

        The result is normalised so that the bin probabilities sum to unity
        over the full magnitude range, provided the distribution does not
        extend significantly beyond ``mrbins``.

        Parameters
        ----------
        mrbins : array-like of float, length N+1
            Edges of the apparent magnitude bins. Output has length N.
        z : float
            Redshift at which to evaluate the distribution. Values outside
            the tabulated range are extrapolated from the nearest edge bin.

        Returns
        -------
        pmr : np.ndarray, length N
            Probability in each apparent magnitude bin (sums to ≤ 1).
        """
        
        fsfr = self.fsfr
        
        # gets interpolation coefficients
        lz = np.log10(z)
        if lz < self.clogzbins[0]:
            # sets values equal to that of smallest bin, to avoid interpolation
            i1=0
            i2=1
            k1=1.
            k2=0.
        elif lz > self.clogzbins[-1]:
            i1=self.nzbins-2
            i2=self.nzbins-1
            k1=0
            k2=1.
        else:
            i1 = np.where(lz > self.clogzbins)[0][-1] # gets lowest value where zs are larger
            i2=i1+1
            k2 = (lz-self.clogzbins[i1])/(self.clogzbins[i2]-self.clogzbins[i1])
            k1 = 1.-k2
        
        z1=self.czbins[i1]
        z2=self.czbins[i2]
        
        # the mr distributions are apparent magnitudes
        # hence, we have to interpolate between z-bins using first-order shifting
        # this is *very* important for low values of z
        
        DL = cos.dl(z)
        DL1 = cos.dl(z1)
        DL2 = cos.dl(z2)
        
        # calculates shifts in logarithm. Still shifts when z is lower or higher than m_r
        # note: a factor of 2 in DL means a factor of 4 in luminosity, meaning
        # 5/2 log10(4) in mr = 5 log10(2).
        dmr1 = 5.*np.log10(DL/DL1) # will be a positive shift
        dmr2 = 5.*np.log10(DL/DL2) # will be a negative shift
        
        
        mr_centres = (mrbins[:-1]+mrbins[1:])/2.
        
        # will interpolate the values at *lower* magnitudes, effectively shifting distribution up
        p_mr_mass1 = self.mass_splines[i1](mr_centres - dmr1)
        
        # will interpolate the values at *higher* magnitudes, effectively shifting distribution down
        p_mr_mass2 = self.mass_splines[i2](mr_centres - dmr2)
        
        
        # will interpolate the values at *lower* magnitudes, effectively shifting distribution up
        p_mr_sfr1 = self.sfr_splines[i1](mr_centres - dmr1)
        # will interpolate the values at *higher* magnitudes, effectively shifting distribution down
        p_mr_sfr2 = self.sfr_splines[i2](mr_centres - dmr2)
        
        # distribution for that redshift assuming mass weighting
        pmass = k1*p_mr_mass1 + k2*p_mr_mass2
        
        # just left here for testing purposes
        if False:
            print("Redshift bins are ",z,z1,z2)
            print("Luminosity distances are ",DL,DL1,DL2)
            print("shifts are therefore ",dmr1,dmr2)
            
            # generate an example plot showing interpolation
            plt.plot(self.rmags,p_mr_mass1,linestyle="-",label="scaled from z0")
            plt.plot(self.rmags,p_mr_mass2,linestyle="--",label="scaled from z1")
            plt.plot(self.rmags,self.p_mr_mass[i1],linestyle=":",label="z0")
            plt.plot(self.rmags,self.p_mr_mass[i2],linestyle=":",label="z1")
            plt.legend()
            plt.tight_layout()
            plt.show()
            exit()
        
        # distribution for that redshift assuming sfr weighting
        psfr = k1*p_mr_sfr1 + k2*p_mr_sfr2
        
        # mean weighted distribution
        pmr = pmass*(1.-fsfr) + psfr*fsfr
        
        # normalise by relative bin width - recall, bins should sum to unity
        pmr *= (mrbins[1]-mrbins[0])/self.drmag
        
        # remove negative probabilities - set to zero, and re-normalise
        prevsum = np.sum(pmr)
        bad = np.where(pmr < 0.)[0]
        pmr[bad] = 0.
        newsum = np.sum(pmr)
        pmr *= prevsum / newsum
        
        return pmr
        
    def load_p_mr_distributions(self,data_dir,fname: str = 'p_mr_distributions_dz0.01_z_in_0_1.2.h5') -> tuple:
        """
        This code originally written by Nick Loudas. Used with permission
        
        Load the p(mr|z) distributions from an HDF5 file.
        Args:
            fname (str): Input filename.
            output_dir (str): Directory where the file is stored. Optional (otherwise defaults as below)
        Returns:
            zbins (np.array): Redshift bin edges.
            rmag_centers (np.array): Centers of r-band magnitude bins.
            p_mr_sfr (np.array): p(mr|z) for SFR-weighted population. Shape: (len(zbins) - 1,
                    rmag_resolution). rmag_resolution(=len(rmag_centers)) is fixed across redshift bins.
            p_mr_mass (np.array): p(mr|z) for Mass-weighted population. Shape: (len(zbins) - 1,
                    rmag_resolution). rmag_resolution(=len(rmag_centers)) is fixed across redshift bins.
        Note:
            The PDF in m_r within a given redshift bin [z1,z2] has been computed at the right edge of the bin (z = z2).
        """
        infile = os.path.join(data_dir,fname)
        with h5py.File(infile, 'r') as hf:
            zbins = np.array(hf['zbins'])
            zbins = zbins[1:] # first bin is "extra" for "reasons"
            rmag_centers = np.array(hf['rmag_centers'])
            p_mr_sfr = np.array(hf['p_mr_sfr'])
            p_mr_mass = np.array(hf['p_mr_mass'])
            
            # normalise these probabilities such that the bins sum to unity
            p_mr_sfr = (p_mr_sfr.T / np.sum(p_mr_sfr,axis=1)).T
            p_mr_mass = (p_mr_mass.T / np.sum(p_mr_mass,axis=1)).T
        
        print(f"p(mr|z) distributions loaded successfully from 'p_mr_dists/{fname}'")
        n_redshift_bins = len(zbins) - 1
        return zbins, rmag_centers, p_mr_sfr, p_mr_mass
    
    def give_p_mr_mass(self,z: float):
        """
        Return p(m_r | z) for the stellar-mass-weighted host population.

        Uses a nearest-bin lookup (no interpolation) in the tabulated redshift
        grid. For interpolated results with luminosity-distance shifting, use
        ``get_pmr_gz`` with ``fSFR=0`` instead.

        Args:
            z (float): Redshift value.

        Returns:
            np.ndarray: p(m_r | z) values at the tabulated r-band magnitude
                centres (``self.rmags``), normalised to sum to unity.
        """
        # Find the appropriate redshift bin index
        idx = np.clip(np.searchsorted(self.zbins, z) - 1, 0,  n_redshift_bins - 1)
        return self.p_mr_mass[idx]
    
    def give_p_mr_sfr(self,z: float):
        """
        Return p(m_r | z) for the star-formation-rate-weighted host population.

        Uses a nearest-bin lookup (no interpolation) in the tabulated redshift
        grid. For interpolated results with luminosity-distance shifting, use
        ``get_pmr_gz`` with ``fSFR=1`` instead.

        Args:
            z (float): Redshift value.

        Returns:
            np.ndarray: p(m_r | z) values at the tabulated r-band magnitude
                centres (``self.rmags``), normalised to sum to unity.
        """
        # Find the appropriate redshift bin index
        idx = np.clip(np.searchsorted(self.zbins, z) - 1, 0,  n_redshift_bins - 1)
        return self.p_mr_sfr[idx]
    
    def init_args(self,fSFR):
        """
        Set the SFR/mass mixing fraction for the Loudas model.

        Args:
            fSFR (float or array-like of length 1): fraction of FRB hosts
                that trace star-formation rate. ``fSFR=0`` gives a purely
                mass-weighted population; ``fSFR=1`` gives a purely
                SFR-weighted population. Intermediate values linearly mix
                the two. If an array is passed, only the first element is used.
        """
        # for numerical purposes, fSFR may have to be a vector
        if hasattr(fSFR,'__len__'):
            fSFR = fSFR[0]
        self.fsfr = fSFR
    
    def init_priors(self,zlist):
        """
        Generates magniude prior distributions for a list of redshifts
        This allows faster interpolation later.
        
        Currently, this is not used!
        """
        print("WARNING: redundant init priors!!!!!!")
        exit()
        mass_priors = np.zeros([zlist.size,self.nmr])
        sfr_priors = np.zeros([zlist.size,self.nmr])
        for i,z in enumerate(zlist):
            mass_priors[i,:] = self.get_p_mr(z,0.)
            sfr_priors[i,:] = self.get_p_mr(z,1.)
        self.mass_priors = mass_priors
        self.sfr_priors = sfr_priors

class simple_host_model:
    """
    A class to hold information about the intrinsic properties of FRB
    host galaxies. This is a simple but generic model.
    
    Ingredients are:
        A model for describing the intrinsic distribution
            of host galaxies. This model must be described 
            by some set of parameters, and be able to return a
            prior as a function of intrinsic host galaxy magnitude.
            This model is initialised via opstate.AbsModelID.
            Here, it is just 10 parameters at different absolute
            magnitudes, with linear/spline interpolation
            
        A model for converting absolute to apparent host magnitudes.
            This is by defult an apparent r-band magnitude, though
            in theory a user can work in whatever band they wish.
            
    Internally, this class initialises:
        An array of absolute magnitudes, which get weighted according
        to the host model. Internal variables associated with this
        are prefaced "Model"
        
    Note that while this class describes the intrinsic "magnitudes",
    really magnitude here is a proxy for whatever parameter is used
    to intrinsically describe FRBs. However, only 1D descriptions are
    currently implemented. Future descriptions will include redshift
    evolution, and 2D descriptions (e.g. mass, SFR) at any given redshift.
    
    """
    def __init__(self,OpticalState=None,verbose=False):
        """
        Initialise the simple host magnitude model.

        Args:
            OpticalState (OpticalState, optional): optical parameter state
                providing model configuration (magnitude ranges, number of
                bins, interpolation scheme, k-correction flag). A default
                ``OpticalState`` is created if not provided.
            verbose (bool, optional): if True, print which sub-models are
                being initialised. Defaults to False.
        """
        # uses the "simple hosts" descriptor
        if OpticalState is None:
            self.OpticalState = op.OpticalState()
        else:
            self.OpticalState = OpticalState
        self.opstate = self.OpticalState.simple
        
        # checks that cosmology is initialised
        if not cos.INIT:
            cos.init_dist_measures()
        
        if self.opstate.AppModelID == 0:
            if verbose:
                print("Initialising simple luminosity function")
            # must take arguments of (absoluteMag,k,z)
            self.CalcApparentMags = SimpleApparentMags
            self.CalcAbsoluteMags = SimpleAbsoluteMags
        elif self.opstate.AppModelID == 1:
            if verbose:
                print("Initialising k-corrected luminosity function")
            # must take arguments of (absoluteMag,k,z)
            self.CalcApparentMags = SimplekApparentMags
            self.CalcAbsoluteMags = SimplekAbsoluteMags
        else:
            raise ValueError("Model ",self.opstate.AppModelID," not implemented")
        
        if self.opstate.AbsModelID == 0:
            if verbose:
                print("Describing absolute mags with N independent bins")
        elif self.opstate.AbsModelID == 1:
            if verbose:
                print("Describing absolute mags with linear interpoilation of N points")
        elif self.opstate.AbsModelID == 2:
            if verbose:
                print("Describing absolute mags with spline interpoilation of N points")
        elif self.opstate.AbsModelID == 3:
            if verbose:
                print("Describing absolute mags with spline interpoilation of N log points")
        else:
            raise ValueError("Model ",self.opstate.AbsModelID," not implemented")
        
        
        self.AppModelID = self.opstate.AppModelID
        self.AbsModelID = self.opstate.AbsModelID
        
        
        
        self.init_abs_bins()
        self.init_model_bins()
        
        # could perhaps use init args for this?
        if self.opstate.AbsPriorMeth==0:
            # uniform prior in log space of absolute magnitude
            AbsPrior = np.full([self.ModelNBins],1./self.NAbsBins)
        else:
            # other methods to be added as required
            raise ValueError("Luminosity prior method ",self.opstate.AbsPriorMeth," not implemented")
        
        # enforces normalisation of the prior to unity
        self.AbsPrior = AbsPrior/np.sum(AbsPrior)
        
        # k-correction
        self.k = self.opstate.k
        
        # this maps the weights from the parameter file to the absoluate magnitudes use
        # internally within the program. We now initialise this during an "init"
        self.init_abs_mag_weights()
        
        # the below is done for the wrapper function
        #self.ZMAP = False # records that we need to initialise this
    
    def get_args(self):
        """
        function to return args as a vector in the form of init_args
        """
        
        if self.opstate.AppModelID == 0:
            args = self.AbsPrior
        elif self.opstate.AppModelID == 1:
            args = np.zeros([self.ModelNBins+1])
            args[1:] = self.AbsPrior
            args[0] = self.k
        return args
    
    def init_abs_bins(self):
        """
        Initialises internal array of absolute magnitudes
        This is a simple set of uniformly log-spaced bins in terms
        of absolute magnitude, which the absolute magnitude model gets
        projected onto.
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
        
    
    def init_args(self,Args):
        """
        Initialises prior on absolute magnitude of galaxies according to the method.
        
        Args:
            - Args (list of floats): The prior on absolute magnitudes
                        to set for this model
                        IF AppModelID == 1, then interpret the first as the k-correction
        
        """
        # Eventually, incorporate the AbsPrior vector into SimpleParams
        #self.opstate = OpticalState.SimpleParams
        
        if self.opstate.AppModelID == 0:
            AbsPrior=Args
        elif self.opstate.AppModelID == 1:
            AbsPrior=Args[1:]
            self.k = Args[0]
        
        # enforces normalisation of the prior to unity
        self.AbsPrior = AbsPrior/np.sum(AbsPrior)
        
        # this maps the weights from the parameter file to the absoluate magnitudes use
        # internally within the program. We now initialise this during an "init"
        self.init_abs_mag_weights()
    
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
            
        else:
            # bins on edges
            ModelBins = np.linspace(self.Absmin,self.Absmax,ModelNBins)
        
        self.ModelBins = ModelBins
        self.dModel = ModelBins[1]-ModelBins[0]
    
    def get_pmr_gz(self,mrbins,z):
        """
        Return the apparent magnitude probability distribution p(m_r | z).

        Converts each apparent magnitude bin centre back to an absolute
        magnitude using ``CalcAbsoluteMags``, then linearly interpolates
        the absolute magnitude weight array (``self.AbsMagWeights``) to
        obtain a probability density at each bin.

        Parameters
        ----------
        mrbins : np.ndarray of float, length N+1
            Edges of the apparent magnitude bins. The output has length N,
            with one probability value per bin centre.
        z : float
            Redshift at which to evaluate the distribution.

        Returns
        -------
        pmr : np.ndarray, length N
            Probability density at each apparent magnitude bin centre.
            Values at the edges of the absolute magnitude range are
            clamped to the nearest valid bin rather than extrapolated.

        Notes
        -----
        The returned values are NOT renormalised to sum to unity; the sum
        may be less than one if some absolute magnitudes lie outside the
        model range ``[Absmin, Absmax]``. This is intentional: the
        shortfall represents probability mass for hosts too faint or too
        bright to appear in the apparent magnitude range.
        """
        
        old = False
        if old:
            # mapping of apparent to absolute magnitude
            if self.opstate.AppModelID == 0:
                mrvals = self.CalcApparentMags(self.AbsMags,z) # works with scalar z
            elif self.opstate.AppModelID == 1:
                mrvals = self.CalcApparentMags(self.AbsMags,self.k,z) # works with scalar z
            
            # creates weighted histogram of apparent magnitudes,
            # using model weights from wmap (which are fixed for all z)
            hist,bins = np.histogram(mrvals,weights=self.AbsMagWeights,bins=mrbins)
            
            #smoothing function - just to flatten the params
            NS=10
            smoothf = self.gauss(mrvals[0:NS] - np.average(mrvals[0:NS]))
            smoothf /= np.sum(smoothf)
            smoothed = np.convolve(hist,smoothf,mode="same")
            
            #smoothed=hist. Not sure yet if smoothing is the right thing to do!
            pmr = smoothed
        else:
            # probability density at bin centre
            mrbars = (mrbins[:-1] + mrbins[1:])/2.
            
            # get absolute magnitudes corresponding to these apparent magnitudes
            if self.opstate.AppModelID == 0:
                Mrvals = self.CalcAbsoluteMags(mrbars,z) # works with scalar z
            elif self.opstate.AppModelID == 1:
                Mrvals = self.CalcAbsoluteMags(mrbars,self.k,z) # works with scalar z
            
            # linear interpolation
            # note that dMr = dmr, so we just map probability densities
            
            
            kmag2s = (Mrvals - self.Absmin)/self.dMag
            imag1s = np.floor(kmag2s).astype('int')
            kmag2s -= imag1s
            kmag1s = 1.-kmag2s
            imag2s = imag1s+1
            
            # guards against things that are too low
            toolow = np.where(imag1s < 0)[0]
            imag1s[toolow]=0
            kmag1s[toolow]=1
            imag2s[toolow]=1
            kmag2s[toolow]=0
            
            # guards against things that are too high
            toohigh = np.where(imag2s >= self.NAbsBins)[0]
            imag1s[toohigh]=self.NAbsBins-2
            kmag1s[toohigh]=0
            imag2s[toohigh]=self.NAbsBins-1
            kmag2s[toohigh]=1.
            
            pmr = self.AbsMagWeights[imag1s] * kmag1s + self.AbsMagWeights[imag2s] * kmag2s
            
        # # NOTE: these should NOT be re-normalised, since the normalisation reflects
        # true magnitudes which fall off the apparent magnitude histogram.
        return pmr
    
    def gauss(self,x,mu=0,sigma=0.1):
        """
        simple Gaussian smoothing function
        """
        return np.exp(-0.5*(x-mu)**2/sigma**2)
    
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
            # linear interpolation
            # gives mapping from model bins to mag bins
            weights = np.interp(self.AbsMags,self.ModelBins,self.AbsPrior)
            
        elif self.AbsModelID == 2:
            # As above, but with spline interpolation of model.
            # coefficients span full range
            cs = CubicSpline(self.ModelBins,self.AbsPrior)
            weights = cs(self.AbsMags)
            # ensures no negatives
            toolow = np.where(weights < 0.)
            weights[toolow] = 0.
            # ensures that if everything is zero above/below a point, so is the interpolation
            iFirstNonzero = np.where(self.AbsPrior > 0.)[0][0]
            if iFirstNonzero > 0:
                toolow = np.where(self.AbsMags < self.ModelBins[iFirstNonzero -1])
                weights[toolow] = 0.
            iLastNonzero = np.where(self.AbsPrior > 0.)[0][-1]
            if iLastNonzero < self.AbsPrior.size - 1:
                toohigh = np.where(self.AbsMags > self.ModelBins[iLastNonzero+1])
                weights[toohigh] = 0.
        elif self.AbsModelID == 3:
            # As above, but splines interpolate in *log* space
            cs = CubicSpline(self.ModelBins,self.AbsPrior)
            weights = cs(self.AbsMags)
            weights = 10**weights
            
        else:
            raise ValueError("This weighting scheme not yet implemented")
        
        
        # renormalises the weights, so all internal apparent mags sum to unit
        self.AbsMagWeights = weights / np.sum(weights)
        
        return 
        

class model_wrapper:
    """
    Generic functions applicable to all models.
    
    The program flow is to initialise with a host model ("model"),
    then given arrays of Mr and zvalues, pre-calculate an array
    of p(Mr|z), and then for individual host galaxies with a
    p(z|DM) distribution, be able to return priors for PATH.
    
    Internally, the code uses an array of apparent magnitudes,
    which is used to compare with host galaxy candidates.
    Internal variables associated with this are prefaced "App"
        
    The Arrays mapping intrinsic to absolute magnitude as a function
        of redshift, to allow quick estimation of p(apparent_mag | DM)
        for a given FRB survey with many FRBs
        Internal variables associated with this are prefaced "Abs"
    
    
    The workflow is:
        -init with a model class and array of z values. This sets 
            absolute magnitude bins. The z values should correspond
            to those from a grid object.
            Initialisation primarily calls p(Mr|z) repeatedly for all internal
            Mr and z values, to allow fast evaluation in the future
        - set up PATH functions to point to this array:
            pathpriors.USR_raw_prior_Oi = wrapper.path_raw_prior_Oi
        - initialise this class for a given init_path_raw_prior_Oi(DM,grid).
            This calculates magnitude priors given p(z|DM) (grid)
            and p(mr|z) (host model).
       
    """
    def __init__(self,model,zvals):
        """
        Initialises model wrapper.
        
        
        Args:
            model (class object): Model is one of the host model class objects
                    that can calculate p(Mr|z)
            zvals (np.array): redshift values corresponding to grid object
            opstate (class optical): state containing optical info
            
        """
        
        # higher level state defining optical parameters
        self.OpticalState = model.OpticalState
        
        #parameters defining chance of identifying a galaxy with magnitude m
        self.pU_mean = self.OpticalState.id.pU_mean
        self.pU_width = self.OpticalState.id.pU_width
        
        # specific substate of the model
        self.opstate = model.opstate
        
        self.model = model # checks the model has required attributes
        
        
        # initialise bins in apparent magnitude
        self.init_app_bins()
        
        self.init_zmapping(zvals)
        
        
    def init_app_bins(self):
        """
        Initialises bins in apparent magnitude
        It uses these to calculate priors for any given host galaxy magnitude.
        This is a very simple set of uniformly log-spaced bins in magnitude space,
        and linear interpolation is used between them.
        """
        
        
        self.Appmin = self.OpticalState.app.Appmin
        self.Appmax = self.OpticalState.app.Appmax
        self.NAppBins = self.OpticalState.app.NAppBins
        
        # this creates the bin edges
        self.AppBins = np.linspace(self.Appmin,self.Appmax,self.NAppBins+1)
        dAppBin = self.AppBins[1] - self.AppBins[0]
        self.AppMags = self.AppBins[:-1] + dAppBin/2.
        self.dAppmag = dAppBin
    
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
        
        self.zvals=zvals
        
        # we aim to produce a grid of p(z,m_r) for rapid convolution 
        # with a p(z) array
        self.nz = zvals.size
        
        p_mr_z = np.zeros([self.NAppBins,self.nz])
        
        for i,z in enumerate(zvals):
            # use the model to calculate p(mr|z) for range of z-values
            # this is then stored in an array.
            # NOTE! This could become un-normalised due to
            # interpolation falling off the edge
            # hence, we normalise it
            this_p_mr_z = self.model.get_pmr_gz(self.AppBins,z)
            this_p_mr_z /= np.sum(this_p_mr_z)
            p_mr_z[:,i] = this_p_mr_z
        self.p_mr_z = p_mr_z
        
        # records that this has been initialised
        self.ZMAP = True
     
    #############################################################
    ##################    Path Calculations    #################
    #############################################################
    
    
    def init_path_raw_prior_Oi(self,DM,grid):
        """
        Initialise the apparent magnitude prior for a single FRB DM.

        Computes p(m_r | DM_EG) by convolving the precomputed p(m_r | z)
        grid (``self.p_mr_z``) with the zdm posterior p(z | DM_EG) extracted
        from ``grid``. The result is stored internally so that
        ``path_raw_prior_Oi`` can be called repeatedly for different host
        galaxy candidates belonging to the same FRB without recomputing the
        DM integral.

        Also computes the probability that the host is undetected:
        - ``self.priors``: p(m_r | DM) weighted by the detection probability
          p(detected | m_r).
        - ``self.PUdist``: the magnitude-resolved contribution to P_U.
        - ``self.PU``: scalar total prior probability that the host is unseen,
          returned by ``estimate_unseen_prior()``.

        After this call, ``pathpriors.USR_raw_prior_Oi`` is automatically
        pointed at ``self.path_raw_prior_Oi``.

        Args:
            DM (float): extragalactic dispersion measure of the FRB (pc cm⁻³).
            grid (Grid): initialised zdm grid object providing p(z, DM).
        """
        
        # we start by getting the posterior distribution p(z)
        # for an FRB with DM DM seen by the 'grid'
        pz = get_pz_prior(grid,DM)
        
        # checks that pz is normalised
        pz /= np.sum(pz)
        
        priors = np.sum(self.p_mr_z * pz,axis=1) # sums over z
        
        # stores knowledge of the DM used to calculate the priors
        self.prior_DM = DM
        self.raw_priors = priors
        
        pU = pUgm(self.AppMags,self.pU_mean,self.pU_width)
        
        self.priors = self.raw_priors * (1.-pU)
        self.PUdist = self.raw_priors * pU
        self.PU = np.sum(self.PUdist)
        
        # sets the PATH user function to point to its own
        pathpriors.USR_raw_prior_Oi = self.path_raw_prior_Oi
        
        #return priors
    
    def get_posterior(self, grid, DM):
        """
        Return apparent magnitude and redshift posteriors for a given DM.

        Computes p(z | DM) from the grid and convolves it with the
        precomputed ``self.maghist`` to obtain p(m_r | DM).

        Note: from PATH's perspective this is a prior on host magnitude,
        but from zdm's perspective it is a posterior on redshift given DM.

        This method predates ``init_path_raw_prior_Oi`` and may not be
        actively used in current scripts.

        Args:
            grid (Grid): initialised zdm grid object providing p(z, DM).
            DM (float or np.ndarray): FRB extragalactic DM(s) in pc cm⁻³.

        Returns:
            papps (np.ndarray): probability distribution of apparent magnitude
                given DM, p(m_r | DM).
            pz (np.ndarray): probability distribution of redshift given DM,
                p(z | DM).
        """
        # Step 1: get prior on z
        pz = get_pz_prior(grid,DM)
        
        ### STEP 2: get apparent magnitude distribution ###
        if hasattr(DM,"__len__"):
            papps = np.dot(self.maghist,pz)
        else:
            papps = self.maghist*pz
        
        
        return papps,pz    
    
    
    def estimate_unseen_prior(self):
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
        
        # smooth cutoff
        #pU_g_mr = pogm(self.AppMags,mean,width)
        
        # simple hard cutoff - now redundant
        #invisible = np.where(self.AppMags > mag_limit)[0]
        
        #PU = np.sum(pU_g_mr * self.priors)
        
        #PU = np.sum(self.priors[invisible])
        
        # we now pre-calculate this at the init raw path prior stage
        #PU = np.sum(self.PU)
        
        return self.PU
    
    def path_base_prior(self,mags):
        """
        Evaluate the apparent magnitude prior p(m_r | DM) at a list of magnitudes.

        Linearly interpolates ``self.priors`` (which already incorporates the
        detection probability p(detected | m_r)) at each requested magnitude,
        converting from the internally normalised sum-to-unity convention to a
        probability density by dividing by the bin width ``self.dAppmag``.

        Unlike ``path_raw_prior_Oi``, this method does NOT divide by the
        galaxy surface density Sigma_m, so it returns the raw magnitude prior
        without the PATH normalisation factor.

        Args:
            mags (list or tuple of float): apparent r-band magnitudes of
                candidate host galaxies at which to evaluate the prior.

        Returns:
            Ois (np.ndarray): prior probability density p(m_r | DM) evaluated
                at each magnitude in ``mags``.
        """
        ngals = len(mags)
        Ois = []
        for i,mag in enumerate(mags):
            
            #print(mag)
            # calculate the bins in apparent magnitude prior
            kmag2 = (mag - self.Appmin)/self.dAppmag
            imag1 = int(np.floor(kmag2))
            imag2 = imag1 + 1
            kmag2 -= imag1 #residual; float
            kmag1 = 1.-kmag2
            
            # careful with interpolation - priors are for magnitude bins
            # with bin edges give by Appmin + N dAppmag.
            # We probably want to smooth this eventually due to minor
            # numerical tweaks
            
            #kmag2 -= imag1
            #kmag1 = 1.-kmag2
            #imag2 = imag1+1
            #prior = kmag1*self.priors[imag1] + kmag2*self.priors[imag2]
            
            # simple linear interpolation
            Oi = self.priors[imag1] * kmag1 + self.priors[imag2] * kmag2
            
            # correct normalisation - otherwise, priors are defined to sum
            # such that \sum priors = 1; here, we need \int priors dm = 1
            Oi /= self.dAppmag 
            
            Ois.append(Oi)
        
        Ois = np.array(Ois)
        return Ois
        
        
    
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
            imag2 = imag1 + 1
            kmag2 -= imag1 #residual; float
            kmag1 = 1.-kmag2
            
            # careful with interpolation - priors are for magnitude bins
            # with bin edges give by Appmin + N dAppmag.
            # We probably want to smooth this eventually due to minor
            # numerical tweaks
            
            #kmag2 -= imag1
            #kmag1 = 1.-kmag2
            #imag2 = imag1+1
            #prior = kmag1*self.priors[imag1] + kmag2*self.priors[imag2]
            
            # simple linear interpolation
            Oi = self.priors[imag1] * kmag1 + self.priors[imag2] * kmag2
            
            # correct normalisation - otherwise, priors are defined to sum
            # such that \sum priors = 1; here, we need \int priors dm = 1
            Oi /= self.dAppmag 
            
            # modify sigma_ms by P(m|O)
            pogm = (1.-pUgm(mag,self.pU_mean,self.pU_width))
            numerator = Sigma_ms[i] * pogm
            Oi /= numerator # normalise by host counts
            
            Ois.append(Oi)
        
        Ois = np.array(Ois)
        return Ois
    



################# Useful functions not associated with a class #########


def load_marnoch_data():
        """
        Loads the Marnoch et al data on r-band magnitudes from FRB hosts
        """
        from astropy.table import Table
        datafile="magnitudes_and_probabilities_vlt-fors2_R-SPECIAL.ecsv"
        infile =  os.path.join(resources.files('zdm'), 'data', 'optical', datafile)
        table = Table.read(infile, format='ascii.ecsv')
        return table

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


def SimplekApparentMags(Abs,k,zs):
    """
    Convert absolute to apparent magnitudes with a power-law k-correction.

    Applies the distance modulus plus a k-correction of the form
    ``2.5 * k * log10(1 + z)``.

    Args:
        Abs (float or np.ndarray): absolute magnitude(s) M_r.
        k (float): k-correction power-law index. ``k=0`` reduces to a
            pure distance modulus (identical to ``SimpleApparentMags``).
        zs (float or np.ndarray): redshift(s) of the galaxies.

    Returns:
        ApparentMags: apparent magnitude(s). Scalar if both inputs are
            scalar; 1-D array if one is scalar and one is an array; 2-D
            array of shape (NAbs, Nz) if both are arrays (computed via
            ``np.outer``).
    """
    
    # calculates luminosity distances (Mpc)
    lds = cos.dl(zs)
    
    # finds distance relative to absolute magnitude distance
    dabs = 1e-5 # in units of Mpc
    
    # k-corrections
    kcorrs = (1+zs)**k
    
    dk = 2.5*np.log10(kcorrs) #i.e., 2.5*k*np.log10(1+z)
    
    # relative magnitude
    dMag = 2.5*np.log10((lds/dabs)**(2)) + dk
    
    
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

def SimplekAbsoluteMags(App,k,zs):
    """
    Convert apparent to absolute magnitudes with a power-law k-correction.

    Inverse of ``SimplekApparentMags``: subtracts the distance modulus and
    k-correction ``2.5 * k * log10(1 + z)`` from the apparent magnitude.

    Args:
        App (float or np.ndarray): apparent magnitude(s) m_r.
        k (float): k-correction power-law index. ``k=0`` reduces to a
            pure distance modulus (identical to ``SimpleAbsoluteMags``).
        zs (float or np.ndarray): redshift(s) of the galaxies.

    Returns:
        AbsoluteMags: absolute magnitude(s). Scalar if both inputs are
            scalar; 1-D array if one is scalar and one is an array; 2-D
            array of shape (NApp, Nz) if both are arrays (computed via
            ``np.outer``).
    """
    
    # calculates luminosity distances (Mpc)
    lds = cos.dl(zs)
    
    # finds distance relative to absolute magnitude distance
    dabs = 1e-5 # in units of Mpc
    
    # k-corrections
    kcorrs = (1+zs)**k
    
    dk = 2.5*np.log10(kcorrs)
    
    # relative magnitude
    dMag = 2.5*np.log10((lds/dabs)**(2)) + dk
    
    
    if np.isscalar(zs) or np.isscalar(Abs):
        # just return the product, be it scalar x scalar,
        # scalar x array, or array x scalar
        # this also ensures that the dimensions are as expected
        
        AbsoluteMags = App - dMag
    else:
        # Convert to multiplication so we can use
        # numpy.outer
        temp1 = 10**App
        temp2 = 10**-dMag
        AbsoluteMags = np.outer(temp1,temp2)
        AbsoluteMags = np.log10(ApparentMags)
    return AbsoluteMags

def SimpleAbsoluteMags(App,zs):
    """
    Convert apparent to absolute magnitudes using the distance modulus only.

    Subtracts ``5 * log10(D_L / 10 pc)`` from the apparent magnitude, where
    D_L is the luminosity distance in Mpc. No k-correction is applied.

    Args:
        App (float or np.ndarray): apparent magnitude(s) m_r.
        zs (float or np.ndarray): redshift(s) of the galaxies.

    Returns:
        AbsoluteMags: absolute magnitude(s). Scalar if both inputs are
            scalar; 1-D array if one input is scalar and one is an array;
            2-D array of shape (NApp, Nz) if both are arrays (computed via
            ``np.outer``).
    """
    
    # calculates luminosity distances (Mpc)
    lds = cos.dl(zs)
    
    # finds distance relative to absolute magnitude distance
    dabs = 1e-5 # in units of Mpc
    
    # relative magnitude
    dMag = 2.5*np.log10((lds/dabs)**(2))
    
    
    if np.isscalar(zs) or np.isscalar(Abs):
        # just return the product, be it scalar x scalar,
        # scalar x array, or array x scalar
        # this also ensures that the dimensions are as expected
        
        AbsoluteMags = App - dMag
    else:
        # Convert to multiplication so we can use
        # numpy.outer
        temp1 = 10**App
        temp2 = 10**-dMag
        AbsoluteMags = np.outer(temp1,temp2)
        AbsoluteMags = np.log10(ApparentMags)
    return AbsoluteMags

def SimpleApparentMags(Abs,zs):
    """
    Convert absolute to apparent magnitudes using the distance modulus only.

    Adds ``5 * log10(D_L / 10 pc)`` to the absolute magnitude, where
    D_L is the luminosity distance in Mpc. No k-correction is applied.

    Args:
        Abs (float or np.ndarray): absolute magnitude(s) M_r.
        zs (float or np.ndarray): redshift(s) of the galaxies.

    Returns:
        ApparentMags: apparent magnitude(s). Scalar if both inputs are
            scalar; 1-D array if one input is scalar and one is an array;
            2-D array of shape (NAbs, Nz) if both are arrays (computed via
            ``np.outer``).
    """
    
    # calculates luminosity distances (Mpc)
    lds = cos.dl(zs)
    lds_pc = lds*1e6
    
    # relative magnitude
    dMag = 5*np.log10(lds_pc) - 5
    
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
    Return the probability that an FRB host galaxy is unseen in typical VLT observations.

    Digitises Figure 3 of Marnoch et al. 2023 (MNRAS 525, 994), which shows
    p(U | z) — the cumulative probability that a host galaxy at redshift z
    falls below the VLT/FORS2 R-band detection limit. A cubic polynomial is
    fit to the digitised curve and evaluated at the requested redshifts.
    Values are clamped to [0, 1].

    Args:
        zvals (float or np.ndarray): redshift(s) at which to evaluate p(U | z).
        plot (bool): if True, save a diagnostic comparison plot of the raw
            digitised data, linear interpolation, and polynomial fit to
            ``p_unseen.pdf``. Defaults to False.

    Returns:
        fitv (np.ndarray): p(U | z) evaluated at each element of ``zvals``,
            clamped to the range [0, 1].
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
    Reduce a TNS FRB name to a six-character YYMMDD[L] identifier.

    Strips the leading ``FRB`` prefix (if present) and the year's
    century digits, retaining only the six-digit date plus any
    trailing letter suffix, to allow case-insensitive matching
    between survey entries and external FRB catalogues.

    Args:
        TNSname (str): FRB name in TNS format, e.g. ``'FRB20180924B'``
            or ``'20180924B'``.

    Returns:
        name (str): simplified six-character identifier, e.g. ``'180924B'``
            (six digits plus optional letter).
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
    Find the index of an FRB in a survey by TNS name.

    Uses ``simplify_name`` to normalise both the query name and the survey
    entries, so minor formatting differences (century digits, trailing
    letters) do not prevent a match.

    Args:
        TNSname (str): TNS name of the FRB to look up, e.g.
            ``'FRB20180924B'``.
        survey (Survey): loaded survey object whose ``frbs["TNS"]`` column
            contains TNS names of detected FRBs.

    Returns:
        int or None: index into ``survey.frbs`` of the first matching FRB,
            or ``None`` if the FRB is not found in the survey.
    """
    
    name = simplify_name(TNSname)
    match = None
    for i,frb in enumerate(survey.frbs["TNS"]):
        if name == simplify_name(frb):
            match = i
            break
    return match


# this defines the ICS FRBs for which we have PATH info
# notes: FRB20230731A and 'FRB20230718A' are too reddened, so are removed
# still aiming to follow up frb20240208A and frb20240318A
frblist=['FRB20180924B','FRB20181112A','FRB20190102C','FRB20190608B',
        'FRB20190611B','FRB20190711A','FRB20190714A','FRB20191001A',
        'FRB20191228A','FRB20200430A','FRB20200906A','FRB20210117A',
        'FRB20210320C','FRB20210807D','FRB20210912A','FRB20211127I','FRB20211203C',
        'FRB20211212A','FRB20220105A','FRB20220501C',
        'FRB20220610A','FRB20220725A','FRB20220918A',
        'FRB20221106A','FRB20230526A','FRB20230708A',
        'FRB20230902A','FRB20231226A','FRB20240201A',
        'FRB20240210A','FRB20240304A','FRB20240310A']




def plot_frb(name,ralist,declist,plist,opfile):
    """
    Plot an FRB localisation and its PATH host galaxy candidates.

    Produces a scatter plot showing the FRB position and a set of
    deviated/sampled positions colour-coded by their PATH posterior
    P(O|x), overlaid with circles representing candidate host galaxies
    scaled by their angular size. All coordinates are shown in arcseconds
    relative to the FRB position.

    Args:
        name (str): TNS FRB name (e.g. ``'FRB20180924B'``), used to load
            the FRB object and the corresponding PATH candidate table.
        ralist (np.ndarray): right ascension values (degrees) of deviated
            FRB positions to plot, colour-coded by ``plist``.
        declist (np.ndarray): declination values (degrees) of deviated
            FRB positions.
        plist (np.ndarray): PATH posterior values P(O|x) for the deviated
            positions, used to set the colour scale.
        opfile (str): output file path for the saved figure.
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



def pUgm(mag,mean,width):
    """
    Return the probability that a galaxy is undetected as a function of magnitude.

    Models the survey detection completeness as a logistic function that
    transitions from ~0 (bright, always detected) to ~1 (faint, never
    detected) with a smooth rolloff centred on ``mean``:

        p(U | m) = 1 / (1 + exp((mean - m) / width))

    Args:
        mag (float or np.ndarray): r-band apparent magnitude(s) at which to
            evaluate the detection-failure probability.
        mean (float): magnitude at which p(U | m) = 0.5 (the 50% completeness
            limit of the survey).
        width (float): characteristic width of the completeness rolloff in
            magnitudes. Smaller values give a sharper transition.

    Returns:
        pU (float or np.ndarray): probability of non-detection at each
            magnitude in ``mag``, in the range [0, 1].
    """
    
    # converts to a number relative to the mean. Will be weird for mags < 0.
    diff = (mean-mag)/width
    
    # converts the diff to a power of 10
    pU = 1./(1+np.exp(diff))
    
    return pU
    
    
