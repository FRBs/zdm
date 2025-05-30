from IPython.terminal.embed import embed
import numpy as np
import datetime

from zdm import cosmology as cos
from zdm import misc_functions
from zdm import energetics
from zdm import pcosmic
from zdm import io
import time
import warnings

class Grid:
    """A class to hold a grid of z-dm plots
    
    Fundamental assumption: each z point represents FRBs created *at* that redshift
    Interpolation is performed under this assumption.
    
    It also assumes a linear uniform grid.
    """

    def __init__(self, survey, state, zDMgrid, zvals, dmvals, smear_mask, wdist=None, prev_grid=None):
        """
        Class constructor.

        Args: 
            survey (survey.Survey):
            state (parameters.State): 
                Defines the parameters of the analysis
                Note, each grid holds the *same* copy so modifying
                it in one place affects them all.
            zvals (np.1darray, float):
                redshift values of the grid. These are "bin centres",
                representing ranges from +- dz/2.
            dmvals (np.1darray, float:
                DM values of the grid. These are These are "bin centres",
                representing ranges from +- dDM/2.
            smear_mask (np.1darray, float):
                1D array
            wdist (bool):
                If True, allow for a distribution of widths
            prev_grid (grid.Grid):
                Another grid with the same parameters just 
                corresponding to a different survey
        """
        self.grid = None
        self.survey = survey
        self.verbose = False
        # Beam
        self.beam_b = survey.beam_b
        self.beam_o = survey.beam_o
        self.b_fractions = None
        # State
        self.state = state
        self.MCinit = False
        self.source_function = cos.choose_source_evolution_function(
            state.FRBdemo.source_evolution
        )

        # Energetics
        if self.state.energy.luminosity_function in [3]:
            self.use_log10 = True
        else:
            self.use_log10 = False
        self.luminosity_function = self.state.energy.luminosity_function
        self.init_luminosity_functions()
        self.nuObs= survey.meta['FBAR']*1e6 #from MHz to Hz
        
        # Init the grid
        #   THESE SHOULD BE THE SAME ORDER AS self.update()
        self.parse_grid(zDMgrid.copy(), zvals.copy(), dmvals.copy())

        if prev_grid is None:
            self.calc_dV()
            self.smear_dm(smear_mask.copy())
        else:
            self.dV = prev_grid.dV.copy()
            self.smear = prev_grid.smear.copy()
            self.smear_grid = prev_grid.smear_grid.copy()
            
        if wdist is not None:
            efficiencies = survey.efficiencies  # two OR three dimensions
            weights = survey.wplist
            # Warning -- THRESH could be different for each FRB, but we don't treat it that way
            self.calc_thresholds(survey.meta["THRESH"],
                             efficiencies,weights=weights)
        else:
            # if this is the case, why calc thresholds again below?
            efficiencies = survey.mean_efficiencies # one dimension
            weights = None
            self.calc_thresholds(survey.meta["THRESH"], efficiencies, weights=weights)
            efficiencies=survey.mean_efficiencies
        
        # Calculate
        self.calc_pdv()
        self.set_evolution()  # sets star-formation rate scaling with z - here, no evoltion...
        self.calc_rates()  # includes sfr smearing factors and pdv mult

    def init_luminosity_functions(self):
        """ Set the luminsoity function for FRB energetics """
        if self.luminosity_function == 0:  # Power-law
            self.array_cum_lf = energetics.array_cum_power_law
            self.vector_cum_lf = energetics.vector_cum_power_law
            self.array_diff_lf = energetics.array_diff_power_law
            self.vector_diff_lf = energetics.vector_diff_power_law
        elif self.luminosity_function == 1:  # Gamma function
            embed(header="79 of grid -- BEST NOT TO USE THIS!!!!")
            self.array_cum_lf = energetics.array_cum_gamma
            self.vector_cum_lf = energetics.vector_cum_gamma
            self.array_diff_lf = energetics.array_diff_gamma
            self.vector_diff_lf = energetics.vector_diff_gamma
        elif self.luminosity_function == 2:  # Spline gamma function
            self.array_cum_lf = energetics.array_cum_gamma_spline
            self.vector_cum_lf = energetics.vector_cum_gamma_spline
            self.array_diff_lf = energetics.array_diff_gamma
            self.vector_diff_lf = energetics.vector_diff_gamma
        elif self.luminosity_function == 3:  # Linear + log10
            self.array_cum_lf = energetics.array_cum_gamma_linear
            self.vector_cum_lf = energetics.vector_cum_gamma_linear
            self.array_diff_lf = energetics.array_diff_gamma
            self.vector_diff_lf = energetics.vector_diff_gamma
        else:
            raise ValueError(
                "Luminosity function must be 0, not ", self.luminosity_function
            )

    def parse_grid(self, zDMgrid, zvals, dmvals):
        self.grid = zDMgrid
        self.zvals = zvals
        self.dmvals = dmvals
        #
        self.check_grid()
        # self.calc_dV()

        # this contains all the values used to generate grids
        # these parameters begin at None, and get filled when
        # ever something is regenerated. They are semi-hierarchical
        # in that if a low-level value is reset, high-level ones
        # get put to None.

    def load_grid(self, gridfile, zfile, dmfile):
        self.grid = io.load_data(gridfile)
        self.zvals = io.load_data(zfile)
        self.dmvals = io.load_data(dmfile)
        self.check_grid()
        self.volume_grid()
    
    def get_dm_coeffs(self, DMlist):
        """
        Returns indices and coefficients for interpolating between DM values
        
        dmlist: np.ndarray of dispersion measures (extragalactic!)
        """
        # get indices in dm space
        kdms=DMlist/self.ddm - 0.5 # idea: if DMobs = ddm, we are half-way between bin 0 and bin 1
        Bin0 = np.where(kdms < 0.)[0] # if DMs are in the lower half of the lowest bin, use lowest bin only
        kdms[Bin0] = 0. 
        idms1=kdms.astype('int') # rounds down
        idms2=idms1+1
        dkdms2=kdms-idms1 # applies to idms2, i.e. the upper bin. If DM = ddm, then this should be 0.5
        dkdms1 = 1.-dkdms2 # applies to idms1
        return idms1,idms2,dkdms1,dkdms2
    
    def get_z_coeffs(self,zlist):
        """
        Returns indices and coefficients for interpolating between z values
        
        zlist: np.ndarray of dispersion measures (extragalactic!)
        """
        
        kzs=zlist/self.dz - 0.5
        Bin0 = np.where(kzs < 0.)[0]
        kzs[Bin0] = 0. 
        izs1=kzs.astype('int')
        izs2=izs1+1
        dkzs2=kzs-izs1 # applies to izs2
        dkzs1 = 1. - dkzs2
        
        # checks for values which are too large
        toobigz = np.where(zlist > self.zvals[-1] + self.dz/2.)[0]
        if len(toobigz) > 0:
            raise ValueError("Redshift values ",zlist[toobigz],
                " too large for grid max of ",self.zvals[-1] + self.dz/2.)
        
        # checks for zs in top half of top bin - only works because of above bin
        topbin = np.where(zlist > self.zvals[-1])[0]
        if len(topbin) > 0:
            izs2[topbin] = self.zvals.size-1
            izs1[topbin] = self.zvals.size-2
            dkzs2[topbin] = 1.
            dkzs1[topbin] = 0.
        
        return izs1, izs2, dkzs1, dkzs2
    
    def check_grid(self,TOLERANCE = 1e-6):
        """
        Check that the grid values are behaving as expected
        
        TOLERANCE: defines the max relative difference in expected
                    and found values that will be tolerated
        """
        self.nz = self.zvals.size
        self.ndm = self.dmvals.size
        
        # check to see if these are log-spaced
        if (self.zvals[-1] - self.zvals[-2]) / (self.zvals[1] - self.zvals[0]) > 1.01:
            if (
                np.abs(self.zvals[-1] * self.zvals[0] - self.zvals[-2] * self.zvals[1])
                > 0.01
            ):
                raise ValueError("Cannot determine scaling of zvals, exiting...")
            self.zlog = True
            self.dz = np.log(self.zvals[1] / self.zvals[0])
        else:
            self.zlog = False
            self.dz = self.zvals[1] - self.zvals[0]
        
        self.ddm = self.dmvals[1] - self.dmvals[0]
        shape = self.grid.shape
        if shape[0] != self.nz:
            if shape[0] == self.ndm and shape[1] == self.nz:
                print("Transposing grid, looks like first index is DM")
                self.grid = self.grid.transpose
            else:
                raise ValueError("wrong shape of grid for zvals and dm vals")
        else:
            if shape[1] == self.ndm:
                if self.verbose:
                    print("Grid successfully initialised")
            else:
                raise ValueError("wrong shape of grid for zvals and dm vals")

        # checks that the grid is approximately linear to high precision
        if self.zlog:
            expectation = np.exp(np.arange(0, self.nz) * self.dz) * self.zvals[0]
        else:
            expectation = self.dz * np.arange(0, self.nz) + self.zvals[0]
        diff = self.zvals - expectation
        maxoff = np.max(diff ** 2)
        if maxoff > TOLERANCE * self.dz:
            raise ValueError(
                "Maximum non-linearity in z-grid of ",
                maxoff ** 0.5,
                "detected, aborting",
            )
        
        # Ensures that log-spaced bins are truly bin centres
        if not self.zlog and np.abs(self.zvals[0] - self.dz/2.) > TOLERANCE*self.dz:
            raise ValueError(
                "Linear z-grids *must* begin at dz/2. e.g. 0.05,0.15,0.25 etc, ",
                " first value ",self.zvals[0]," expected to be half of spacing ",
                self.dz,", aborting..."
            )
                
        
        expectation = self.ddm * np.arange(0, self.ndm) + self.dmvals[0]
        diff = self.dmvals - expectation
        maxoff = np.max(diff ** 2)
        if maxoff > TOLERANCE * self.ddm:
            raise ValueError(
                "Maximum non-linearity in dm-grid of ",
                maxoff ** 0.5,
                "detected, aborting",
            )

    def calc_dV(self, reINIT=False):
        """ Calculates volume per steradian probed by a survey.
        
        Does this only in the z-dimension (for obvious reasons!)
        """

        if (cos.INIT is False) or reINIT:
            # print('WARNING: cosmology not yet initiated, using default parameters.')
            cos.init_dist_measures()
        if self.zlog:
            # if zlog, dz is actually .dlogz. And dlogz/dz=1/z, i.e. dz= z dlogz
            self.dV = cos.dvdtau(self.zvals) * self.dz * self.zvals
        else:
            self.dV = cos.dvdtau(self.zvals) * self.dz

    def EF(self, alpha=0, bandwidth=1e9):
        """Calculates the fluence--energy conversion factors as a function of redshift
        This does NOT account for the central frequency
        """
        if self.state.FRBdemo.alpha_method == 0:
            self.FtoE = cos.F_to_E(
                1,
                self.zvals,
                alpha=alpha,
                bandwidth=bandwidth,
                Fobs=self.nuObs,
                Fref=self.nuRef,
            )
        elif self.state.FRBdemo.alpha_method == 1:
            self.FtoE = cos.F_to_E(1, self.zvals, alpha=0.0, bandwidth=bandwidth)
        else:
            raise ValueError("alpha method must be 0 or 1, not ", self.alpha_method)

    def set_evolution(self):  # ,n,alpha=None):
        """ Scales volumetric rate by SFR """
        self.sfr=self.source_function(self.zvals,
                                      self.state.FRBdemo.sfr_n)
        if self.state.FRBdemo.alpha_method==1:
            self.sfr *= (1.0 + self.zvals)**(-self.state.energy.alpha) #reduces rate with alpha

            # changes absolute normalisation at z=0 according to central frequency
            self.sfr *= (
                self.nuObs / self.nuRef
                ) ** -self.state.energy.alpha  # alpha positive, nuObs<nuref, expected rate increases

    def calc_pdv(self, beam_b=None, beam_o=None):
        """ Calculates the rate per cell.
        Assumed model: a power-law between Emin and Emax (erg)
                       with slope gamma.
        Efficiencies: list of efficiency response to DM
        So-far: does NOT include time x solid-angle factor
        
        NOW: this includes a solid-angle and beam factor if initialised
        
        This will recalculate beam factors if they are passed, however
        during iteration this is not recalculated
        """

        if beam_b is not None:
            self.beam_b = beam_b
            self.beam_o = beam_o
            try:
                x = beam_o.shape
                x = beam_b.shape
            except:
                raise ValueError(
                    "Beam values must be numby arrays! Currently ", beam_o, beam_b
                )
        # linear weighted sum of probabilities: pdVdOmega now. Could also be used to include time factor

        # For convenience and speed up
        Emin = 10 ** self.state.energy.lEmin
        Emax = 10 ** self.state.energy.lEmax

        # this implementation allows us to access the b-fractions later on
        if (not (self.b_fractions is not None)) or (beam_b is not None):
            self.b_fractions = np.zeros(
                [self.zvals.size, self.dmvals.size, self.beam_b.size]
            )

        # for some arbitrary reason, we treat the beamshape slightly differently... no need to keep an intermediate product!
        main_beam_b = self.beam_b
        
        # call log10 beam
        if self.use_log10:
            new_thresh = np.log10(
                self.thresholds
            )  # use when calling in log10 space conversion
            main_beam_b = np.log10(main_beam_b)

        for i, b in enumerate(main_beam_b):
            # if eff_weights is 2D (i.e., z-dependent) then w is a vector of length NZ
            for j, w in enumerate(self.eff_weights):
                # using log10 space conversion
                if self.use_log10:
                    thresh = new_thresh[j, :, :] - b
                else:  # original
                    thresh = self.thresholds[j, :, :] / b
                
                # the below is to ensure this works when w is a vector of length nz
                w = np.array(w)
                
                self.b_fractions[:, :, i] += (
                        self.beam_o[i]
                        * (self.array_cum_lf(
                            thresh, Emin, Emax, self.state.energy.gamma, self.use_log10
                        ).T * w.T).T
                    )
        # here, b-fractions are unweighted according to the value of b.
        self.fractions = np.sum(
            self.b_fractions, axis=2
        )  # sums over b-axis [ we could ignore this step?]
        self.pdv = np.multiply(self.fractions.T, self.dV).T

    def calc_rates(self):
        """ multiplies the rate per cell with the appropriate pdm plot """

        try:
            self.sfr
        except:
            print("WARNING: no evolutionary weight yet applied")
            exit()

        try:
            self.smear_grid
        except:
            print("WARNING: no DMx smearing yet applied")
            exit()

        try:
            self.pdv
        except:
            print("WARNING: no volumetric probability pdv yet calculated")
            exit()

        self.sfr_smear = np.multiply(self.smear_grid.T, self.sfr).T

        self.rates = self.pdv * self.sfr_smear

    def calc_thresholds(self, F0:float, 
                        eff_table, 
                        bandwidth=1e9, 
                        nuRef=1.3e9, weights=None):
        """ Sets the effective survey threshold on the zdm grid

        Args:
            F0 (float): base survey threshold
            eff_table ([type]): table of efficiencies corresponding to DM-values. 1, 2, or 3 dimensions!
            bandwidth ([type], optional): [description]. Defaults to 1e9.
            nuObs ([float], optional): survey frequency (affects sensitivity via alpha - only for alpha method)
                Defaults to 1.3e9.
            nuRef ([float], optional): reference frequency we are calculating thresholds at
                Defaults to 1.3e9.
            weights ([type], optional): [description]. Defaults to None.

        Raises:
            ValueError: [description]
            ValueError: [description]
        """
        # keep the inputs for later use
        self.F0 = F0
        self.nuRef = nuRef

        self.bandwidth = bandwidth
        if eff_table.ndim == 1:  # only a single FRB width: dimensions of NDM
            self.nthresh = 1
            self.eff_weights = np.array([1])
            self.eff_table = np.array([eff_table])  # make it an extra dimension
        else:  # multiple FRB widths: dimensions nW x NDM
            # check that the weights dimensions check out
            self.nthresh = eff_table.shape[0] # number of width bins.
            if weights is not None:
                if weights.shape[0] != self.nthresh:
                    raise ValueError(
                        "Dimension of weights must equal first dimension of efficiency table"
                    )
            else:
                raise ValueError(
                    "For a multidimensional efficiency table, please set relative weights"
                )
            # I have removed weight normalisation here. In theory, normalisation to <1 is
            # a feature, not a bug, representing more/less scattering moving into the
            # observable range
            self.eff_table = eff_table
            self.eff_weights = weights
        
        # now two or three dimensions
        Eff_thresh = F0 / self.eff_table
        
        self.EF(self.state.energy.alpha, bandwidth)  # sets FtoE values - could have been done *WAY* earlier

        self.thresholds = np.zeros([self.nthresh, self.zvals.size, self.dmvals.size])

        # Performs an outer multiplication of conversion from fluence to energy.
        # The FtoE array has one value for each redshift.
        # The effective threshold array has one value for each combination of
        # FRB width (nthresh) and DM.
        # We loop over nthesh and generate a NDM x Nz array for each
        for i in np.arange(self.nthresh):
            if self.eff_table.ndim == 2:
                self.thresholds[i,:,:] = np.outer(self.FtoE, Eff_thresh[i,:])
            else:
                self.thresholds[i,:,:] =  ((Eff_thresh[i,:,:]).T * self.FtoE).T
        
        
    def smear_dm(self, smear:np.ndarray):  # ,mean:float,sigma:float):
        """ Smears DM using the supplied array.
        Example use: DMX contribution

        smear_grid is created in place

        Args:
            smear (np.ndarray): Smearing array
        """
        # just easier to have short variables for this

        ls = smear.size
        lz, ldm = self.grid.shape

        if not hasattr(self, "smear_grid"):
            self.smear_grid = np.zeros([lz, ldm])
        self.smear = smear

        # this method is O~7 times faster than the 'brute force' above for large arrays
        for i in np.arange(lz):
            # we need to get the length of mode='same', BUT
            # we do not want it 'centred', hence must make cut on full
            if smear.ndim == 1:
                self.smear_grid[i, :] = np.convolve(
                    self.grid[i, :], smear, mode="full"
                )[0:ldm]
            elif smear.ndim == 2:
                self.smear_grid[i, :] = np.convolve(
                    self.grid[i, :], smear[i, :], mode="full"
                )[0:ldm]
            else:
                raise ValueError(
                    "Wrong number of dimensions for DM smearing ", smear.shape
                )

    def get_p_zgdm(self, DMs: np.ndarray):
        """ Calcuates the probability of redshift given a DM
        We already have our grid of observed DM values.
        Just take slices!

        Args:
            DMs (np.ndarray): array of DM values

        Returns:
            np.ndarray: array of priors for the DMs
        """
        # first gets ids of matching DMs
        priors = np.zeros([DMs.size, self.zvals.size])
        for i, dm in enumerate(DMs):
            DM2 = np.where(self.dmvals > dm)[0][0]
            DM1 = DM2 - 1
            kDM = (dm - self.dmvals[DM1]) / (self.dmvals[DM2] - self.dmvals[DM1])
            priors[i, :] = kDM * self.rates[:, DM2] + (1.0 - kDM) * self.rates[:, DM1]
            priors[i, :] /= np.sum(priors[i, :])
        return priors

    def GenMCSample(self, N, Poisson=False):
        """
        Generate a MC sample of FRB events
        
        If Poisson=True, then interpret N as a Poisson expectation value
        Otherwise, generate precisely N FRBs
        
        Generated values are [MCz, MCDM, MCb, MCs, MCw]
        NOTE: the routine GenMCFRB does not know 'w', merely
            which w bin it generates.
        
        """
        # Boost?
        if self.state.energy.luminosity_function in [1, 2]:
            Emax_boost = 3.0
        else:
            Emax_boost = 0.0

        if Poisson:
            # from np.random import poisson
            NFRB = np.random.poisson(N)
        else:
            NFRB = int(N)  # just to be sure...
        sample = []
        
        for i in np.arange(NFRB):
            if (i % 100) == 0:
                print(i)
            
            # Regen if the survey would not find this FRB
            frb = self.GenMCFRB(Emax_boost)
            # This is a pretty naive method of generation.
            while frb[1] > self.survey.max_dm:
                print("Regenerating MC FRB with too high DM ",frb[1],self.survey.max_dm)
                frb = self.GenMCFRB(Emax_boost)

            sample.append(frb)
            
        sample = np.array(sample)
        return sample
    
    
    def initMC(self):
        """
        Initialises the MC sample, if it has not been doen already
        This uses a great deal of RAM - hence, do not do this lightly!
        """
        
        # shorthand
        lEmin = self.state.energy.lEmin
        lEmax = self.state.energy.lEmax
        gamma = self.state.energy.gamma
        Emin = 10 ** lEmin
        Emax = 10 ** lEmax
        
        # grid of beam values, weights
        nw = self.eff_weights.size
        nb = self.beam_b.size
        
        if self.eff_weights.ndim > 1:
            raise ValueError("MC generation from z-dependent widths not currently enabled")
        
        # holds array of probabilities in w,b space
        pwb = np.zeros([nw * nb])
        rates = []
        pzcs = []
        
        # gets list of DM probabilities to set to zero due to
        # the survey missing these FRBs
        if self.survey.max_dm is not None:
            setDMzero = np.where(self.dmvals +self.ddm/2. > self.survey.max_dm)[0]
                  
        # Generates a joint distribution in B,w
        for i, b in enumerate(self.beam_b):
            for j, w in enumerate(self.eff_weights):
                # each of the following is a 2D array over DM, z which we sum to generate B,w values
                pzDM = self.array_cum_lf(
                            self.thresholds[j, :, :] / b,
                            Emin, Emax, gamma)
                
                # sets to zero if we have a max survey DM
                if self.survey.max_dm is not None:
                    pzDM [:,setDMzero] = 0.
                
                # weighted pzDM
                wb_fraction = (self.beam_o[i] * w * pzDM)
                pdv = np.multiply(wb_fraction.T, self.dV).T
                rate = pdv * self.sfr_smear
                rates.append(rate)
                pwb[i * nw + j] = np.sum(rate)
                
                pz = np.sum(rate, axis=1)
                pzc = np.cumsum(pz)
                pzc /= pzc[-1]
                
                pzcs.append(pzc)
        
        # generates cumulative distribution for sampling w,b
        pwbc = np.cumsum(pwb)
        pwbc /= pwbc[-1]
        
        # saves cumulative distributions for sampling
        self.MCpwbc = pwbc
        
        # saves individal wxb zDM rates for sampling these distributions
        self.MCrates = rates
        
        # saves projections onto z-axis
        self.MCpzcs = pzcs
        
        self.MCinit = True
    
    def GenMCFRB(self, Emax_boost):
        """
        Generates a single FRB according to the grid distributions
        
        Samples beam position b, FRB DM, z, s=SNR/SNRth, and w
        Currently: no interpolation included.
        This should be implemented in s,DM, and z.
        
        NOTE: currently, the actual FRB widths are not part of 'grid'
            only the relative probabilities of any given width.
            Hence, this routine only returns the integer of the width bin
            not the width itelf.

        Args:
            pwb (optional): probability(width,beam)
            Emax_boost (float, optional): 
                Allow for larger energies than Emax
                The factor is logarithmic, i.e. Emax_boost = 2. allows
                for 10**2 higher energies than Emax

        Returns:
            tuple: FRBparams=[MCz,MCDM,MCb,j,MCs], pwb values
            These are:
                MCz: redshift
                MCDM: dispersion measure (extragalactic)
                MCb: beam value 
                j: 
                MCs: SNR/SNRth value of FRB
                MCw: width value of FRB
            [MCz, MCDM, MCb, j, MCs, MCw]
        """
        
        # shorthand
        lEmin = self.state.energy.lEmin
        lEmax = self.state.energy.lEmax
        gamma = self.state.energy.gamma
        Emin = 10 ** lEmin
        Emax = 10 ** lEmax
        
        # grid of beam values, weights
        nw = self.eff_weights.size
        if self.eff_weights.ndim > 1:
            raise ValueError("MC generation from z-dependent widths not currently enabled")
        nb = self.beam_b.size
        
        if not self.MCinit:
            self.initMC()
        
        # sample distribution in w,b
        # we do NOT interpolate here - we treat these as qualitative values
        # i.e. as if there was an irregular grid of them
        r = np.random.rand(1)[0]
        which = np.where(self.MCpwbc > r)[0][0]
        i = int(which / nw)
        j = which - i * nw
        MCb = self.beam_b[i]
        MCw = self.eff_weights[j]
        
        # get p(z,DM) distribution for this b,w
        pzDM = self.MCrates[which]
        
        pzc = self.MCpzcs[which]
        
        r = np.random.rand(1)[0]
        
        # sampling in DM and z space
        # First choose z: pzc is the cumulative distribution in z
        # for all dm
        # each probability represents the p(bin), i.e. z-dz/2. to z+dz/2
        # first, find the bin where the cumulative probability is higher
        # than the sampled amount.
        # Alternative method: just use the distribution from the bin,
        # then multiply the resulting DM linearly with deltaz/z.
        # Would be better at low z, worse at high z
        iz2 = np.where(pzc > r)[0][0]
        if iz2 > 0:
            iz1 = iz2 - 1
            iz3 = iz2 + 1
            dr = r - pzc[iz1]
            fz = dr / (pzc[iz2] - pzc[iz1])  # fraction of way to upper z value
            
            # move a fraction of kz2 between z-dz/0.5 and z + dz/0.5
            #MCz = self.zvals[iz2] + (kz2-0.5)*dz
            
            # weigts between iz1 and iz2
            if fz < 0.5:
                kz1 = 0.5 - fz
                kz2 = 0.5 + fz
                kz3 = 0.
                iz3 = 0 # dummy
            elif iz2 == self.zvals.size-1:
                # we are in the last bin - don't extrapolate, just use it
                kz1 = 0.
                kz2 = 1.
                kz3 = 0.
                iz1 = 0 # dummy
                iz3 = 0 # dummy
            else:
                kz1 = 0.
                kz2 = (1.5-fz)
                kz3 = fz-0.5
                iz1 = 0 #dummy
            pDM = pzDM[iz1, :] * kz1 + pzDM[iz2, :] * kz2 + pzDM[iz3, :] * kz3
            MCz = self.zvals[iz1] * kz1 + self.zvals[iz2] * kz2 + self.zvals[iz3]*kz3
        else:
            # we perform a simple linear interpolation in z from 0 to minimum bin
            fz = r / pzc[iz2]
            kz1 = 0.
            kz2 = 1.
            kz3 = 0.
            iz1 = 0 # dummy
            iz3 = 0 # dummy
            MCz = (self.zvals[iz2] + self.dz/0.5) * fz
            # Just use the value of lowest bin.
            # This is a gross and repugnant approximation
            pDM = pzDM[iz2, :]
            
        # NOW DO dm
        # DM represents the distribution for the centre of z-bin
        pDMc = np.cumsum(pDM)
        pDMc /= pDMc[-1]
        r = np.random.rand(1)[0]
        iDM2 = np.where(pDMc > r)[0][0]
        if iDM2 > 0:
            iDM1 = iDM2 - 1
            iDM3 = iDM2 + 1
            dDM = r - pDMc[iDM1]
            # fraction of value through DM bin
            fDM = dDM / (pDMc[iDM2] - pDMc[iDM1])
            
            # get the MC DM through interpolation
            if fDM < 0.5:
                kDM1 = 0.5 - fDM
                kDM2 = 0.5 + fDM
                kDM3 = 0.
                iDM3 = 0    # dummy
                # sets iDM3 to be safe at 0
                MCDM = self.dmvals[iDM1] * kDM1 + self.dmvals[iDM2] * kDM2
            elif iDM2 == self.dmvals.size-1:
                kDM1 = 0.
                kDM2 = 1. # for future use, not here
                kDM3 = 0.
                iDM1 = 0    # dummy
                iDM3 = 0    # dummy
                MCDM = self.dmvals[iDM2] + (fDM - 0.5)*self.dDM # upper DM bins
            else:
                kDM1 = 0.
                kDM2 = 1.5-fDM
                kDM3 = fDM - 0.5
                iDM1 = 0    # dummy
                
                MCDM = self.dmvals[iDM3] * kDM3 + self.dmvals[iDM2] * kDM2
        else:
            # interpolate linearly from 0 to the minimum value
            fDM = r / pDMc[iDM2]
            MCDM = (self.dmvals[iDM2] + self.ddm/2.) * fDM
            kDM1 = 0.
            kDM2 = 1.
            kDM3 = 0.
            iDM1 = 0 #dummy
            iDM3 = 0 #dummy
            
        
        # This is constructed such that weights and iz, iDM will work out
        # for all cases of the above. Note that only four of these terms at
        # most will ever be non-zero.
        Eth = self.thresholds[j, iz1, iDM1] * kz1 * kDM1 \
                + self.thresholds[j, iz1, iDM2] * kz1 * kDM2 \
                + self.thresholds[j, iz1, iDM3] * kz1 * kDM3 \
                + self.thresholds[j, iz2, iDM1] * kz2 * kDM1 \
                + self.thresholds[j, iz2, iDM2] * kz2 * kDM2 \
                + self.thresholds[j, iz2, iDM3] * kz2 * kDM3 \
                + self.thresholds[j, iz3, iDM1] * kz3 * kDM1 \
                + self.thresholds[j, iz3, iDM2] * kz3 * kDM2 \
                + self.thresholds[j, iz3, iDM3] * kz3 * kDM3 \
        
        # now account for beamshape
        Eth /= MCb

        # NOW GET snr
        # Eth=self.thresholds[j,k,l]/MCb
        Es = np.logspace(np.log10(Eth), lEmax + Emax_boost, 1000)
        PEs = self.vector_cum_lf(Es, Emin, Emax, gamma)
        PEs /= PEs[0]  # normalises: this is now cumulative distribution from 1 to 0
        r = np.random.rand(1)[0]
        iE1 = np.where(PEs > r)[0][-1]  # returns list starting at 0 and going upwards
        iE2 = iE1 + 1
        # iE1 should never be the highest energy, since it will always have a probability of 0 (or near 0 for Gamma)
        kE1 = (r - PEs[iE2]) / (PEs[iE1] - PEs[iE2])
        kE2 = 1.0 - kE1
        MCE = 10 ** (np.log10(Es[iE1]) * kE1 + np.log10(Es[iE2]) * kE2)
        MCs = MCE / Eth

        FRBparams = [MCz, MCDM, MCb, MCs, MCw]
        return FRBparams

    def build_sz(self):
        pass

    def update(self, vparams: dict, ALL=False, prev_grid=None):
        """Update the grid based on a set of input
        parameters
        
        Hierarchy:
        Each indent corresponds to one 'level'.
        This is used in the program control below
        to dictate how far each tree should proceed
        in calculation.
        Direct variable inputs are always listed first
        We see that sfr evolution and dm smearing
        lie just before the pdv step
        Hence, we deal with these first, and
        calc rates as a final step regardless
        of what else has changed.

        Args:
            vparams (dict):  dict containing the parameters
                to be updated and their values
            prev_grid (Grid, optional):
                If provided, it is assumed this grid has been
                updated on items that need not be repeated for
                the current grid.  i.e. Speed up!
            ALL (bool, optional):  If True, update the full grid
        
        calc_rates:
            calc_pdv
                Emin
                Emax
                gamma
                H0
                calc_thresholds
                    F0
                    alpha
                    bandwidth
            set_evolution
                sfr_n
                H0
            
            smear_grid
                grid
                mask
                    dmx_params (lmean, lsigma)
            dV
            zdm_grid
                H0
        
        Note that the grid is independent of the constant C (trivial dependence)

        Args:
            vparams (dict): [description]
        """
        warnings.warn("grid.update is deprecated, create a new instantiation instead", DeprecationWarning)

        # Init
        reset_cos, get_zdm, calc_dV = False, False, False
        smear_mask, smear_dm, calc_pdv, set_evol = False, False, False, False
        new_sfr_smear, new_pdv_smear, calc_thresh = False, False, False
        
        # if we are updating a grid, the MC will in-general need to be
        # re-initialised
        self.MCinit = False
        
        # Cosmology -- Only H0 so far
        if self.chk_upd_param("H0", vparams, update=True):
            reset_cos = True
            get_zdm = True
            calc_dV = True
            smear_dm = True
            calc_thresh = True
            calc_pdv = True
            set_evol = True
            new_sfr_smear = True

        # IGM
        if self.chk_upd_param("logF", vparams, update=True):
            get_zdm = True
            smear_dm = True
            # calc_thresh = False  # JMB
            # calc_pdv = False  # JMB
            # set_evol = False  # JMB
            new_sfr_smear = True

        # DM_host
        # IT IS IMPORTANT TO USE np.any so that each item is executed!!
        if np.any(
            [
                self.chk_upd_param("lmean", vparams, update=True),
                self.chk_upd_param("lsigma", vparams, update=True),
            ]
        ):
            smear_mask = True
            smear_dm = True
            new_sfr_smear = True

        # SFR?
        if self.chk_upd_param("sfr_n", vparams, update=True):
            set_evol = True
            new_sfr_smear = True  # True for either alpha_method
        if self.chk_upd_param("alpha", vparams, update=True):
            set_evol = True
            if self.state.FRBdemo.alpha_method == 0:
                calc_thresh = True
                calc_pdv = True
                new_pdv_smear = True
            elif self.state.FRBdemo.alpha_method == 1:
                new_sfr_smear = True

        ##### examines the 'pdv tree' affecting sensitivity #####
        # begin with alpha
        # alpha does not change thresholds under rate scaling, only spec index
        if np.any(
            [
                self.chk_upd_param("lEmin", vparams, update=True),
                self.chk_upd_param("lEmax", vparams, update=True),
                self.chk_upd_param("gamma", vparams, update=True),
            ]
        ):
            calc_pdv = True
            new_pdv_smear = True

        if self.chk_upd_param("DMhalo", vparams, update=True):
            # Update survey params
            self.survey.init_DMEG(vparams["DMhalo"])
            # NOTE: In future we can change this to not need to recalc every time
            self.survey.get_efficiency_from_wlist(self.survey.DMlist,self.survey.wlist,self.survey.wplist,model=self.survey.meta['WBIAS'])
            self.eff_table = self.survey.efficiencies

            calc_thresh = True
            calc_pdv = True
            new_pdv_smear = True

        # ###########################
        # NOW DO THE REAL WORK!!

        # TODO -- For cubes with multiple surveys can we do these
        #   first two steps (even the first 5!) only once??
        # Update cosmology?
        if reset_cos and prev_grid is None:
            cos.set_cosmology(self.state)
            cos.init_dist_measures()

        if get_zdm or ALL:
            if prev_grid is None:
                zDMgrid, zvals, dmvals = misc_functions.get_zdm_grid(
                    self.state,
                    new=True,
                    plot=False,
                    method="analytic",
                    save=False,
                    nz=self.zvals.size,
                    zmax=self.zvals[-1],
                    ndm=self.dmvals.size,
                    dmmax=self.dmvals[-1],
                    zlog=self.zlog,
                )
                self.parse_grid(zDMgrid, zvals, dmvals)
            else:
                # Pass a copy (just to be safe)
                self.parse_grid(
                    prev_grid.grid.copy(),
                    prev_grid.zvals.copy(),
                    prev_grid.dmvals.copy(),
                )

        if calc_dV or ALL:
            if prev_grid is None:
                self.calc_dV()
            else:
                self.dV = prev_grid.dV.copy()

        # Smear?
        if smear_mask or ALL:
            if prev_grid is None:
                self.smear = pcosmic.get_dm_mask(
                    self.dmvals,
                    (self.state.host.lmean, self.state.host.lsigma),
                    self.zvals,
                )
            else:
                self.smear = prev_grid.smear.copy()
        if smear_dm or ALL:
            if prev_grid is None:
                self.smear_dm(self.smear)
            else:
                self.smear = prev_grid.smear.copy()
                self.smear_grid = prev_grid.smear_grid.copy()

        if calc_thresh or ALL:
            self.calc_thresholds(
                self.F0,
                self.eff_table,
                bandwidth=self.bandwidth,
                weights=self.eff_weights,
            )
            
        if calc_pdv or ALL:
            self.calc_pdv()
        if set_evol or ALL:
            self.set_evolution()  # sets star-formation rate scaling with z - here, no evoltion...
        if new_sfr_smear or ALL:
            self.calc_rates()  # includes sfr smearing factors and pdv mult
        elif new_pdv_smear:
            self.rates = self.pdv * self.sfr_smear  # does pdv mult only, 'by hand'

        # Catch all the changes just in case, e.g. lCf
        # Can no longer do this because of repeat_grid
        self.state.update_params(vparams)

        return new_sfr_smear, new_pdv_smear, (get_zdm or smear_dm or calc_dV) # If either is true, need to also recalc repeater grids

    def chk_upd_param(self, param: str, vparams: dict, update=False):
        """ Check to see whether a parameter is
        differs from that in self.state

        Args:
            param (str): Paramter in question
            vparams (dict): Dict holding the value
            update (bool, optional): If True,
                update the value in self.state. 
                Defaults to False.

        Returns:
            bool: True if the parameter is different
        """
        updated = False
        DC = self.state.params[param]
        # In dict?
        if param in vparams.keys():
            # Changed?
            if vparams[param] != getattr(self.state[DC], param):
                updated = True
                if update:
                    self.state.update_param(param, vparams[param])
        #
        return updated
