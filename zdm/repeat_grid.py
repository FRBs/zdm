
"""
Class definition for repeating FRBs.

Input:
    A grid, as well as a time-per-field.
    We have two calculation methods: exact and MC.
    MCs take a very long time to converge on the average (guess:10^4 iterations?)
    Exact only calculates <singles> and <repeaters>
    Repetition parameters Rmin, Rmax, and Rgamma are stored in grid.state
        - Rmin: Minimum repetition rate of repeaters
        - Rmax: Maximum repetition rate of repeaters
        - Rgamma: *Differential* number of repeaters: dN/dR ~ R^Rgamma




Some notes regarding time dilation:
    
    #1: dVdtau
        grid.py includes the time-dilation effect, dVdtau, in grid.dV.
        We need to undo this: the number of repeaters does not change
        with dtau, just the rate per repeater. Hence, all instances
        of dV need to have an extra multiple of (1+z) attached, to undo
        the time-dilation effect.
    
    #2: Rmult. When alpha method == 1, we use "rate scaling". This means
            that the rate *per repeater* must change with frequency,
            since otherwise the number of progenitors is frequency-
            dependent, which is nonsense. This should be handled
            under "Rmult", and implies two corrections:
                - correct from nominal frequency to central frequency
                - correct for (1+z) factor
    
    #3: sfr factor
        grid.sfr, if alpha_method=0, includes the rate scaling with alpha
        However, here we use this to calculate the number of progenitors,
        thus we must calculate sfr from first principles if alpha_method=1.
        This is now handled by assigning self.use_sfr to be self.grid.sfr
        when alpha_method=0, or it is recalculated if alpha_method=1
        
"""

from zdm import grid
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from zdm import energetics
import mpmath
from scipy import interpolate
import time

# NZ is meant to toggle between calculating nonzero elements or not
# In theory, if False, it tries to perform calculations
# where there is no flux
# Currenty, it seems like setting it to False causes problems
NZ=True

# These constants represent thresholds below which we assume the chance
# to produce repetition is identically zero
LSRZ = -10. #Was causing problems at merely -6
SET_REP_ZERO=10**LSRZ # forces p(n=0) bursts to be unity when a maximally repeating FRB has SET_REP_ZERO expected events or less
# the above is crazy, the problem is it should zet p_zero + p_single to be unity, i.e. chance of double to be zero
# but how do we do this? presumably it's the total minus psingle?
#LZRZ = np.log10(SET_REP_ZERO)

#### sets energetics parameters #####
energetics.SplineMin = LSRZ
energetics.SplineMax = 6.
energetics.NSpline = int((energetics.SplineMax-energetics.SplineMin)*1000)


class repeat_Grid:
    """
    This class is designed to take a p(z,DM) grid and calculate the
    effects of repeating FRBs.
    
    It is an "add-on" to an existing grid class, i.e. it
    is passed an existing grid, and performs calculations
    from there.
    
    Repetition parameters are contained within the grid
    object itself. Optionally, these can be sepcified
    as additional arguments at compilation.
    
    Currently, only Poisson repeaters are enabled.
    This will change. Well, maybe...
    
    """
    
    
    def __init__(self,grid,Tfield=None,Nfields=None,opdir=None,Exact=True,
        MC=False,verbose=False,bmethod=1):
        """
        Initialises repeater class
        Args:
            grid: zdm grid class object
            Nfields: number of separate pointings by the survey
            Tfield: time spent on each field
            bmethod: int (1 or 2)
                1: beam represents solid angle viewed at each value of b,
                    for time Tfield
                2: beam represents time (in days) spent on any given source
                    at sensitivity level b. Tfield is solid angle. Nfields
                    then becomes a multiplier of the time.
        """
        
        # inherets key properties from the grid's state parameters
        self.state=grid.state
        self.grid=grid
        
        # these define the repeating population - repeaters with
        # rates between Rmin and Rmax with a power-law of Rgamma
        # dN(R)/dR ~ R**Rgamma
        self.Rmin=grid.state.rep.Rmin
        self.Rmax=grid.state.rep.Rmax
        self.Rgamma=grid.state.rep.Rgamma
        self.newRmin = True
        self.newRmax = True
        self.newRgamma = True
        
        # get often-used data from the grid - for simplicity
        self.Emin = 10**self.grid.state.energy.lEmin
        self.Emax = 10**self.grid.state.energy.lEmax
        self.gamma= self.grid.state.energy.gamma
        
        self.Rmults=None
        self.Rmult=None
        
        self.bmethod=bmethod
        
        # set these on init, to remember for update purposes
        self.Exact = Exact
        self.MC = MC
        
        # handles plotting
        if opdir is not None:
            #for plotting
            import os
            if not os.path.exists(opdir):
                os.mkdir(opdir)
            self.opdir=opdir
            doplots=True
        else:
            self.opdir=None
            doplots=False
        
        # redshift array
        self.zvals=self.grid.zvals
        
        # checks we have the necessary data to construct Nfields and Tfield
        if Nfields is not None:
            self.Nfields=Nfields
        elif grid.survey.meta.has_key('Nfields'):
            self.Nfields = grid.survey.meta['Nfields']
        else:
            self.Nfields = 1
            #raise ValueError("Nfields not specified")
        
        if Tfield is not None:
            self.Tfield=Tfield
        elif grid.survey.meta.has_key('Tfield'):
            self.Tfield = grid.survey.meta['Tfield']
        else:
            raise ValueError("Tfield not specified")
        
        # calculates constant Rc in front of dN/dR = Rc R^Rgamma
        # this needs to be updated if any of the repeat parameters change
        #self.Rc,self.NRtot=self.calc_constant(verbose=verbose)
        #grid.state.rep.RC=self.Rc
        # this now updated the parameters above
        self.calc_constant(verbose=verbose)
        
        if verbose:
            print("Calculated constant as ",self.Rc)
        
        # number of expected FRBs per volume in a given field
        self.Nexp_field = self.grid.dV*self.Rc
        # undoes the time-dilation effect, which is included in the FRB rate scaling
        self.Nexp_field *= (1.+self.grid.zvals) # previously was reduced by this amount
        self.Nexp = self.Nexp_field * self.Nfields
        
        # the below has units of Mpc^3 per zbin, and multiplied by p(DM|z) for each bin
        # Note that grid.dV already has a (1+z) time dilation factor in it
        # This is actually accounted for in the rate scaling of repeaters
        # Here, we want the actual volume, hence must remove this factor
        # recently removed this from sim_repeaters function
        self.tvolume_grid = (self.grid.smear_grid.T * (self.grid.dV * (1. + self.grid.zvals))).T
        #volume_grid *= Solid #accounts for solid angle viewed at that beam sensitivity
        #self.volume_grid = volume_grid
        
        
        # to remove alpha effect from grid.sfr if alpha_method==1
        if self.grid.state.FRBdemo.alpha_method==1:
            self.use_sfr = self.grid.source_function(self.grid.zvals,self.grid.state.FRBdemo.sfr_n)
        else:
            self.use_sfr = self.grid.sfr
        
        self.calc_Rthresh(Exact=Exact,MC=MC,doplots=doplots)

    def update(self,Rmin = None,Rmax = None,Rgamma = None):
        """
        Routine to update based upon new Rmin,Rmax,gamma parameters.
        It does *not* handle new grid parameters like Emin, Emax and so on.
        A to-do item will be to see if there is any fast way of applying
        those updates - currently, an entire new grid must be generated.
        
        Inputs:
            Rmin (float): Minimum repetition rate (per day)
            Rmax (float): Maximum repetition rate (per day)
            Rgamma (float): Differential power-law index
                of the repetition rate between Rmin and Rmax
        
        If the above are None, it assumes they have been left unchanged.
        """
        ### first check which have changed ###
        self.newRmin = False
        self.newRmax = False
        self.newRgamma = False
        
        
        if Rmin is not None and Rmin != self.Rmin:
            self.newRmin = True
            self.Rmin = Rmin
            self.grid.state.rep.Rmin = Rmin
        
        if Rmax is not None and Rmax != self.Rmax:
            self.newRmax = True
            self.Rmax = Rmax
            self.grid.state.rep.Rmax = Rmax
        
        if Rgamma is not None and Rgamma != self.Rgamma:
            self.newRgamma = True
            self.Rgamma = Rgamma
            self.grid.state.rep.Rgamma = Rgamma
        
        if not (self.newRmin or self.newRmax or self.newRgamma):
            # nothing has changed
            print("WARNING: updating repeat grid, but the parameters ",
                Rmin,Rmax,Rgamma," are not new")
            return
        
        ### do this if *any* parameters change ###
        # keep for later speed-ups
        oldRc = self.Rc
        self.calc_constant(verbose=False)
        
        # simple linear scaling of the number of repeaters per field
        self.Nexp *= self.Rc / oldRc
        self.Nexp_field *= self.Rc / oldRc
        
        ### everything up to here updates the main 'init' function
        # now proceed to "calc_Rthresh"
        # Rmult does *not* need to be re-calculated
        # In calc_Rthresh, it already checks if this
        # has already been done. Also with summed Rmult.
        # Thus calling calc_Rthresh should jump straight to
        # calling "sim repeaters". Everything in
        # calc_Rthresh after that is necessary if anything changes
        # hence: just calling calc_Rthresh is good!
        # if we are updating, never do plots, waste of time!
        self.calc_Rthresh(Exact=self.Exact,MC=self.MC,doplots=False)
        
        
    def calc_constant(self,verbose=False):
        """
        Calculates the constant of the repeating FRB function:
            d\Phi(R)/dR = Cr * (R/R0)^power
            between Rmin and Rmax.
            Here R0 is the unit of R, taken to be 1/day
        
        The constant C is the constant burst rate at energy E0 per unit volume
        By definition, \int_{R=Rmin}^{R=Rmax} R d\Phi(R)/dR dR = C
        Hence, C=Cr * (power+2)^-1 (R_max^(power+2)-R_min^(power+2))
        and thus Cr = C * (power+2)/(R_max^(power+2)-R_min^(power+2))
        
        We also need to account for frequency dependence.
        The default frequency for FRB properties is 1.3 GHz.
        Under the rate approximation, the rate per repeater changes with frequency
        Under the spectral index approximation, FRBs become brighter/dimmer.
        This MUST therefore factor into calculations!
        
        NOTE: if updating, in theory all steps before Rc = ... could be held in memory
        
        """
        
        # sets repeater constant as per global FRB rate constant
        # C is in bursts per Gpc^3 per year
        # rate is thus in bursts/year
        C=10**(self.grid.state.FRBdemo.lC)
        if verbose:
            print("Initial burst rate above ",self.Emin," is ",C*1e9*365.25," per Gpc^-3 per year")
        
        # account for repeat parameters being defined at a different energy than Emin
        fraction = self.grid.vector_cum_lf(self.state.rep.RE0,self.Emin,self.Emax,self.gamma)
        
        C = C*fraction
        
        if verbose:
            print("This is ",C*1e9*365.25," Gpc^-3 year^-1 above ",self.state.rep.RE0," erg")
        
        # The total number of bursts, C, is \int_rmin^rmax R dN/dR dR
        # = \int_rmin^rmax Rc * R^(Rgamma+1) = Rc * (Rmax^(Rgamma+2) - Rmin^(Rgamma+2))/(Rgamma+2)
        # We solve for Rc by inverting this below
        Rc = C * (self.Rgamma+2.)/(self.Rmax**(self.Rgamma+2.)-self.Rmin**(self.Rgamma+2.))
        
        # total number of repeaters
        Ntot = (Rc/(self.Rgamma+1)) * (self.Rmax**(self.Rgamma+1)-self.Rmin**(self.Rgamma+1))
        
        if verbose:
            print("Corresponding to ",Ntot," repeaters every cubic Mpc")
        
        self.Rc = Rc
        self.NRtot = Ntot
        self.grid.state.rep.RC = Rc
        return Rc,Ntot
    
    
    def calc_Rthresh(self,Exact=True,MC=False,Pthresh=0.01,doplots=False,old=[False,False,False]):
        """
        For FRBs on a zdm grid, calculate the repeater rates below
        which we can reasonably treat the distribution as continuous
        
        Steps:
            - Calculate the apparent repetition threshold to ignore repetition
            - Calculate rate scaling between energy at which repeat rates
              are defined, and energy threshold
        
        Exact: if True, performs an "exact" analytic calculation of
            the singles and repeater rates.
        
        MC (bool or int): if bool and True, perform a single MC evaluation
            if bool and False, use the analytic method
            if int >0: performs MC Monte Carlo evaluations 
        
        Pthresh: only if MC is true
            threshold above which to allow progenitors to produce multilke bursts
        
        Old defines which parameters (Rmin, Rmax, gamma) are being redone
        
        """
        
        
        # calculates rate corresponding to Pthresh
        # P = 1.-(1+R) * np.exp(-R)
        # P = 1.-(1+R) * (1.-R + R^2/2) = 1 - [1 +R -R - R^2 + R^2/2 + cubic]
        # P = 0.5 R^2 # assumes R is small to get
        # R = (2P)**0.5
        Rthresh = (2.*Pthresh)**0.5 #only works for small rates
        # we loop over beam values. Sets up Rmults array to hold these
        self.Nbeams = self.grid.beam_b.size
        # create a list of rate multipliers
        if self.Rmults is None:
            
            nb = self.grid.beam_b.size
            nz = self.grid.zvals.size
            ndm = self.grid.dmvals.size
            
            # create empty arrays for saving for later
            self.Rmults = np.zeros([nb,nz,ndm])
            if self.bmethod==2: # we have T(B), not Omega(B)
                self.avals=[None]
                self.bvals=[None]
                self.snorms1=[None]
                self.snorms2=[None]
                self.znorms1=[None]
                self.znorms2=[None]
                self.NotTooLows=[None]
                self.NotTooLowbs=[None]
                self.TooLow=[None]
                self.nonzeros=[None]
            else: # We have Omega(B) for fixed T.
                self.avals=[None]*self.grid.beam_b.size
                self.bvals=[None]*self.grid.beam_b.size
                self.snorms1=[None]*self.grid.beam_b.size
                self.snorms2=[None]*self.grid.beam_b.size
                self.znorms1=[None]*self.grid.beam_b.size
                self.znorms2=[None]*self.grid.beam_b.size
                self.nonzeros=[None]*self.grid.beam_b.size
                self.NotTooLows=[None]*self.grid.beam_b.size
                self.NotTooLowbs=[None]*self.grid.beam_b.size
                self.TooLow=[None]*self.grid.beam_b.size
            
            for ib,b in enumerate(self.grid.beam_b):
                if self.bmethod==1:
                    time=self.Tfield # here, time is total time on field
                else:
                    time=self.grid.beam_o[ib]*self.Nfields # here, o is time on field, not solid angle
                Rmult=self.calcRmult(b,time)
                # keeps a record of this Rmult, and sets the current value
                self.Rmults[ib,:,:] = Rmult
            if self.bmethod==2:
                self.summed_Rmult = np.sum(self.Rmults,axis=0) # just sums the multipliers over the beam axis
        
        if self.bmethod==2:
            self.Nth=0
            b=None # irrelevant value
            solid_unit = self.Tfield # this value is "per steradian" factor, since beam is time
            exactset,MCset = self.sim_repeaters(Rthresh,b,solid_unit,doplots=doplots,Exact=Exact,
                    MC=MC,Rmult=self.summed_Rmult)
            if Exact:
                exact_singles,exact_zeroes,exact_rep_bursts,exact_reps = exactset
            if MC:
                expNrep,Nreps,single_array,mult_array,summed_array,exp_array,\
                    poisson_rates,numbers = MCset
        else:
            for ib,b in enumerate(self.grid.beam_b):
                self.Nth=ib # sets internal logging for faster recalculation
                exactset,MCset = self.sim_repeaters(Rthresh,b,self.grid.beam_o[ib],
                    doplots=doplots,Exact=Exact,MC=MC,Rmult=self.Rmults[ib])
                if ib==0:
                    if Exact:
                        exact_singles,exact_zeroes,exact_rep_bursts,exact_reps = exactset
                    if MC:
                        expNrep,Nreps,single_array,mult_array,summed_array,exp_array,\
                            poisson_rates,numbers = MCset
                else:
                    if Exact:
                        exact_singles += exactset[0]
                        exact_zeroes  += exactset[1]
                        exact_rep_bursts  += exactset[2]
                        exact_reps += exactset[3]
                    if MC:
                        expNrep += MCset[0]
                        Nreps += MCset[1]
                        single_array += MCset[2]
                        mult_array += MCset[3]
                        summed_array += MCset[4]
                        exp_array += MCset[5]
                        poisson_rates += MCset[6]
                        numbers += MCset[7]
        
        if Exact:
            # sets self values of these arrays
            self.exact_singles = exact_singles
            self.exact_zeroes = exact_zeroes
            self.exact_rep_bursts = exact_rep_bursts
            self.exact_reps = exact_reps
        
        if MC:
            self.MC_expprogenitors = expNrep
            self.MC_Nprogenitors = Nreps
            self.MC_singles = single_array
            self.MC_reps = mult_array
            self.MC_rep_bursts = summed_array
            self.MC_exp_bursts = exp_array
            self.MC_poisson_rates = poisson_rates
            self.MC_numbers = numbers
            
        
        if MC:
            ####### we now make summary plots of the total number of bursts as a function of redshift
            # the total single bursts
            # and the total from repeaters
            
            # Initially, Poisson is unweighted by constants or observation time
            # We now need to multiply by Tobs and the constant
            #Poisson *= self.Tfield * 10**(self.grid.state.FRBdemo.lC)
            TotalSingle = poisson_rates + single_array
            single_array + summed_array
            total_bursts = TotalSingle + summed_array # single bursts plus bursts from repeaters
            
        # calculates the expected number of bursts in the no-repeater case from grid info
        no_repeaters = self.grid.rates * self.Tfield * 10**(self.grid.state.FRBdemo.lC)
        
    def perform_exact_calculations(self,slow=False):
        """
        Performs exact calculations of:
            - <single bursts>
            - <no bursts>
            - < progenitors with N>2 bursts>
            - < number of N>2 bursts>
        """
        exact_singles = self.calc_singles_exactly()
        exact_zeroes = self.calc_zeroes_exactly(exact_singles)
        exact_rep_bursts = self.calc_expected_repeat_bursts(exact_singles)
        exact_reps = self.calc_expected_repeaters(exact_singles,exact_zeroes)
        
        # the below is a simple slow loop which is good for test purposes
        # it does not use spline interpolation
        # testing so results are good to ~10^-12 differences
        if slow:
            s,z,m,n,t=self.slow_exact_calculation(exact_singles,exact_zeroes,exact_rep_bursts,exact_reps)
        
        exact_singles = exact_singles*self.volume_grid
        exact_zeroes = exact_zeroes*self.volume_grid
        exact_rep_bursts = exact_rep_bursts*self.volume_grid
        exact_reps = exact_reps*self.volume_grid
        
        exact_singles = (exact_singles.T * self.use_sfr).T
        exact_zeroes = (exact_zeroes.T * self.use_sfr).T
        exact_rep_bursts = (exact_rep_bursts.T * self.use_sfr).T
        exact_reps = (exact_reps.T * self.use_sfr).T
        
        return exact_singles,exact_zeroes,exact_rep_bursts,exact_reps
    
    def calc_expected_repeaters(self,expected_singles,expected_zeros):
        """
        Calculates the expected number of FRBs observed as repeaters.
        The total expected number of bursts from
        """
        
        # The probability of any repeater giving more than one burst is
        # 1-p(1)-p(0)
        # this is \int_Rmin^Rmax R^Rgamma [1.- R' exp(-R') - exp(-R')] dR
        # the latter two terms have already been calculated
        # the first term is analytic
        # = \int R^Rgamma dR - p(singles) - p(mults)
        
        effGamma=self.Rgamma+1.
        total_repeaters = self.Rc * (1./effGamma) * (self.Rmax**effGamma-self.Rmin**effGamma)
        expected_repeaters = total_repeaters - expected_singles - expected_zeros
        # do we need to artificially set some regions to zero, based on
        # previous cuts?
        
        # sometimes this evaluates to less than zero due to numerical errors
        themin = np.min(expected_repeaters)
        shape = expected_repeaters.shape
        zero = np.where(expected_repeaters.flatten() < 0)[0]
        expected_repeaters = expected_repeaters.flatten()
        expected_repeaters[zero] = 0.
        expected_repeaters[self.TooLow[self.Nth]] = 0. # to account for floating errors
        expected_repeaters = expected_repeaters.reshape(shape)
        
        TOO_SMALL = -1e-6
        if themin < TOO_SMALL:
            print("WARNING: found significantly small value ",themin," in expected repeaters")
        
        return expected_repeaters
    
    def calc_expected_repeat_bursts(self,expected_singles):
        """
        Calculates the expected total number of bursts from repeaters.
        
        This is simply calculated as <all bursts> - <single bursts>
        """
        
        a = self.Rmin*self.Rmult
        b = self.Rmax*self.Rmult
        effGamma=self.Rgamma+2
        total_rate = self.Rc * (1./effGamma) * (self.Rmax**effGamma-self.Rmin**effGamma) * self.Rmult
        mult_rate = total_rate - expected_singles
        return mult_rate
        
    
    def calc_singles_exactly(self):
        """
        Calculates exact expected number of single bursts from a repeater population
        
        Probability is: \int constant * R exp(-R) * R^(Rgamma)
        definition of gamma function is R^x-1 exp(-R) for gamma(x)
        # hence here x is gamma+2
        limits set by Rmin and Rmax (determind after multiplying intrinsic by Rmult)
        
        This is Gamma(Rgamma+2,Rmin) - Gamma(Rgamma+2,Rmax)
        """
        # We wish to integrate R R^gammaR exp(-R) from Rmin to Rmax
        # this can be done by mpmath.gammainc(self.Rgamma+2, a=self.Rmin*Rmult[i,j])
        # which integrates \int_Rmin*Rmult ^ infinity R(Rgamma+2-1) exp(-R)
        # and subtracting the Rmax from it
        
        global SET_REP_ZERO
        
        nz,ndm=self.Rmult.shape
        
        effGamma=self.Rgamma+2
        if effGamma not in energetics.igamma_splines.keys():
            energetics.init_igamma_splines([effGamma])
        # nonzero may or may not be a shortcut
        global NZ
        # get list of indices we bother to operate on
        # we proceed to calculate avals, bvals, and norms using these only
        # only at the end do we re-insert this
        if NZ:
            if self.nonzeros[self.Nth] is not None:
                nonzero = self.nonzeros[self.Nth]
            else:
                nonzero = np.where(self.Rmult.flatten() > 1e-20)[0]
                self.nonzeros[self.Nth]=nonzero
        # the following is for saving time when updating
        if self.newRmin == False:
            avals=self.avals[self.Nth]
        else:
            if NZ:
                avals=self.Rmin*self.Rmult.flatten()[nonzero]
            else:
                avals=self.Rmin*self.Rmult.flatten()
            self.avals[self.Nth] = avals
            
        if self.newRmax == False:
            bvals=self.bvals[self.Nth]
        else:
            if NZ:
                bvals=self.Rmax*self.Rmult.flatten()[nonzero]
            else:
                bvals=self.Rmax*self.Rmult.flatten()
            self.bvals[self.Nth] = bvals
        
        # We now correct avals for values which are too low
        # There are three cases - bvals too low, only avals too low,
        # and neither.
        # Each has a slightly different calculation method.
        # can be common that both are too low
        
        if self.newRmin == False:
            NotTooLow = self.NotTooLows[self.Nth]
            NTL = self.NTL
        else:
            NotTooLow = np.where(avals > SET_REP_ZERO)[0]
            self.NotTooLows[self.Nth] = NotTooLow
            if len(NotTooLow) > 0:
                NTL = True
            else:
                NTL = False
            self.NTL = NTL
        
        if self.newRmax == False:
             NotTooLowb = self.NotTooLowbs[self.Nth]
             NTLb = self.NTLb
        else:
            NotTooLowb = np.where(bvals > SET_REP_ZERO)[0]
            self.NotTooLowbs[self.Nth] = NotTooLowb
            if len(NotTooLowb) > 0:
                NTLb = True
            else:
                NTLb = False
            self.NTLb = NTLb
            
            # gets regions which are too low everywhere
            TooLow = np.where(bvals <= SET_REP_ZERO)[0]
            self.TooLow[self.Nth] = TooLow
        
        # technically, we now never save norms 1...
        # now do calculations. Must be re-done *every* time
        # this is inefficient, I should find a way to split up the different segments
        # currently, NotTooLow is a subset of NotTooLowb. I could/should make it independent.
        if self.newRmin == False and self.newRgamma == False and self.newRmax == False:
            norms1 = self.snorms1[self.Nth]
        else:
            norms1 = -avals**effGamma / effGamma # this is the "too low" value
            analytic = (SET_REP_ZERO**effGamma - avals[NotTooLowb]**effGamma)/effGamma
            if NTLb:
                norms1[NotTooLowb] = interpolate.splev([SET_REP_ZERO], energetics.igamma_splines[effGamma])
                norms1[NotTooLowb] += analytic
            if NTL:
                norms1[NotTooLow] = interpolate.splev(avals[NotTooLow], energetics.igamma_splines[effGamma])
            
            self.snorms1[self.Nth] = norms1
        # now do calculations
        if self.newRmax == False and self.newRgamma == False:
            norms2 = self.snorms2[self.Nth]
        else:
            norms2 = -bvals**effGamma / effGamma
            if NTLb:
                norms2[NotTooLowb] = interpolate.splev(bvals[NotTooLowb], energetics.igamma_splines[effGamma])
            self.snorms2[self.Nth] = norms2
        # subtract components
        norms = norms1 - norms2
        
        # integral is in units of R'=R*Rmult
        # however population is specified in number density of R
        # hence R^gammadensity factor must be normalised
        # the following array could also be saved, we shall see
        if NZ:
            norms /= self.Rmult.flatten()[nonzero]**(self.Rgamma+1) 
        else:
            norms /= self.Rmult.flatten()**(self.Rgamma+1)
            
        # multiplies this by the number density of repeating FRBs
        norms *= self.Rc
        
        # get rid of negative parts - might come from random floating point errors
        themin = np.min(norms)
        if themin < -1e-20:
            print("Significant negative value found in singles",themin)
        zero = np.where(norms < 0.)[0]
        norms[zero]=0.
        
        #we create a zero array, which is mostly zero due to Rmult being very low.
        if NZ:
            tempnorms = np.zeros([nz*ndm])
            tempnorms[nonzero] = norms
            norms = tempnorms.reshape([nz,ndm])
        else:
            norms = norms.reshape([nz,ndm])
        
        return norms
        
    def calc_zeroes_exactly(self,singles):
        """
        Calculates the probability of observing zero bursts exactly
        
        probability is: constant * exp(-R) * R^(Rgamma)
        definition of gamma function is R^x-1 exp(-R) for gamma(x)
        # hence here x is gamma+1
        limits set by Rmin and Rmax (determind after multiplying intrinsic by Rmult)
        
        This is Gamma(Rgamma+1,Rmin) - Gamma(Rgamma+1,Rmax)
        """
        global SET_REP_ZERO
        # We wish to integrate R^gammaR exp(-R) from Rmin to Rmax
        # this can be done by mpmath.gammainc(self.Rgamma+1, a=self.Rmin*Rmult[i,j])
        # which integrates \int_Rmin*Rmult ^ infinity R(Rgamma+1-1) exp(-R)
        # and subtracting the Rmax from it
        
        nz,ndm=self.Rmult.shape
        
        effGamma=self.Rgamma+1
        if effGamma not in energetics.igamma_splines.keys():
            energetics.init_igamma_splines([effGamma])
        
        # here we 'know' that avals and bvals have been calculated in
        # the routine "calc_singles_exactly"
        # hence we use those
        # recall: norms1 is the term involving avals
        # norms2 the term involving bvals
        # when using splines, need spline(a) - spline(b)
        #   = \int_a^inf - \int_b^\inf
        #    when analytics, need \int_a^b f' = f(b)-f(a)
        
        avals = self.avals[self.Nth]
        bvals = self.bvals[self.Nth]
        
        NotTooLow = self.NotTooLows[self.Nth]
        NotTooLowb = self.NotTooLowbs[self.Nth]
        NTL = self.NTL
        NTLb = self.NTLb
        
        global NZ
        if NZ:
            nonzero = self.nonzeros[self.Nth]
        
        # now do calculations for a (lower bound)
        if self.newRmin == False and self.newRgamma == False  and self.newRmax == False:
            norms1 = self.znorms1[self.Nth]
        else:
            norms1 = -avals**effGamma / effGamma
            analytic = (SET_REP_ZERO**effGamma - avals[NotTooLowb]**effGamma)/effGamma
            
            if NTLb:
                norms1[NotTooLowb] = interpolate.splev([SET_REP_ZERO], energetics.igamma_splines[effGamma])
                norms1[NotTooLowb] += analytic
            # the below over-writes the above
            if NTL:
                norms1[NotTooLow] = interpolate.splev(avals[NotTooLow], energetics.igamma_splines[effGamma])
            self.znorms1[self.Nth] = norms1
            
        
        # now do calculations
        if self.newRmax == False and self.newRgamma == False:
            norms2 = self.znorms2[self.Nth]
        else:
            norms2 = -bvals**effGamma / effGamma
            if NTLb:
                norms2[NotTooLowb] = interpolate.splev(bvals[NotTooLowb], energetics.igamma_splines[effGamma])
            self.znorms2[self.Nth] = norms2
        
        # integral is in units of R'=R*Rmult
        # however population is specified in number density of R
        # hence R^gammadensity factor must be normalised
        norms = norms1 - norms2
        norms *= self.Rc
        
        if NZ:
            norms /= self.Rmult.flatten()[nonzero]**(effGamma) # gamma due to integrate, one more due to dR
        else:
            norms /= self.Rmult.flatten()**(effGamma)
        
        # get rid of negative parts - might come from random floating point errors
        themin = np.min(norms)
        zero = np.where(norms < 0.)[0]
        norms[zero]=0.
        if themin < -1e-20:
            print("Significant negative value found in zeroes",themin)
        
        # the problem here is that when Rmult is zero, we need to ensure that all the FRBs
        # are detected as such
        everything = self.Rc * (1./effGamma) * (self.Rmax**effGamma-self.Rmin**effGamma)
        
        if NZ:
            tempnorms = np.full([nz*ndm],everything) # by default, 100% are detected zero times
            tempnorms[nonzero] = norms
            norms=tempnorms.reshape([nz,ndm])
        else:
            norms = norms.reshape([nz,ndm])
        
        return norms
        
    def slow_exact_calculation(self,exact_singles,exact_zeroes,exact_rep_bursts,exact_reps,plot=True,zonly=True):
        """
        Calculates exact expected number of single bursts from a repeater population
        
        Probability is: \int constant * R exp(-R) * R^(Rgamma)
        definition of gamma function is R^x-1 exp(-R) for gamma(x)
        # hence here x is gamma+2
        limits set by Rmin and Rmax (determind after multiplying intrinsic by Rmult)
        
        This is Gamma(Rgamma+2,Rmin) - Gamma(Rgamma+2,Rmax)
        """
        # We wish to integrate R R^gammaR exp(-R) from Rmin to Rmax
        # this can be done by mpmath.gammainc(self.Rgamma+2, a=self.Rmin*Rmult[i,j])
        # which integrates \int_Rmin*Rmult ^ infinity R(Rgamma+2-1) exp(-R)
        # and subtracting the Rmax from it
        
        nz,ndm=self.Rmult.shape
        #norms=np.zeros([nz,ndm])
        
        effGamma=self.Rgamma+2
        
        # a slow test. prints a whole bunch of junk
        s = np.zeros([nz,ndm])
        z = np.zeros([nz,ndm])
        m = np.zeros([nz,ndm])
        n = np.zeros([nz,ndm])
        t = np.zeros([nz,ndm])
        # this is the long version, the above is the array version
        for i in np.arange(nz):
            if not zonly:
                print("Performing exact calculation for iteration ",i," of ",nz)
            for j in np.arange(ndm):
                a=self.Rmin*self.Rmult[i,j]
                b=self.Rmax*self.Rmult[i,j]
                
                zero = mpmath.gammainc(self.Rgamma+1, a=a, b=b)/(self.Rmult[i,j]**(self.Rgamma+1))
                single = mpmath.gammainc(effGamma, a=a, b=b)/(self.Rmult[i,j]**(self.Rgamma+1))
                nrep = (1./(self.Rgamma+1)) * (self.Rmax**(self.Rgamma+1)-self.Rmin**(self.Rgamma+1))
                nrep = nrep - single - zero
                total = (1./effGamma) * (b**effGamma-a**effGamma)/(self.Rmult[i,j]**(self.Rgamma+1))
                mult = total-single
                t[i,j]=total*self.Rc
                s[i,j]=single*self.Rc
                z[i,j]=zero*self.Rc
                n[i,j]=nrep*self.Rc
                m[i,j]=mult*self.Rc
                print(i,j,self.grid.zvals[i],n[i,j],exact_reps[i,j],n[i,j]-exact_reps[i,j])
                #norms2[i,j] = float(mpmath.gammainc(effGamma, a=a, b=b))
                if zonly:
                    break
        
        if plot:
            self.do_2D_plot(exact_singles-s,self.opdir+'diffs'+'.pdf',log=False,zmax=9)
            self.do_2D_plot(exact_zeroes-z,self.opdir+'diffz'+'.pdf',log=False,zmax=9)
            self.do_2D_plot(exact_reps-n,self.opdir+'diffn'+'.pdf',log=False,zmax=9)
            self.do_2D_plot(exact_rep_bursts-m,self.opdir+'diffm'+'.pdf',log=False,zmax=9)
            
            self.do_2D_plot(np.abs((exact_singles-s)/s),self.opdir+'fdiffs'+'.pdf',log=False,zmax=9)
            self.do_2D_plot(np.abs((exact_zeroes-z)/z),self.opdir+'fdiffz'+'.pdf',log=False,zmax=9)
            self.do_2D_plot(np.abs((exact_reps-n)/n),self.opdir+'fdiffn'+'.pdf',log=False,zmax=9)
            self.do_2D_plot(np.abs((exact_rep_bursts-m)/m),self.opdir+'fdiffm'+'.pdf',log=False,zmax=9)
            
            self.do_2D_plot(s,self.opdir+'slows'+'.pdf',log=False,zmax=9)
            self.do_2D_plot(z,self.opdir+'slowz'+'.pdf',log=False,zmax=9)
            self.do_2D_plot(n,self.opdir+'slown'+'.pdf',log=False,zmax=9)
            self.do_2D_plot(m,self.opdir+'slowm'+'.pdf',log=False,zmax=9)
        
        return s,z,m,n,t

    def sim_repeaters(self,Rthresh,beam_b,Solid,doplots=False,Exact=True,MC=False,Rmult=None):
        # contains threshold information as function of FRB width, zvals, DMvals)
        # approximate this only as z x DM for now
        """
        Simulates once-off and repeat bursts for a given value of b (sensitivity)
        Rthresh: threshold in ergs on a z-DM grid
        beam_b: value of beam for this calculation
        solid: solid angle corresponding to beam value b
        MC: if True, performs MC evaluation. Skips if False
        doplots: plots a bunch of stuff if True
        
        """
        # should always be defined - this is aa historical just-in-case
        if Rmult is None:
            Rmult=self.calcRmult(beam_b,self.Tfield)
            # keeps a record of this Rmult, and sets the current value
            #self.Rmults.append(Rmult)
        self.Rmult = Rmult
        
        self.volume_grid = self.tvolume_grid*Solid
        
        if doplots:
            self.do_2D_plot(Rmult,self.opdir+'Rmult_'+str(beam_b)[0:5]+'.pdf',clabel='log$_{10}$ rate multiplier')
        
        if Exact:
            #exact_singles_rate=self.calc_singles_exactly(Rmult)
            exact_singles,exact_zeroes,exact_rep_bursts,exact_reps=self.perform_exact_calculations()
            exact_set = [exact_singles,exact_zeroes,exact_rep_bursts,exact_reps]
        else:
            exact_set = None
        
        if doplots and Exact:
            self.do_2D_plot(exact_singles,self.opdir+'exact_singles_'+str(beam_b)[0:5]+'.pdf',clabel='log$_{10}$ rate')
            self.do_2D_plot(exact_zeroes,self.opdir+'exact_zeroes_'+str(beam_b)[0:5]+'.pdf',clabel='log$_{10}$ rate')
            self.do_2D_plot(exact_rep_bursts,self.opdir+'exact_rep_bursts_'+str(beam_b)[0:5]+'.pdf',clabel='log$_{10}$ rate')
            self.do_2D_plot(exact_reps,self.opdir+'exact_reps_'+str(beam_b)[0:5]+'.pdf',clabel='log$_{10}$ rate',lrange=6)
            total=exact_rep_bursts+exact_singles
            self.do_2D_plot(total,self.opdir+'exact_all_bursts_'+str(beam_b)[0:5]+'.pdf',clabel='log$_{10}$ rate')
            
            self.do_z_plot([total,exact_singles,exact_rep_bursts],
                self.opdir+'zproj_exact_components_'+str(beam_b)[0:5]+'.pdf',
                label=['Total','One-off FRBs','bursts from repeaters'])
            self.do_z_plot(total-self.grid.rates*self.Tfield*10**(self.grid.state.FRBdemo.lC),
                self.opdir+'zproj_exact_difference_'+str(beam_b)[0:5]+'.pdf',log=False,
                label='difference with rates')
            
        
        if MC:
            # in the regions below, everything gets counted as individual bursts
            #no_rep_region = np.where(newRmult * self.Rmax > Rthresh)
            #Rmult.flatten()[no_rep_region]=1000.
            # if Rmult is large, then the relevant threshold for producing many bursts must reduce
            dmzRthresh = Rthresh/Rmult
            dmzRthresh = dmzRthresh.flatten()
            independent = np.where(dmzRthresh > self.Rmax) #100% independent
            stochastic = np.where(dmzRthresh < self.Rmin) #0% repeaters
            dmzRthresh[independent]=self.Rmax
            dmzRthresh[stochastic]=self.Rmin
            
            #dmzRthresh[:]=self.Rmin # makes everything stochastic. Ouch!
            dmzRthresh = dmzRthresh.reshape([self.grid.zvals.size,self.grid.dmvals.size])
            
            # now estimates total rate from bursts with R < dmzRthresh
            # this is integrating R dR from dmzRthresh to Rmax
            Rfraction = (self.Rmax**(self.Rgamma+2.)-dmzRthresh**(self.Rgamma+2.))/(self.Rmax**(self.Rgamma+2.)-self.Rmin**(self.Rgamma+2.))
            Poisson = 1.-Rfraction
            # currently, 
            poisson_rates = self.grid.rates * Poisson * 10**(self.grid.state.FRBdemo.lC) * Solid * self.Tfield
            #print("Number of single FRBs from insignificant FRBs ",np.sum(poisson_rates)) #FRBs per 2000 dats per steradian
        
            # returns zdm grid with values being number of single bursts, number of repeaters, number of repeat bursts
            expNrep,Nreps,single_array,mult_array,summed_array,exp_array,numbers = self.MCsample(dmzRthresh,Rmult,doplots,tag='_b'+str(beam_b)[0:5])
            
            MCset=[expNrep,Nreps,single_array,mult_array,summed_array,exp_array,poisson_rates,numbers]
        else:
            MCset=None
        results=[exact_set,MCset]
        
        
        return results
        
        ########## IGNORE BELOW #########
        # now adds in the probability of getting a single burst from these FRBs
        # this is a weighted integral according to p(0 or 1) vs p(2 or more)
        # p(0 or 1) is (1+R) * np.exp(-R)
        # thus we have for every single cell the integral
        # \int_Rthresh^Rmax p(0 or 1) 
        # all bursts: \int_Rmin^Rmax dr R dN(R)/dR dR
        # fraction above: \int_Rmin^thresh dr R dN(R)/dR dR
        # now perform \int_thresh^Rmax dr p(1) dN(R)/dR dR
        # normally, for weak repeaters, R = 0*p(0) + 1*p(1) = p(1) i.e. it's the same!
        # here: nope
        # thus \int_thresh^Rmax dr R exp(-R) dN(R)/dR dR
        # This evaluates to -Gamma(Rgamma+2,Rmax) + Gamma(Rgamma+2,Rthresh)
        # Gamma is the upper incomplete gamma function
        
    
    def calcRmult(self,beam_b=1,time_days=1):
        """
        beam_b: value of sensitivity
        time: time spent on this pointing
        
        We also need to account for the rate scaling if it exists.
        """
        # calculates Rmult over a range of burst widths and probabilities
        #print("Calculating Rmult using ",beam_b)
        for iw,w in enumerate(self.grid.eff_weights):
            # we want to calculate for each point an Rmult such that
            # Rmult_final = \sum wi Rmulti
            # does this make sense? Effectively its the sum of rates. Well yes it does! AWESOME!
            # numerator for Rmult for this width
            #Note: grid.thresholds already will include effect of alpha
            if iw==0:
                Rmult = w*self.grid.array_cum_lf(self.grid.thresholds[iw,:,:]/beam_b,self.Emin,self.Emax,self.gamma)
            else:
                Rmult += w*self.grid.array_cum_lf(self.grid.thresholds[iw,:,:]/beam_b,self.Emin,self.Emax,self.gamma)
        Rmult /= self.grid.vector_cum_lf(self.state.rep.RE0,self.Emin,self.Emax,self.gamma)
        
        # calculates the expectation value for a single pointing
        # rates were "per day", now "per pointing time on field"
        Rmult *= time_days
        # accounts for time dilation of intrinsic rate
        dilation=1./(1.+self.grid.zvals)
        # multiplies repeater rates by both base rate, and 1+z penalty
        # NEW NEW NEW NEW
        if self.grid.state.FRBdemo.alpha_method==1:
            # scales to frequency of interest
            fscale = (self.grid.nuObs/self.grid.nuRef)**-self.grid.state.energy.alpha
            Rmult *= fscale
            #double negative here: dilation is 1/(1+z)
            # hence if rate goes as f^-alpha, f goes as (1+z), then we recover (1/1+z)**alpha
            dilation = dilation**(1.+self.grid.state.energy.alpha)
        Rmult = (Rmult.T * dilation).T
        return Rmult
    
    def MCsample(self,Rthresh,Rmult,doplots=False,tag=None):
        """
        Samples FRBs from Rthresh upwards
        
        Rmult scales a repeater rate to an expected Nevents rate
        
        Rthresh alrady has the Tfield baked into it.
        Rmult also has the Tfield baked into it.
        
        volume_grid should be in units of Mpc^3, accounting for solid angle and dV/dz/dOmega
        
        tag: only used if plotting
        
        """
        
        # Number of repeaters:
        # int Rc * dN/dR dR from Rthresh to Rmax
        EffRc = self.Rc * (self.Rgamma+1)**-1 * (self.Rmax**(self.Rgamma+1) - Rthresh**(self.Rgamma+1))
        # units above: number per cubic Mpc
        # multiply by cubic Mpc
        
        #Written this way so that it can accept ints or floats
        if isinstance(self.MC,bool):
            MCmult = 1.
        else:
            MCmult = self.MC
        
        
        # number of repeaters should NOT have time but SHOULD have solid angle
        expected_number_of_repeaters = EffRc * self.volume_grid * MCmult
        
        # if the below is the case, then "grid" will already have included
        # this as a volumetric effect: both scaling with z, and to reference
        # frequency. We DO NOT want this!
        # But we then do need to adjust to Rmult...
        expected_number_of_repeaters = (expected_number_of_repeaters.T * self.use_sfr).T
        
        # expected number of repeaters * mean reps/repeater should equal expected bursts
        # mean repeater rate is R dN/dR / dN/dR
        mean_rate = ((self.Rgamma+2)**-1 *(self.Rmax**(self.Rgamma+2) - Rthresh**(self.Rgamma+2)) )
        mean_rate = mean_rate / ((self.Rgamma+1)**-1 *(self.Rmax**(self.Rgamma+1) - Rthresh**(self.Rgamma+1)))
        
        #bad=np.where(Rthresh.flatten()==self.Rmax)[0]
        
        #mean_rate.flatten()[bad]=0.
        
        mean_rate=mean_rate.flatten()
        mean_rate[np.isnan(mean_rate.flatten())]=0.
        mean_rate=mean_rate.reshape([self.grid.zvals.size,self.grid.dmvals.size])
        
        # Rmult here is the rate multiplier due to the distance
        # that is, mean_rate is rate per repeater on average, n reps is number of repeaters, and
        # Rmult is the scaling between the repeater rate and the observed rate
        
        expected_bursts = expected_number_of_repeaters * mean_rate * Rmult * MCmult
        
        Nreps = np.random.poisson(expected_number_of_repeaters)
        sampled_expected = Nreps * mean_rate * Rmult
        
        # sampled number of repeaters. Linearises the array
        nreps_total=np.sum(Nreps)
        nz,ndm=Rthresh.shape
        
        single_array=np.zeros([nz,ndm])
        mult_array = np.zeros([nz,ndm])
        mean_array = np.zeros([nz,ndm])
        exp_array = np.zeros([nz,ndm])
        numbers = np.array([])
        for i in np.arange(nz):
            for j in np.arange(ndm):
                if Nreps[i,j] > 0:
                    # simulate Nreps repeater rates - given there is a repeater here
                    Rs=self.GenNReps(Rthresh[i,j],Nreps[i,j])
                    
                    # simulate the number of detected bursts from each. Includes time factor here
                    Rs_expected=Rs * Rmult[i,j]
                    
                    number=np.random.poisson(Rs_expected)
                    Singles = np.count_nonzero(number==1)
                    Mults = np.count_nonzero(number>1)
                    
                    if Mults > 0:
                        msum = np.where(number >=2)[0]
                        numbers = np.append(numbers,number[msum])
                        msum = np.sum(number[msum])
                        
                    else:
                        msum=0
                    single_array[i,j] = Singles
                    mult_array[i,j] = Mults
                    mean_array[i,j] = msum
                    exp_array[i,j] = np.sum(Rs_expected) # just sums over expected rates of repeaters
                    
                if doplots and i==100 and j==100:
                    Rs=self.GenNReps(Rthresh[i,j],1000)
                    plt.figure()
                    plt.hist(np.log10(Rs),label=str(Rthresh[i,j]))
                    plt.legend()
                    plt.xlabel('log10(R)')
                    plt.ylabel('N(log10(R))')
                    plt.yscale('log')
                    plt.tight_layout()
                    plt.savefig(self.opdir+'repeater_rate_hist.pdf')
                    plt.close()
        
        if doplots:
            if np.max(Nreps)==0:
                dolog=False
            else:
                dolog=True
            self.do_2D_plot(expected_bursts,self.opdir+'expected_bursts'+tag+'.pdf',clabel='Expected number of bursts from repeaters')
            self.do_2D_plot(sampled_expected,self.opdir+'expected_bursts_w_poisson_nrep'+tag+'.pdf',log=dolog,clabel='Expected Nfrb from repeaters (after sampling Nrep)')
            self.do_2D_plot(expected_number_of_repeaters,self.opdir+'expected_repeaters'+tag+'.pdf',log=dolog,clabel='Expected repeaters')
            self.do_2D_plot(Nreps,self.opdir+'Nreps'+tag+'.pdf',log=False,clabel='Number of repeaters')
            self.do_2D_plot(exp_array,self.opdir+'exp'+tag+'.pdf',log=False,clabel='Expected bursts from repeaters')
            self.do_2D_plot(single_array+mean_array,self.opdir+'repsum'+tag+'.pdf',log=False,clabel='All actual bursts from repeaters')
        
        
        return expected_number_of_repeaters,Nreps,single_array,mult_array,mean_array,exp_array,numbers
    
    def GenNReps(self,Rthresh,Nreps):
        """
        Samples N random FRB rates R above Rthresh
        
        CDF [integral Rstar to Rmax] = (R**Rgamma - Rthresh**gamma) / (Rmax**Rgamma - Rthresh**gamma)
        
        """
        rand=np.random.rand(Nreps)
        Rs = ((self.Rmax**(self.Rgamma+1) - Rthresh**(self.Rgamma+1))*rand + Rthresh**(self.Rgamma+1))**(1./(self.Rgamma+1))
        return Rs
    
    def do_2D_plot(self,array,savename,log=True,clabel='',lrange=4,zmax=1):
        """ does a standard imshow """
        if np.nanmax(array)==0.:
            # array is all zeroes
            print("Not plotting ",savename," it is redundant")
            return
        aspect=(self.grid.zvals[-1]/self.grid.dmvals[-1])
        plt.figure()
        plt.xlabel('z')
        plt.ylabel('DM')
        if log:
            toplot=np.log10(array)
            cmax=int(np.nanmax(toplot))+1
            cmin=cmax-lrange
        else:
            toplot=array
        plt.imshow(toplot.T,origin='lower',extent=(0.,self.grid.zvals[-1],0.,self.grid.dmvals[-1]),aspect=aspect)
        cbar=plt.colorbar()
        if log:
            plt.clim(cmin,cmax)
        cbar.set_label(clabel)
        plt.xlim(0,zmax)
        plt.ylim(0,1000)
        plt.tight_layout()
        plt.savefig(savename)
        plt.close()
    
    def do_z_plot(self,array,savename,log=True,label='',ratio=False):
        """ does a standard 1d plot projected onto the z axis
        
        if ratio is True: divide all plots by the last one
        """
        
        plt.figure()
        plt.xlabel('z')
        plt.ylabel('p(z)')
        plt.xlim(0,self.grid.zvals[-1])
        plt.xlim(0,0.5)
        if log:
            plt.yscale('log')
        
        if isinstance(array, list):
            for i,a in enumerate(array):
                zproj=np.sum(a,axis=1)
                if ratio:
                    if i==0:
                        norm=zproj
                    else:
                        plt.plot(self.grid.zvals,zproj/norm,label=label[i])
                else:
                    plt.plot(self.grid.zvals,zproj,label=label[i])
            plt.legend()
        else:
            zproj=np.sum(array,axis=1)
            plt.plot(self.grid.zvals,zproj,label=label)
        
        plt.tight_layout()
        plt.savefig(savename)
        plt.close()
    
    def calc_p_no_repeaters(self,Npointings):
        """
        Calculates the probability per volume that there are no progenitors there
        By default, does this for a single pointing AND all pointings combined
        Not currently used, might be in the future. Useful!
        """
        self.Pnone = np.exp(-self.Nexp)
        self.Pnone_field = np.exp(-self.Nexp_field)
        
    def calc_stochastic_distance(self,p):
        """
        Calculates the distance out to which the chance of there being *any* 
        progenitors becomes p, beyond which the chance is higher
        """
        cexpected = np.cumsum(self.Nexp)
        Psome = 1.-np.exp(cexpected)
        if p < Psome[0]:
            # assume Cartesian local Universe
            # Volume, i.e. [, increases with z^3
            # hence z ~ p^1/3
            zcrit = self.zvals[0] * (p/Psome[0])**(1./3.)
        elif p>Psome[-1]:
            # also assume dumb expansion in z
            zcrit = self.zvals[-1] * (p/Psome[-1])**(1./3.)
        else:
            self.cPsome = np.cumsum(Psome)
            i1=np.where(self.cPsome < p)[0]
            i2=i1+1
            k2=(p-self.cPsome[i1])/(self.cPsome[i2]-self.cPsome[i1])
            k1=1.-k2
            zcrit = k1*self.zvals[i1] + k2*self.zvals[i2]
        return zcrit

    def calc_p_no_bursts(self,Tobs,N1thresh=0.1,N3thresh=10.):
        """
        calculates the chance that no *bursts* are observed in a given volume
        This is an integral over the repeater *and* luminosity distributions
        
        The chance of having no bursts is:
            Outer integral over distribution of repeaters: \int_Rmin ^ Rmax C_r R^gamma dV [] dR
            Inner integral             
            
            p(no bursts) = \int (rate) p(no repeaters) + p(no bursts | repeaters) dR
            
        
        
        The following gives the fraction of the total luminosity function visible
            at any given distance. It has already been integrated over beamshape and width
            self.grid.fractions
        
        # breaks the calculation into many steps. These are:
            # High repetition regime: chance of detecting an FRB given a repeater is 100%
            #   Calculate pure chance of any FRB existing
        
        # calculates the fraction of the luminosity function visible at any given distance
        calc_effective_rate()
        """
        
        ###### all this is ignoring redshift and DM dependence so far
        # later will investigate what it looks like when looping over both
        
        # this gives the scaling between an intrinsic rate R0 and an observable rate effR0
        # this shifts the effective rate distribution to
        effRmin=self.Rmin*self.grid.fractions
        effRmax=self.Rmax*self.grid.fractions
        
        # converts rates into expected burst numbers gives a certain observation time
        Nmin = effRmin*Tobs
        Nmax = effRmax*Tobs
        
        R1 = Rmin*N1thresh/Nmin
        R3 = Rmax*N3hresh/Nmax
            
        
        ######## regime 1: weak repeaters ########
        # we assume that p(>2 bursts)=0
        # hence we integrate over total burst numbers
        if Rmin < R1:
            if Rmax < R1:
                # this is simply the original calculation from the grid
                # all bursts are independent
                Ntot_exp1 = self.grid.rates * Tobs * C
                
                
                # internal calculation as check:
                Ntot_exp1 = C_r*(Rmax**(gamma+2) - Rmin**(gamma+2))/(gamma+2)
            else:
                Ntot_exp1 = C_r*(R1**(gamma+2) - Rmin**(gamma+2))/(gamma+2)
            p1None = np.exp(-Ntot_exp1)
        else:
            p1None=1.
        
        ####### regime 2: intermediate repeaters of doom ####
        # we need to consider p(no repeaters) + p(no_burts|repeaters)
        
        # firstly: p(no repeaters). Calculates thresholds
        if R3 > Rmax:
            if R1 < Rmin:
                # all the repeaters in this regime
                N2reps = np.exp(-Cr) # all the repeaters!
            else:
                N2reps = C_r*(Rmax**(gamma+1) - R1**(gamma+1))/(gamma+1)
        else:
            N2reps = C_r*(R3**(gamma+1) - R1**(gamma+1))/(gamma+1)
        p2no_reps = np.exp(-N2reps)

