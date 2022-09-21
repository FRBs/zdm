
"""
Class definition for repeating FRBs
"""

from zdm import grid
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from zdm import energetics
import mpmath
from scipy import interpolate
import time

SET_REP_ZERO=1e-4 # forces p(n=0) bursts to be unity when a maximally repeating FRB has SET_REP_ZERO expected events

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
    This will change.
    
    """
    
    
    def __init__(self,grid,Tfield=None,Nfields=None,opdir=None,MC=False,verbose=False):
        """
        Initialises repeater class
        Args:
            grid: zdm grid class object
            Nfields: number of separate pointings by the survey
            Tfield: time spent on each field
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
        
        # get often-used data from the grid - for simplicity
        self.Emin = 10**self.grid.state.energy.lEmin
        self.Emax = 10**self.grid.state.energy.lEmax
        self.gamma= self.grid.state.energy.gamma
        
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
            raise ValueError("Nfields not specified")
        
        if Tfield is not None:
            self.Tfield=Tfield
        elif grid.survey.meta.has_key('Tfield'):
            self.Tfield = grid.survey.meta['Tfield']
        else:
            raise ValueError("Tfield not specified")
        
        # calculates constant Rc in front of dN/dR = Rc R^Rgamma
        self.Rc,self.NRtot=self.calc_constant(verbose=True)
        grid.state.rep.RC=self.Rc
        if verbose:
            print("Calculated constant as ",self.Rc)
        
        # number of expected FRBs per volume in a given field
        self.Nexp_field = self.grid.dV*self.Rc
        self.Nexp = self.Nexp_field * self.Nfields
        
        self.calc_Rthresh(MC=MC,doplots=doplots)
        
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
        
        """
        
        # sets repeater constant as per global FRB rate constant
        # C is in bursts per Gpc^3 per year
        # rate is thus in bursts/year
        C=10**(self.grid.state.FRBdemo.lC)
        if verbose:
            print("Initial burst rate as ",C*1e9*365.25," per Gpc^-3 per year")
        
        # account for repeat parameters being defined at a different energy than Emin
        fraction = self.grid.vector_cum_lf(self.state.rep.RE0,self.Emin,self.Emax,self.gamma)
        C = C*fraction
        
        if verbose:
            print("This is ",C*1e9*365.25," Gpc^-3 year^-1 above ",self.state.rep.RE0," erg")
        
        Rc = C * (self.Rgamma+2.)/(self.Rmax**(self.Rgamma+2.)-self.Rmin**(self.Rgamma+2.))
        
        Ntot = (Rc/(self.Rgamma+1)) * (self.Rmax**(self.Rgamma+1)-self.Rmin**(self.Rgamma+1))
        
        if verbose:
            print("Corresponding to ",Ntot," repeaters every cubic Mpc")
        
        return Rc,Ntot
    
    
    def calc_Rthresh(self,MC=False,Pthresh=0.01,doplots=False):
        """
        For FRBs on a zdm grid, calculate the repeater rates below
        which we can reasonably treat the distribution as continuous
        
        Steps:
            - Calculate the apparent repetition threshold to ignore repetition
            - Calculate rate scaling between energy at which repeat rates
              are defined, and energy threshold
        
        MC: if True, perform a single MC evaluations
            if False, use the analytic method
        
        Pthresh: only if MC is true
            threshold above which to allow progenitors to produce multilke bursts
        """
        
        # calculates rate corresponding to Pthresh
        # P = 1.-(1+R) * np.exp(-R)
        # P = 1.-(1+R) * (1.-R + R^2/2) = 1 - [1 +R -R - R^2 + R^2/2 + cubic]
        # P = 0.5 R^2 # assumes R is small to get
        # R = (2P)**0.5
        Rthresh = (2.*Pthresh)**0.5 #only works for small rates
        # we loop over beam values
        for ib,b in enumerate(self.grid.beam_b):
            if ib==0:
                
                exactset,MCset = self.sim_repeaters(Rthresh,b,self.grid.beam_o[ib],doplots,MC)
                exact_singles,exact_zeroes,exact_rep_bursts,exact_reps = exactset
                if MC:
                    expNrep,Nreps,single_array,mult_array,summed_array,exp_array,poisson_rates = MCset
            else:
                exactset,MCset = self.sim_repeaters(Rthresh,b,self.grid.beam_o[ib],doplots,MC)
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
        
        
        if doplots:
            opdir=self.opdir
            self.do_2D_plot(exact_singles,opdir+'exact_singles.pdf',clabel='Expected number of single FRBs')
            self.do_2D_plot(exact_reps,opdir+'exact_repeaters.pdf',clabel='Expected number of repeating FRBs')
            self.do_2D_plot(exact_rep_bursts,opdir+'exact_repeat_bursts.pdf',clabel='Expected number of repeat bursts')
            self.do_2D_plot(exact_zeroes,opdir+'exact_zeroes.pdf',clabel='Expected number of FRBs with zero bursts')
            
            exact_total = exact_singles + exact_rep_bursts
            self.do_z_plot([exact_singles,exact_rep_bursts,exact_total],opdir+'exact_bursts_fz.pdf',
                log=False,label=['single bursts','repeat bursts','total_bursts'])
            
            
        if doplots and MC:
            #self.do_2D_plot(use_thresh,opdir+'erg_threshold.pdf',clabel='Detection threshold [erg]')
            #self.do_2D_plot(Rmult,opdir+'Rmult.pdf',clabel='Rate multiplier Rmult')
            #self.do_2D_plot(dmzRthresh,opdir+'Req_rate.pdf',clabel='Required rate at 1e38 erg')
            #self.do_2D_plot(Rfraction,opdir+'Rfraction.pdf',log=False,clabel='Fraction of bursts coming from significant repeaters')
            #self.do_2D_plot(Poisson,opdir+'poisson_fraction.pdf',log=True,clabel='Burst fraction guaranteed to be independent of repetition')
            self.do_2D_plot(poisson_rates,opdir+'poisson_rates.pdf',clabel='Burst fraction guaranteed to be independent of repetition')
            #self.do_2D_plot(use_thresh,opdir+'Ethresh.pdf',clabel='log10 (Ethresh) [erg]')
            #self.do_2D_plot(volume_grid,opdir+'pdmdV.pdf',clabel='p(DM|z) dV/dz product')
            self.do_2D_plot(expNrep,opdir+'n_repeaters.pdf',clabel='Expected number of repeating objects per bin')
            self.do_2D_plot(Nreps,opdir+'sampled_n_repeaters.pdf',log=False,clabel='Actual number of repeating objects per bin')
            self.do_2D_plot(Nreps/expNrep,opdir+'ratio_sampled_exp_repeaters.pdf',log=False,clabel='Ratio of sampled to expected repeaters')
            self.do_2D_plot(single_array,opdir+'singles.pdf',log=False,clabel='Single bursts from repeaters')
            self.do_2D_plot(mult_array,opdir+'mults.pdf',log=False,clabel='Number of observed repeaters per bin (N >= 2)')
            self.do_2D_plot(summed_array,opdir+'summed_reps.pdf',log=False,clabel='Sum of all multiple bursts from repeaters')
            self.do_2D_plot((summed_array+single_array)/exp_array,opdir+'ratio_exp_reps.pdf',log=False,clabel='Number of bursts from repeaters (single and mult)')
            if np.nanmax(exp_array)>0:
                self.do_2D_plot(exp_array,opdir+'expected_repeat_bursts.pdf',clabel='Expected number of bursts from modelled repeaters')
                self.do_2D_plot(exp_array/no_repeaters,opdir+'ratio_exp_noreps.pdf',log=False,clabel='Ratio of expected repeaters to no repeaters')
            
            self.do_2D_plot(TotalSingle,opdir+'total_single.pdf',clabel='Actual number of single bursts (analytic + reps)')
            self.do_2D_plot(no_repeaters,opdir+'no_repeaters.pdf',clabel='Number of single bursts when ignoring repetition')
            self.do_2D_plot(TotalSingle/no_repeaters,opdir+'ts_ratio.pdf',log=False,clabel='Ratio of single bursts to those expected')
            self.do_2D_plot(total_bursts,opdir+'all_bursts.pdf',clabel='Total number of bursts, including repeaters')
            self.do_2D_plot(total_bursts/no_repeaters,opdir+'totratio.pdf', log=False,clabel='Ratio of total generated bursts to those expected ignoring repeaters')
        
            self.do_z_plot([total_bursts,no_repeaters],opdir+'total_norep_bursts_fz.pdf',
                log=False,label=['all bursts (MC)','all bursts (poisson)'])
            
            #total_bursts_z=np.sum(total_bursts,axis=1)
            #no_repeaters_z=np.sum(no_repeaters,axis=1)
            #plt.figure()
            #plt.plot(self.grid.zvals,total_bursts_z,label='total bursts')
            #plt.plot(self.grid.zvals,no_repeaters_z,label='no repeaters')
            #plt.legend()
            #plt.xlabel('z')
            #plt.ylabel('N(z)')
            #plt.tight_layout()
            #plt.savefig(opdir+'total_norep_bursts_fz.pdf')
            #plt.close()
            
            self.do_z_plot([TotalSingle,no_repeaters],opdir+'z_ts_ratio.pdf',
                log=False,label=['single bursts','all bursts'])
            
            ### z-projection ###
            #z_TotalSingle = np.sum(TotalSingle,axis=1)
            #z_no_repeaters = np.sum(no_repeaters,axis=1)
            #plt.figure()
            #plt.plot(self.grid.zvals,z_TotalSingle,label='Single bursts only')
            #plt.plot(self.grid.zvals,z_no_repeaters,label='All bursts')
            #plt.legend()
            #plt.xlabel('z')
            #plt.ylabel('N(z), 200 day ASKAP ICS pointing')
            #plt.tight_layout()
            #plt.savefig(opdir+'z_ts_ratio.pdf')
            #plt.close()
            
            self.do_z_plot([no_repeaters,total_bursts],opdir+'total_norep_bursts_fz_ratio.pdf',
                log=False,label=['','ratio'],ratio=True)
                
            #totratio_z=total_bursts_z/no_repeaters_z
            #plt.figure()
            #plt.plot(self.grid.zvals,totratio_z)
            #plt.xlabel('z')
            #plt.ylabel('N(z)')
            #plt.tight_layout()
            #plt.savefig(opdir+'total_norep_bursts_fz_ratio.pdf')
            #plt.close()
    
    def perform_exact_calculations(self,Rmult,volume_grid,slow=False):
        """
        Performs exact calculations of:
            - <single bursts>
            - <no bursts>
            - < progenitors with N>2 bursts>
            - < number of N>2 bursts>
        """
        exact_singles = self.calc_singles_exactly(Rmult)
        exact_zeroes = self.calc_zeroes_exactly(Rmult)
        exact_rep_bursts = self.calc_expected_repeat_bursts(Rmult,exact_singles)
        exact_reps = self.calc_expected_repeaters(Rmult,exact_singles,exact_zeroes)
        
        # the below is a simple slow loop which is good for test purposes
        # it does not use spline interpolation
        # testing so results are good to ~10^-12 differences
        if slow:
            s,z,m,n,t=self.slow_exact_calculation(Rmult,exact_singles,exact_zeroes,exact_rep_bursts,exact_reps)
            
        
        exact_singles = exact_singles*volume_grid
        exact_zeroes = exact_zeroes*volume_grid
        exact_rep_bursts = exact_rep_bursts*volume_grid
        exact_reps = exact_reps*volume_grid
        
        exact_singles = (exact_singles.T * self.grid.sfr).T
        exact_zeroes = (exact_zeroes.T * self.grid.sfr).T
        exact_rep_bursts = (exact_rep_bursts.T * self.grid.sfr).T
        exact_reps = (exact_reps.T * self.grid.sfr).T
        
        return exact_singles,exact_zeroes,exact_rep_bursts,exact_reps
    
    def calc_expected_repeaters(self,Rmult,expected_singles,expected_zeros):
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
        summed=expected_singles + expected_zeros
        expected_repeaters = total_repeaters - expected_singles - expected_zeros
        
        return expected_repeaters
    
    def calc_expected_repeat_bursts(self,Rmult,expected_singles):
        """
        Calculates the expected total number of bursts from repeaters.
        
        This is simply calculated as <all bursts> - <single bursts>
        """
        
        a = self.Rmin*Rmult
        b = self.Rmax*Rmult
        effGamma=self.Rgamma+2
        total_rate = self.Rc * (1./effGamma) * (self.Rmax**effGamma-self.Rmin**effGamma) * Rmult
        mult_rate = total_rate - expected_singles
        return mult_rate
        
    
    def calc_singles_exactly(self,Rmult):
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
        
        nz,ndm=Rmult.shape
        norms=np.zeros([nz,ndm])
        
        effGamma=self.Rgamma+2
        if effGamma not in energetics.igamma_splines.keys():
            energetics.init_igamma_splines([effGamma])
        avals=self.Rmin*Rmult.flatten()
        bvals=self.Rmax*Rmult.flatten()
        
        norms = interpolate.splev(avals, energetics.igamma_splines[effGamma])
        norms -= interpolate.splev(bvals, energetics.igamma_splines[effGamma])
        norms=norms.reshape([nz,ndm])
        
        # integral is in units of R'=R*Rmult
        # however population is specified in number density of R
        # hence R^gammadensity factor must be normalised
        norms /= Rmult**(self.Rgamma+1) 
        
        # multiplies this by the number density of repeating FRBs
        norms *= self.Rc
        
        return norms
    
    def slow_exact_calculation(self,Rmult,exact_singles,exact_zeroes,exact_rep_bursts,exact_reps,plot=True,zonly=True):
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
        
        nz,ndm=Rmult.shape
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
                a=self.Rmin*Rmult[i,j]
                b=self.Rmax*Rmult[i,j]
                
                zero = mpmath.gammainc(self.Rgamma+1, a=a, b=b)/(Rmult[i,j]**(self.Rgamma+1))
                single = mpmath.gammainc(effGamma, a=a, b=b)/(Rmult[i,j]**(self.Rgamma+1))
                nrep = (1./(self.Rgamma+1)) * (self.Rmax**(self.Rgamma+1)-self.Rmin**(self.Rgamma+1))
                nrep = nrep - single - zero
                total = (1./effGamma) * (b**effGamma-a**effGamma)/(Rmult[i,j]**(self.Rgamma+1))
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
        #s *= self.Rc
        #z *= self.Rc
        #m *= self.Rc
        #n *= self.Rc
        #t *= self.Rc
        
        
        print("Max diff s is ",np.max(np.abs(exact_singles-s)))
        print("Max diff z is ",np.max(np.abs(exact_zeroes-z)))
        print("Max diff n is ",np.max(np.abs(exact_reps-n)))
        print("Max diff m is ",np.max(np.abs(exact_rep_bursts-m)))
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
    
    def calc_zeroes_exactly(self,Rmult):
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
        
        nz,ndm=Rmult.shape
        norms=np.zeros([nz,ndm])
        
        effGamma=self.Rgamma+1
        if effGamma not in energetics.igamma_splines.keys():
            t0=time.time()
            energetics.init_igamma_splines([effGamma])
            t1=time.time()
            print("Init took ",t1-t0)
        avals=self.Rmin*Rmult.flatten()
        bvals=self.Rmax*Rmult.flatten()
        zero=np.where(bvals<=SET_REP_ZERO)[0]
        norms = interpolate.splev(avals, energetics.igamma_splines[effGamma])
        #self.do_2D_plot(norms.reshape([nz,ndm]),'TEST_a.pdf')
        #norms2 = interpolate.splev(bvals, energetics.igamma_splines[effGamma])
        #self.do_2D_plot(norms2.reshape([nz,ndm]),'TEST_b.pdf')
        
        norms -= interpolate.splev(bvals, energetics.igamma_splines[effGamma])
        norms=norms.reshape([nz,ndm])
        norms *= self.Rc
        norms /= Rmult**(self.Rgamma+1) # gamma due to integrate, one more due to dR
        
        # integral is in units of R'=R*Rmult
        # however population is specified in number density of R
        # hence R^gammadensity factor must be normalised
        
        
        
        # multiplies this by the number density of repeating FRBs
        norms=norms.flatten()
        norms[zero] = self.Rc * (1./effGamma) * (self.Rmax**effGamma-self.Rmin**effGamma)
        norms=norms.reshape([nz,ndm])
        
        return norms
      
    def sim_repeaters(self,Rthresh,beam_b,Solid,MC=False,doplots=False):
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
        
        Rmult = self.calcRmult(beam_b=beam_b)
        if doplots:
            self.do_2D_plot(Rmult,self.opdir+'Rmult_'+str(beam_b)[0:5]+'.pdf',clabel='log$_{10}$ rate multiplier')
        
        # the below has units of Mpc^3 per zbin, and multiplied by p(DM|z) for each bin
        # Note that grid.dV already has a (1+z) time dilation factor in it
        # This is actually accounted for in the rate scaling of repeaters
        # Here, we want the actual volume, hence must remove this factor
        volume_grid = (self.grid.smear_grid.T * (self.grid.dV * (1. + self.grid.zvals))).T
        volume_grid *= Solid #accounts for solid angle viewed at that beam sensitivity
        
        #exact_singles_rate=self.calc_singles_exactly(Rmult)
        exact_singles,exact_zeroes,exact_rep_bursts,exact_reps=self.perform_exact_calculations(Rmult,volume_grid)
        
        exact_set = [exact_singles,exact_zeroes,exact_rep_bursts,exact_reps]
        
        if doplots:
            self.do_2D_plot(exact_singles,self.opdir+'exact_singles_'+str(beam_b)[0:5]+'.pdf',clabel='log$_{10}$ rate')
            self.do_2D_plot(exact_zeroes,self.opdir+'exact_zeroes_'+str(beam_b)[0:5]+'.pdf',clabel='log$_{10}$ rate')
            self.do_2D_plot(exact_rep_bursts,self.opdir+'exact_rep_bursts_'+str(beam_b)[0:5]+'.pdf',clabel='log$_{10}$ rate')
            self.do_2D_plot(exact_reps,self.opdir+'exact_reps_'+str(beam_b)[0:5]+'.pdf',clabel='log$_{10}$ rate',lrange=6)
            total=exact_rep_bursts+exact_singles
            self.do_2D_plot(total,self.opdir+'exact_all_bursts_'+str(beam_b)[0:5]+'.pdf',clabel='log$_{10}$ rate')
            
            #self.do_z_plot([total,exact_singles,exact_rep_bursts,exact_reps,
            #    self.grid.rates*self.Tfield*10**(self.grid.state.FRBdemo.lC)],
            #    self.opdir+'zproj_exact_'+str(beam_b)[0:5]+'.pdf',
            #    label=['Total','One-off FRBs','bursts from repeaters','Repeaters','grid.rates'])
            
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
            expNrep,Nreps,single_array,mult_array,summed_array,exp_array = self.MCsample(dmzRthresh,volume_grid,Rmult,doplots,tag='_b'+str(beam_b)[0:5])
            MCset=[expNrep,Nreps,single_array,mult_array,summed_array,exp_array,poisson_rates]
            results=[exact_set,MCset]
        else:
            results=[exact_set,None]
        
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
        
    
    def calcRmult(self,beam_b=1):
        # calculates Rmult over a range of burst widths and probabilities
        #print("Calculating Rmult using ",beam_b)
        for iw,w in enumerate(self.grid.eff_weights):
            # we want to calculate for each point an Rmult such that
            # Rmult_final = \sum wi Rmulti
            # does this make sense? Effectively its the sum of rates. Well yes it does! AWESOME!
            # numerator for Rmult for this width
            if iw==0:
                Rmult = w*self.grid.array_cum_lf(self.grid.thresholds[iw,:,:]/beam_b,self.Emin,self.Emax,self.gamma)
            else:
                Rmult += w*self.grid.array_cum_lf(self.grid.thresholds[iw,:,:]/beam_b,self.Emin,self.Emax,self.gamma)
        Rmult /= self.grid.vector_cum_lf(self.state.rep.RE0,self.Emin,self.Emax,self.gamma)
        
        # calculates the expectation value for a single pointing
        # rates were "per day", now "per pointing time on field"
        Rmult *= self.Tfield
        
        # accounts for time dilation of intrinsic rate
        dilation=1./(1+self.grid.zvals)
        Rmult = (Rmult.T * dilation).T
        
        return Rmult
    
    def MCsample(self,Rthresh,volume_grid,Rmult,doplots=False,tag=None):
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
        
        
        # number of repeaters should NOT have time but SHOULD have solid angle
        expected_number_of_repeaters = EffRc * volume_grid
        expected_number_of_repeaters = (expected_number_of_repeaters.T * self.grid.sfr).T
        
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
        expected_bursts = expected_number_of_repeaters * mean_rate * Rmult
        
        Nreps = np.random.poisson(expected_number_of_repeaters)
        sampled_expected = Nreps * mean_rate * Rmult
        
        # sampled number of repeaters. Linearises the array
        nreps_total=np.sum(Nreps)
        nz,ndm=Rthresh.shape
        
        single_array=np.zeros([nz,ndm])
        mult_array = np.zeros([nz,ndm])
        mean_array = np.zeros([nz,ndm])
        exp_array = np.zeros([nz,ndm])
        for i in np.arange(nz):
            for j in np.arange(ndm):
                if Nreps[i,j] > 0:
                    # simulate Nreps repeater rates
                    Rs=self.GenNReps(Rthresh[i,j],Nreps[i,j])
                    # simulate the number of detected bursts from each. Includes time factor here
                    Rs_expected=Rs * Rmult[i,j]
                    #print(i,j,Rthresh[i,j],expected_number_of_repeaters[i,j],Nreps[i,j],Rmult[i,j],np.sum(Rs)/Nreps[i,j],mean_rate[i,j])
                    #print("     ",Rs)
                    #print("     ",Rs_expected)
                    number=np.random.poisson(Rs_expected)
                    Singles = np.count_nonzero(number==1)
                    Mults = np.count_nonzero(number>1)
                    
                    if Mults > 0:
                        msum = np.where(number >=2)[0]
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
        
        
        return expected_number_of_repeaters,Nreps,single_array,mult_array,mean_array,exp_array
    
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


def poisson_z(Robs,*args):
    """ integrates Poisson probabilities
    
    UNFINISHED
    """
    
    for i in np.arange(20)+1: # number of repeaters - already done 0!
        
        # total p_zero given N: AND condition
        #pz is the probability of detecting a burst from that repeaters
        pz_total = poisson_N*pz**N 
            
        #now: chance of a burst given repeaters
        while(1):
            # Poisson probability of N repeaters: loop over N, keep going until p is negligible
            
            
            # chance of zero bursts per repeaters
            np.integrate.quad(N1,N3)
            
            
        
        if Rmax > R3:
            # p3None now contains chance that none of these are actually in the volume
            if Rmin > R3:
                p3None = np.exp(-Cr) # all the repeaters!
            else:
                # there may be some FRBs which we are *guaranteed* to see
                # thus calculate the probability they exist
                # \int_(Rthresh)^(Rmax) C_r R^gamma dN
                Ntot_exp3 = C_r*(Rmax**(gamma+1) - R3**(gamma+1))/(gamma+1)
                p3None = np.exp(-Ntot_exp3)
        else:
            p3None=1.
        
        
        # we first identify if we are in the "not many repeaters' regime
        # in that case, we can assume there are 0 or 1 repeaters in each volume
        
        
        
        ######## regime 3: strong repeaters ########
        # we assume p(>2 bursts) = 1.
        if Nmax > N3thresh:
            # p3None now contains chance that none of these are actually in the volume
            if Nmin > N3thresh:
                p3None = np.exp(-Cr)
            else:
                # there may be some FRBs which we are *guaranteed* to see
                # thus calculate the probability they exist
                # \int_(Rthresh)^(Rmax) C_r R^gamma dN
                R3thresh = Rmax * N3thresh / Nmax 
                
                Ntot_exp3 = C_r*(Rmax**(gamma+1) - R3thresh**(gamma+1))/(gamma+1)
                p3None = np.exp(-Ntot_exp3)
        else:
            p3None=1.
            
        
            
def calc_p_rep(Cr,V):
    """
    Calculates the probability that any repeater at all resides
    in a volume.
    
    This is purely a function of the volume (Gpc^3) and the
    total repeater density Cr (repeaters Gpc^-3)
    
    """
    
    # generates expected number of repeaters
    Expected = Cr*V
    
    # generates random number of repeaters
    Nrep = np.random.poisson(Expected)
    

