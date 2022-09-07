
"""
Class definition for repeating FRBs
"""

from zdm import grid
import matplotlib.pyplot as plt
import numpy as np

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
    
    
    def __init__(self, grid, Tfield=None, Nfields=None):
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
        print("Calculated constant as ",self.Rc)
        print("Various factors are ",(1+self.Rgamma)**-1,(1+self.Rgamma)**-2)
        
        # number of expected FRBs per volume in a given field
        self.Nexp_field = self.grid.dV*self.Rc
        self.Nexp = self.Nexp_field * self.Nfields
        
        self.calc_Rthresh()
        
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
        
        Emin = 10**self.grid.state.energy.lEmin
        Emax = 10**self.grid.state.energy.lEmax
        gamma= self.grid.state.energy.gamma
        
        # account for repeat parameters being defined at a different energy than Emin
        fraction = self.grid.vector_cum_lf(self.state.rep.RE0,Emin,Emax,gamma)
        C = C*fraction
        
        if verbose:
            print("This is ",C*1e9*365.25," Gpc^-3 year^-1 above ",self.state.rep.RE0," erg")
        
        Rc = C * (self.Rgamma+2.)/(self.Rmax**(self.Rgamma+2.)-self.Rmin**(self.Rgamma+2.))
        
        Ntot = (Rc/(self.Rgamma+1)) * (self.Rmax**(self.Rgamma+1)-self.Rmin**(self.Rgamma+1))
        
        if verbose:
            print("Corresponding to ",Ntot," repeaters every cubic Mpc")
        
        return Rc,Ntot
    
    def calc_Rthresh(self,Pthresh=0.01):
        """
        For FRBs on a zdm grid, calculate the repeater rates below
        which we can reasonably treat the distribution as continuous
        
        Steps:
            - Calculate the apparent repetition threshold to ignore repetition
            - Calculate rate scaling between energy at which repeat rates
              are defined, and energy threshold
        """
        
        # calculates rate corresponding to Pthresh
        # P = 1.-(1+R) * np.exp(-R)
        # P = 1.-(1+R) * (1.-R + R^2/2) = 1 - [1 +R -R - R^2 + R^2/2 + cubic]
        # P = 0.5 R^2 # assumes R is small to get
        # R = (2P)**0.5
        Rthresh = (2.*Pthresh)**0.5 #only works for small rates
        
        # contains threshold information as function of FRB width, zvals, DMvals)
        # approximate this only as z x DM for now
        print("The shape of the grid thresholds is ",self.grid.thresholds.shape)
        print("We are performing an analysis assuming b=1, and the first width value")
        print("Furthermore, the assumed solid angle is 1")
        print("A to-do in future is to iterate over these dimensions")
        print("We will probably weight the rate through the response to different widths")
        print("However, beam solid angles are independent, so we just put this in a loop and sum it")
        use_thresh=self.grid.thresholds[0,:,:]
        Solid=1
        
        Emin = 10**self.grid.state.energy.lEmin
        Emax = 10**self.grid.state.energy.lEmax
        gamma= self.grid.state.energy.gamma
        
        # calculates rate multiplier for FRBs
        # this is total rate above use_thresh vs rate above RE0
        Rmult=self.grid.array_cum_lf(use_thresh,Emin,Emax,gamma)/self.grid.vector_cum_lf(self.state.rep.RE0,Emin,Emax,gamma)
        # if Rmult is above 1, then use_thresh should be lower than RE0
        
        # calculates the expectation value for a single pointing
        # rates were "per day", now "per pointing time on field"
        Rmult *= self.Tfield
        
        # accounts for time dilation of intrinsic rate
        dilation=1./(1+self.grid.zvals)
        Rmult = (Rmult.T * dilation).T
        
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
        print("Number of single FRBs from insignificant FRBs ",np.sum(poisson_rates)) #FRBs per 2000 dats per steradian
        
        # the below has units of Mpc^3 per zbin, and multiplied by p(DM|z) for each bin
        # Note that grid.dV already has a (1+z) time dilation factor in it
        # This is actually accounted for in the rate scaling of repeaters
        # Here, we want the actual volume, hence must remove this factor
        volume_grid = (self.grid.smear_grid.T * (self.grid.dV * (1. + self.grid.zvals))).T
        volume_grid *= Solid #accounts for solid angle viewed at that beam sensitivity
        
        # returns zdm grid with values being number of single bursts, number of repeaters, number of repeat bursts
        expNrep,Nreps,single_array,mult_array,summed_array,exp_array = self.MCsample(dmzRthresh,volume_grid,Rmult)
        
        
        
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
        
        # MC sampling of FRBs
        
        ########### PLOTTING ###########
        aspect=(self.grid.zvals[-1]/self.grid.dmvals[-1])
        opdir='Repeaters/'
        
        
        plt.figure()
        plt.xlabel('z')
        plt.ylabel('DM')
        plt.imshow(np.log10(use_thresh.T),origin='lower',extent=(0.,self.grid.zvals[-1],0.,self.grid.dmvals[-1]),aspect=aspect)
        cbar=plt.colorbar()
        cbar.set_label('Detection threshold [erg]')
        plt.xlim(0,2)
        plt.ylim(0,2000)
        plt.tight_layout()
        plt.savefig(opdir+'erg_threshold.pdf')
        plt.close()
        
        
        plt.figure()
        toplot=Rmult
        themax=np.max(toplot)
        cmax=int(np.log10(themax))+1
        cmin=cmax-4
        plt.xlabel('z')
        plt.ylabel('DM')
        plt.imshow(np.log10(toplot.T),origin='lower',extent=(0.,self.grid.zvals[-1],0.,self.grid.dmvals[-1]),aspect=aspect)
        cbar=plt.colorbar()
        plt.clim(cmin-2,cmax-1)
        cbar.set_label('Rmult')
        plt.xlim(0,2)
        plt.ylim(0,2000)
        plt.tight_layout()
        plt.savefig(opdir+'Rmult.pdf')
        plt.close()
        
        plt.figure()
        plt.xlabel('z')
        plt.ylabel('DM')
        plt.imshow(np.log10(dmzRthresh.T),origin='lower',extent=(0.,self.grid.zvals[-1],0.,self.grid.dmvals[-1]),aspect=aspect)
        cbar=plt.colorbar()
        cbar.set_label('Required rate at 1e38 erg')
        plt.xlim(0,2)
        plt.ylim(0,2000)
        plt.tight_layout()
        plt.savefig(opdir+'Req_rate.pdf')
        plt.close()
        
        plt.figure()
        plt.xlabel('z')
        plt.ylabel('DM')
        plt.imshow(Rfraction.T,origin='lower',extent=(0.,self.grid.zvals[-1],0.,self.grid.dmvals[-1]),aspect=aspect)
        cbar=plt.colorbar()
        plt.clim(0,1)
        cbar.set_label('Fraction of bursts coming from significant repeaters')
        plt.xlim(0,2)
        plt.ylim(0,2000)
        plt.tight_layout()
        plt.savefig(opdir+'Rfraction.pdf')
        plt.close()
        
        plt.figure()
        plt.xlabel('z')
        plt.ylabel('DM')
        plt.imshow(np.log10(Poisson.T),origin='lower',extent=(0.,self.grid.zvals[-1],0.,self.grid.dmvals[-1]),aspect=aspect)
        cbar=plt.colorbar()
        cbar.set_label('Burst fraction guaranteed to be independent of repetition')
        plt.xlim(0,2)
        plt.ylim(0,2000)
        plt.tight_layout()
        plt.savefig(opdir+'poisson_fraction.pdf')
        plt.close()
        
        plt.figure()
        plt.xlabel('z')
        plt.ylabel('DM')
        plt.imshow(np.log10(poisson_rates.T),origin='lower',extent=(0.,self.grid.zvals[-1],0.,self.grid.dmvals[-1]),aspect=aspect)
        cbar=plt.colorbar()
        cbar.set_label('Burst fraction guaranteed to be independent of repetition')
        plt.xlim(0,2)
        plt.ylim(0,2000)
        plt.tight_layout()
        plt.savefig(opdir+'poisson_rates.pdf')
        plt.close()
        
        plt.figure()
        plt.xlabel('z')
        plt.ylabel('DM')
        plt.imshow(np.log10(use_thresh.T),origin='lower',extent=(0.,self.grid.zvals[-1],0.,self.grid.dmvals[-1]),aspect=aspect)
        cbar=plt.colorbar()
        cbar.set_label('log10 (Ethresh) [erg]')
        plt.xlim(0,2)
        plt.ylim(0,2000)
        plt.tight_layout()
        plt.savefig(opdir+'Ethresh.pdf')
        plt.close()
        
        plt.figure()
        plt.xlabel('z')
        plt.ylabel('DM')
        plt.imshow(np.log10(volume_grid.T),origin='lower',extent=(0.,self.grid.zvals[-1],0.,self.grid.dmvals[-1]),aspect=aspect)
        cbar=plt.colorbar()
        cbar.set_label('p(DM|z) dV/dz product')
        plt.clim(-1,7)
        plt.xlim(0,2)
        plt.ylim(0,2000)
        plt.tight_layout()
        plt.savefig(opdir+'pdmdV.pdf')
        plt.close()
        
        plt.figure()
        toplot=np.log10(expNrep)
        cmax=int(np.max(toplot))+1
        cmin=cmax-4
        plt.xlabel('z')
        plt.ylabel('DM')
        plt.imshow(toplot.T,origin='lower',extent=(0.,self.grid.zvals[-1],0.,self.grid.dmvals[-1]),aspect=aspect)
        cbar=plt.colorbar()
        cbar.set_label('Expected number of repeating objects per bin')
        plt.clim(cmin,cmax)
        plt.xlim(0,2)
        plt.ylim(0,2000)
        plt.tight_layout()
        plt.savefig(opdir+'n_repeaters.pdf')
        plt.close()
        
        plt.figure()
        plt.xlabel('z')
        plt.ylabel('DM')
        plt.imshow(np.log10(Nreps.T),origin='lower',extent=(0.,self.grid.zvals[-1],0.,self.grid.dmvals[-1]),aspect=aspect)
        cbar=plt.colorbar()
        cbar.set_label('Actual number of repeating objects per bin')
        plt.clim(cmin,cmax)
        plt.xlim(0,2)
        plt.ylim(0,2000)
        plt.tight_layout()
        plt.savefig(opdir+'sampled_n_repeaters.pdf')
        plt.close()
        
        plt.figure()
        plt.xlabel('z')
        plt.ylabel('DM')
        plt.imshow((Nreps/expNrep).T,origin='lower',extent=(0.,self.grid.zvals[-1],0.,self.grid.dmvals[-1]),aspect=aspect)
        cbar=plt.colorbar()
        cbar.set_label('Ratio of sampled to expected repeaters')
        plt.clim(0,2)
        plt.xlim(0,2)
        plt.ylim(0,2000)
        plt.tight_layout()
        plt.savefig(opdir+'ratio_sampled_exp_repeaters.pdf')
        plt.close()
        
        plt.figure()
        plt.xlabel('z')
        plt.ylabel('DM')
        plt.imshow(np.log10(single_array.T),origin='lower',extent=(0.,self.grid.zvals[-1],0.,self.grid.dmvals[-1]),aspect=aspect)
        cbar=plt.colorbar()
        cbar.set_label('Actual number of repeating objects per bin')
        plt.clim(0,2)
        plt.xlim(0,2)
        plt.ylim(0,2000)
        plt.tight_layout()
        plt.savefig(opdir+'singles.pdf')
        plt.close()
        
        
        plt.figure()
        plt.xlabel('z')
        plt.ylabel('DM')
        plt.imshow(np.log10(mult_array.T),origin='lower',extent=(0.,self.grid.zvals[-1],0.,self.grid.dmvals[-1]),aspect=aspect)
        cbar=plt.colorbar()
        cbar.set_label('Number of observed repeaters per bin (N >= 2)')
        plt.clim(0,2)
        plt.xlim(0,2)
        plt.ylim(0,2000)
        plt.tight_layout()
        plt.savefig(opdir+'mults.pdf')
        plt.close()
        
        plt.figure()
        toplot=summed_array
        themax=np.max(toplot)
        cmax=int(np.log10(themax))+1
        cmin=cmax-4
        plt.xlabel('z')
        plt.ylabel('DM')
        plt.imshow(np.log10(toplot.T),origin='lower',extent=(0.,self.grid.zvals[-1],0.,self.grid.dmvals[-1]),aspect=aspect)
        cbar=plt.colorbar()
        cbar.set_label('Number of bursts observed repeaters N >= 2')
        plt.clim(cmin,cmax)
        plt.xlim(0,2)
        plt.ylim(0,2000)
        plt.tight_layout()
        plt.savefig(opdir+'summed_reps.pdf')
        plt.close()
        
        
        # this works, producing the right number of repeats
        plt.figure()
        toplot=(summed_array+single_array)/exp_array
        themax=np.nanmax(toplot)
        cmax=1.2
        cmin=0.8
        #cmin=cmax-4
        plt.xlabel('z')
        plt.ylabel('DM')
        plt.imshow(toplot.T,origin='lower',extent=(0.,self.grid.zvals[-1],0.,self.grid.dmvals[-1]),aspect=aspect)
        cbar=plt.colorbar()
        cbar.set_label('Number of bursts from repeaters (single and mult)')
        plt.clim(cmin,cmax)
        plt.xlim(0,2)
        plt.ylim(0,2000)
        plt.tight_layout()
        plt.savefig(opdir+'ratio_exp_reps.pdf')
        plt.close()
        
        plt.figure()
        toplot=np.log10(exp_array)
        themax=np.nanmax(toplot)
        cmax=int(themax)+1
        cmin=cmax-4
        plt.xlabel('z')
        plt.ylabel('DM')
        plt.imshow(toplot.T,origin='lower',extent=(0.,self.grid.zvals[-1],0.,self.grid.dmvals[-1]),aspect=aspect)
        cbar=plt.colorbar()
        cbar.set_label('Expected number of bursts from modelled repeaters')
        plt.clim(-1,3)
        plt.xlim(0,2)
        plt.ylim(0,2000)
        plt.tight_layout()
        plt.savefig(opdir+'expected_repeat_bursts.pdf')
        plt.close()
        
        
        ####### we now make summary plots of the total number of bursts as a function of redshift
        # the total single bursts
        # and the total from repeaters
        
        # Initially, Poisson is unweighted by constants or observation time
        # We now need to multiply by Tobs and the constant
        #Poisson *= self.Tfield * 10**(self.grid.state.FRBdemo.lC)
        TotalSingle = poisson_rates + single_array
        
        plt.figure()
        toplot=TotalSingle
        themax=np.max(toplot)
        cmax=int(np.log10(themax))+1
        cmin=cmax-4
        plt.xlabel('z')
        plt.ylabel('DM')
        plt.imshow(np.log10(toplot.T),origin='lower',extent=(0.,self.grid.zvals[-1],0.,self.grid.dmvals[-1]),aspect=aspect)
        cbar=plt.colorbar()
        cbar.set_label('Actual number of single bursts (analytic + reps)')
        plt.clim(cmin,cmax)
        plt.xlim(0,2)
        plt.ylim(0,2000)
        plt.tight_layout()
        plt.savefig(opdir+'total_single.pdf')
        plt.close()
        
        # wait: grid.rates is maybe wrong, since in this excercise we have ignored beamshape etc. Damn!
        no_repeaters = self.grid.rates * self.Tfield * 10**(self.grid.state.FRBdemo.lC)
        #no_repeaters = MEOW
        
        # this works, producing the right number of repeats
        plt.figure()
        toplot=exp_array/no_repeaters
        themax=np.nanmax(toplot)
        cmax=1.2
        cmin=0.0
        #cmin=cmax-4
        plt.xlabel('z')
        plt.ylabel('DM')
        plt.imshow(toplot.T,origin='lower',extent=(0.,self.grid.zvals[-1],0.,self.grid.dmvals[-1]),aspect=aspect)
        cbar=plt.colorbar()
        cbar.set_label('Ratio of expected repeaters to no repeaters')
        plt.clim(cmin,cmax)
        plt.xlim(0,2)
        plt.ylim(0,2000)
        plt.tight_layout()
        plt.savefig(opdir+'ratio_exp_noreps.pdf')
        plt.close()
        
        
        plt.figure()
        toplot=no_repeaters
        themax=np.max(toplot)
        cmax=int(np.log10(themax))+1
        cmin=cmax-4
        plt.xlabel('z')
        plt.ylabel('DM')
        plt.imshow(np.log10(toplot.T),origin='lower',extent=(0.,self.grid.zvals[-1],0.,self.grid.dmvals[-1]),aspect=aspect)
        cbar=plt.colorbar()
        cbar.set_label('Number of single bursts when excluding repeaters')
        plt.clim(cmin,cmax)
        plt.xlim(0,2)
        plt.ylim(0,2000)
        plt.tight_layout()
        plt.savefig(opdir+'no_repeaters.pdf')
        plt.close()
        
        
        ratio=TotalSingle/no_repeaters
        
        plt.figure()
        plt.xlabel('z')
        plt.ylabel('DM')
        plt.imshow(ratio.T,origin='lower',extent=(0.,self.grid.zvals[-1],0.,self.grid.dmvals[-1]),aspect=aspect)
        cbar=plt.colorbar()
        cbar.set_label('Ratio of single bursts to those expected')
        plt.clim(0.0,0.1)
        plt.xlim(0,2)
        plt.ylim(0,2000)
        plt.tight_layout()
        plt.savefig(opdir+'ts_ratio.pdf')
        plt.close()
        
        total_bursts = TotalSingle + summed_array # single bursts plus bursts from repeaters
        
        plt.figure()
        toplot=total_bursts
        themax=np.max(toplot)
        cmax=int(np.log10(themax))+1
        cmin=cmax-4
        plt.xlabel('z')
        plt.ylabel('DM')
        plt.imshow(np.log10(toplot.T),origin='lower',extent=(0.,self.grid.zvals[-1],0.,self.grid.dmvals[-1]),aspect=aspect)
        cbar=plt.colorbar()
        cbar.set_label('Total number of bursts, including repeaters')
        plt.clim(cmin,cmax)
        plt.xlim(0,2)
        plt.ylim(0,2000)
        plt.tight_layout()
        plt.savefig(opdir+'all_bursts.pdf')
        plt.close()
        
        ratiotot=total_bursts/no_repeaters
        
        plt.figure()
        plt.xlabel('z')
        plt.ylabel('DM')
        plt.imshow(ratiotot.T,origin='lower',extent=(0.,self.grid.zvals[-1],0.,self.grid.dmvals[-1]),aspect=aspect)
        cbar=plt.colorbar()
        cbar.set_label('Ratio of total generated bursts to those expected without repeaters')
        plt.clim(0.0,1.2)
        plt.xlim(0,2)
        plt.ylim(0,2000)
        plt.tight_layout()
        plt.savefig(opdir+'totratio.pdf')
        plt.close()
        
        total_bursts_z=np.sum(total_bursts,axis=1)
        no_repeaters_z=np.sum(no_repeaters,axis=1)
        plt.figure()
        plt.plot(self.grid.zvals,total_bursts_z,label='total bursts')
        plt.plot(self.grid.zvals,no_repeaters_z,label='no repeaters')
        plt.legend()
        plt.xlabel('z')
        plt.ylabel('N(z)')
        plt.tight_layout()
        plt.savefig(opdir+'total_norep_bursts_fz.pdf')
        plt.close()
        
        
        totratio_z=total_bursts_z/no_repeaters_z
        plt.figure()
        plt.plot(self.grid.zvals,totratio_z)
        plt.xlabel('z')
        plt.ylabel('N(z)')
        plt.tight_layout()
        plt.savefig(opdir+'total_norep_bursts_fz_ratio.pdf')
        plt.close()
        
    def MCsample(self,Rthresh,volume_grid,Rmult):
        """
        Samples FRBs from Rthresh upwards
        
        Rmult scales a repeater rate to an expected Nevents rate
        
        Rthresh alrady has the Tfield baked into it.
        Rmult also has the Tfield baked into it.
        
        volume_grid should be in units of Mpc^3, accounting for solid angle and dV/dz/dOmega
        
        """
        opdir='Repeaters/'
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
        
        # Rmult here is the rate multiplier due to the distance
        # that is, mean_rate is rate per repeater on average, n reps is number of repeaters, and
        # Rmult is the scaling between the repeater rate and the observed rate
        expected_bursts = expected_number_of_repeaters * mean_rate * Rmult
        
        aspect=(self.grid.zvals[-1]/self.grid.dmvals[-1])
        plt.figure()
        plt.xlabel('z')
        plt.ylabel('DM')
        plt.imshow(np.log10(expected_bursts.T),origin='lower',extent=(0.,self.grid.zvals[-1],0.,self.grid.dmvals[-1]),aspect=aspect)
        cbar=plt.colorbar()
        cbar.set_label('Expected number of bursts from repeaters')
        plt.clim(-1,3)
        plt.xlim(0,2)
        plt.ylim(0,2000)
        plt.tight_layout()
        plt.savefig(opdir+'expected_bursts.pdf')
        plt.close()
        
        Nreps = np.random.poisson(expected_number_of_repeaters)
        sampled_expected = Nreps * mean_rate * Rmult
        
        plt.figure()
        plt.xlabel('z')
        plt.ylabel('DM')
        plt.imshow(np.log10(sampled_expected.T),origin='lower',extent=(0.,self.grid.zvals[-1],0.,self.grid.dmvals[-1]),aspect=aspect)
        cbar=plt.colorbar()
        cbar.set_label('Expected Nfrb from repeaters (after sampling Nrep)')
        plt.clim(-1,3)
        plt.xlim(0,2)
        plt.ylim(0,2000)
        plt.tight_layout()
        plt.savefig(opdir+'expected_bursts_w_poisson_nrep.pdf')
        plt.close()
        
        
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
                    
                if i==100 and j==100:
                    Rs=self.GenNReps(Rthresh[i,j],1000)
                    plt.figure()
                    plt.hist(np.log10(Rs),label=str(Rthresh[i,j]))
                    plt.legend()
                    plt.xlabel('log10(R)')
                    plt.ylabel('N(log10(R))')
                    plt.yscale('log')
                    plt.tight_layout()
                    plt.savefig(opdir+'repeater_rate_hist.pdf')
                    plt.close()
        
        
        return expected_number_of_repeaters,Nreps,single_array,mult_array,mean_array,exp_array
    
    def GenNReps(self,Rthresh,Nreps):
        """
        Samples N random FRB rates R above Rthresh
        
        CDF [integral Rstar to Rmax] = (R**Rgamma - Rthresh**gamma) / (Rmax**Rgamma - Rthresh**gamma)
        
        """
        rand=np.random.rand(Nreps)
        Rs = ((self.Rmax**(self.Rgamma+1) - Rthresh**(self.Rgamma+1))*rand + Rthresh**(self.Rgamma+1))**(1./(self.Rgamma+1))
        return Rs
    
    
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
    
    
    
    
