
from zdm import grid


class Grid:
    """
    This class is designed to take a p(z,DM) grid and calculate the
    effects of repeating FRBs. The standard Grid class assumes bursts are
    independent. Here we remove that restriction.
    
    """
    
    
    def __init__(self, grid):
        """
        Initialises repeater class
        Args:
            grid: zdm grid class object
            Nfields: number of separate pointings by the survey
            Tfield: time spent on each field
        """
        
        # inherets key properties from the grid's state parameters
        self.state=grid.state
        self.Rmin=grid.state.repeater.Rmin
        self.Rmax=grid.state.repeater.Rmax
        self.power=grid.state.repeater.Rpower
        
        # redshift array
        self.zvals=self.grid.zvals
        
        # key survey properties - extended data beyond what is normally required
        self.Nfields = Nfields
        self.Tfield = Tfield
        
        # sets repeater constant as per global FRB rate constant
        C=10**(grid.state.FRBdemo.lC)
        self.Rc=self.calc_constant(C)
        grid.state.repeater.Rc=self.Rc
        
        # number of expected FRBs per volume in a given field
        self.Nexp_field = grid.dV*self.Rc
        self.Nexp = self.Nexp_field * self.Nfields
        
    def calc_p_no_repeaters(self,Npointings):
        """
        Calculates the probability per volume that there are no progenitors there
        By default, does this for a single pointing AND all pointings combined
        """
        self.Pnone = np.exp(-self.Nexp)
        self.Pnone_field= = np.exp(-self.Nexp_field)
        
    def calc_stochastic_distance(self,p)
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

    def calc_p_no_bursts(Tobs,N1thresh=0.1,N3thresh=10.):
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
    """ integrates Poisson probabilities """
    
    for i in np.arange(20)+1: # number of repeaters - already done 0!
        pz
        
        # total p_zero given N: AND condition
            #pz is the probability of detecting a burst from that repeaters
            pz_total = poisson_N*pz**N 
            
        #now: chance of a burst given repeaters
        while(1):
            # Poisson probability of N repeaters: loop over N, keep going until p is negligible
            
            
            # chance of zero bursts per repeaters
            np.integrate.quad(N1,N3,
            
            
        
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
    
    
    
def calc_constant(C):
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
    
    Cr = C * (self.Rpower+2.)/(self.Rmax^(self.Rpower+2.)-self.Rmin^(self.Rpower+2.))
    return Cr
