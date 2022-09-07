from IPython.terminal.embed import embed
import numpy as np
from dataclasses import dataclass, field
from typing import IO, List
from astropy.cosmology import Planck18


from zdm import data_class


# Analysis parameters
@dataclass
class AnalysisParams(data_class.myDataClass):
    NewGrids: bool = field(
        default=True,
        metadata={'help': 'Generate new z, DM grids?'})
    sprefix: str = field(
        default='Std',
        metadata={'help': 'Full:  more detailed estimates. Takes more space and time \n'+\
                 'Std: faster - fine for max likelihood calculations, not as pretty'})

# Beam parameters
@dataclass
class BeamParams(data_class.myDataClass):
    Bmethod: int = field(
        default=2,
        metadata={'help': 'Method for calculation. See beams.py:simplify_beam() for options',
                  'unit': ''})
    Bthresh: float = field(
        default=0.0,
        metadata={'help': 'Minimum value of beam sensitivity to consider',
                  'unit': '',
                  'Notation': 'B_{\rm min}'})
    #def __post_init__(self):
    #    self.Nbeams = [5,5,5,10]

# Cosmology parameters
@dataclass
class CosmoParams(data_class.myDataClass):
    H0: float = field(
        default=Planck18.H0.value,
        metadata={'help': "Hubble's constant", 
                  'unit': 'km s$^{-1}$ Mpc$^{-1}$',
                  'Notation': 'H_0',
                  })
    Omega_k: float = field(
        default=0.,
        metadata={'help': 'photo density. Ignored here (we do not go back far enough)'})
    Omega_lambda: float = field(
        default=Planck18.Ode0,
        metadata={'help': 'dark energy / cosmological constant (in current epoch)',
                  'unit': '',
                  'Notation': '\Omega_{\Lambda}',
                  })
    Omega_m: float = field(
        default=Planck18.Om0,
        metadata={'help': 'matter density in current epoch',
                  'unit': '',
                  'Notation': '\Omega_m',
                  })
    Omega_b: float = field(
        default=Planck18.Ob0,
        metadata={'help': 'baryon density in current epoch',
                  'unit': '',
                  'Notation': '\Omega_b',
                  })
    Omega_b_h2: float = field(
        default=Planck18.Ob0 * (Planck18.H0.value/100.)**2,
        metadata={'help': 'Baryon density weighted by $h_{100}^2$',
                  'unit': '',
                  'Notation': '\Omega_b h^2',
                  })
    fix_Omega_b_h2: bool = field(
        default=True,
        metadata={'help': 'Fix Omega_b_h2 by the Placnk18 value?'})

# FRB Demographics -- FRBdemo
@dataclass
class FRBDemoParams(data_class.myDataClass):
    source_evolution: int = field(
        default=0,
        metadata={'help': 'Integer flag specifying the function used.  '+\
                 '0: SFR^n; 1: (1+z)^(2.7n)', 
                 'options': [0,1]}) 
    alpha_method: int = field(
        default=0, 
        metadata={'help': 'Integer flag specifying the nature of scaling. '+\
                 '0: spectral index interpretation: includes k-correction. Slower to update ' +\
                 '1: rate interpretation: extra factor of (1+z)^alpha in source evolution', 
                 'options': [0,1]})
    sfr_n: float = field(
        default = 1.77,
        metadata={'help': 'scaling with star-formation rate',
                  'unit': '',
                  'Notation': 'n',
                  })
    lC: float = field(
        default = 4.19,
        metadata={'help': 'log10 constant in number per Gpc^-3 yr^-1 at z=0'})


# FRB Demographics -- repeaters
@dataclass
class RepeatParams(data_class.myDataClass):
    Rmin: float = field(
        default=1e-3,
        metadata={'help': 'Minimum repeater rate',
                  'unit': 'day^-1',
                  'Notation': '$R_{\rm min}$',
                  })
    Rmax: float = field(
        default=10,
        metadata={'help': 'Maximum repeater rate',
                  'unit': 'day^-1',
                  'Notation': '$R_{\rm max}$',
                  })
    Rgamma: float = field(
        default = -2.375,
        metadata={'help': 'differential index of repeater density',
                  'unit': '',
                  'Notation': '$\gamma_r$',
                  })
    RC: float = field(
        default = 1e-2,
        metadata={'help': 'Constant repeater density',
                  'unit': 'Repeaters day / Gpc^-3',
                  'Notation': '$C_R$',
                  })
    RE0: float = field(
        default = 1.e38,
        metadata={'help': 'Energy at which rates are defined',
                  'unit': 'erg',
                  'Notation': '$E_R$',
                  })
                  

# Galactic parameters
@dataclass
class MWParams(data_class.myDataClass):
    ISM: float = field(
        default=35.,
        metadata={'help': 'Assumed DM for the Galactic ISM in units of pc/cm^3'})
    DMhalo: float = field(
        default=50.,
        metadata={'help': 'DM for the Galactic halo',
                  'unit': 'pc cm$^{-3}$',
                  'Notation': '{\\rm DM}_{\\rm halo}',
        })
# IGM parameters
@dataclass
class IGMParams(data_class.myDataClass):
    F: float = field(
        default=0.32,
        metadata={'help': 'F parameter in DM$_{\\rm cosmic}$ PDF for the Cosmic web',
                  'unit': '',
                  'Notation': 'F',
        })

# Host parameters -- host
@dataclass
class HostParams(data_class.myDataClass):
    lmean: float = field(
        default=2.16,
        metadata={'help': 'log10 mean of DM host contribution in pc cm$^{-3}$',
                  'unit': '',
                  'Notation': '\mu_{\\rm host}',
        })
    lsigma: float = field(
        default=0.51,
        metadata={'help': 'log10 sigma of DM host contribution in pc cm$^{-3}$',
                  'unit': '',
                  'Notation': '\sigma_{\\rm host}',
        })

# FRB intrinsic width parameters
@dataclass
class WidthParams(data_class.myDataClass):
    Wlogmean: float = field(
        default = 1.70267, 
        metadata={'help': 'Intrinsic width log of mean',
                  'unit': 'ms',
                  'Notation': '\mu_{w}',
                  })
    Wlogsigma: float = field(
        default = 0.899148,
        metadata={'help': 'Intrinsic width log of sigma',
                  'unit': 'ms',
                  'Notation': '\sigma_{w}',
                  })
    Wthresh: int = field(
        default=0.5,
        metadata={'help': 'Starting fraction of intrinsic width for histogramming',
                  'unit': '',
                  'Notation': 'w_{\rm min}'})
    Wmethod: int = field(
        default=2,
        metadata={'help': 'Method of calculating FRB widths; 1 std, 2 includes scattering',
                  'unit': ''})
    Wbins: int = field(
        default=5,
        metadata={'help': 'Number of bins for FRB width distribution',
                  'unit': ''})
    Wscale: int = field(
        default=3.5,
        metadata={'help': 'Log-scaling of bins for width distribution',
                  'unit': ''})
    
# FRB intrinsic scattering parameters
@dataclass
class ScatParams(data_class.myDataClass):
    Slogmean: float = field(
        default = 0.7, 
        metadata={'help': 'Intrinsic width log of mean',
                  'unit': 'ms',
                  'Notation': '\tau_{s}',
                  })
    Slogsigma: float = field(
        default = 1.9,
        metadata={'help': 'Intrinsic width log of sigma',
                  'unit': 'ms',
                  'Notation': '\sigma_{\tau}',
                  })
    Sfnorm: float = field(
        default = 600,
        metadata={'help': 'Frequency of scattering width',
                  'unit': 'MHz',
                  'Notation': '\nu_{\tau}',
                  })
    Sfpower: float = field(
        default = -4.,
        metadata={'help': 'Power-law scaling with frequency, nu^lambda',
                  'unit': '',
                  'Notation': '\lambda',
                  })

# FRB Energetics -- energy
@dataclass
class EnergeticsParams(data_class.myDataClass):
    lEmin: float = field(
        default = 30.,
        metadata={'help': 'log10 of minimum energy ',
                  'unit': 'erg',
                  'Notation': 'E_{\\rm min}',
                  })
    lEmax: float = field(
        default = 41.84,
        metadata={'help': 'log 10 of maximum energy',
                  'unit': 'erg',
                  'Notation': 'E_{\\rm max}',
                  })
    alpha: float = field(
        default = 1.54,
        metadata={'help': 'spectral index',
                  'unit': '',
                  'Notation': '\\alpha',
                  })
    gamma: float = field(
        default = -1.16,
        metadata={'help': 'slope of luminosity distribution function',
                  'unit': '',
                  'Notation': '\gamma',
                  })
    luminosity_function: int = field(
        default = 0,
        metadata={'help': 'luminosity function applied (0=power-law, 1=gamma)'})

class State(data_class.myData):
    """ Initialize the full state for the analysis 
    with the default parameters

    """
    def __init__(self):

        self.set_dataclasses()

        self.set_params()


    def set_dataclasses(self):
        self.scat = ScatParams()
        self.width = WidthParams()
        self.MW = MWParams()
        self.analysis = AnalysisParams()
        self.beam = BeamParams()
        self.FRBdemo = FRBDemoParams()
        self.cosmo = CosmoParams()
        self.host = HostParams()
        self.IGM = IGMParams()
        self.energy = EnergeticsParams()
        self.rep = RepeatParams()

    def update_param(self, param:str, value):
        DC = self.params[param]
        setattr(self[DC], param, value)
        # Special treatment
        if DC == 'cosmo' and param == 'H0':
            if self.cosmo.fix_Omega_b_h2:
                self.cosmo.Omega_b = self.cosmo.Omega_b_h2/(
                    self.cosmo.H0/100.)**2

    def set_astropy_cosmo(self, cosmo):
        """Slurp the values from an astropy Cosmology object
        into our format

        Args:
            cosmo (astropy.cosmology): [description]
        """
        self.cosmo.H0 = cosmo.H0.value
        self.cosmo.Omega_lambda = cosmo.Ode0
        self.cosmo.Omega_m = cosmo.Om0
        self.cosmo.Omega_b = cosmo.Ob0
        self.cosmo.Omega_b_h2 = cosmo.Ob0 * (cosmo.H0.value/100.)**2
        return
