from IPython.terminal.embed import embed
import numpy as np
from dataclasses import dataclass, field
from typing import IO, List
from astropy.cosmology import Planck18

# Add a few methods to be shared by them all
@dataclass
class myDataClass:
    def meta(self, attribute_name):
        return self.__dataclass_fields__[attribute_name].metadata
    def chk_options(self, attribute_name):
        options = self.__dataclass_fields__[attribute_name].metadata['options']

# Analysis parameters
@dataclass
class AnalysisParams(myDataClass):
    NewGrids: bool = field(
        default=True,
        metadata={'help': 'Generate new z, DM grids?'})
    sprefix: str = field(
        default='Std',
        metadata={'help': 'Full:  more detailed estimates. Takes more space and time \n'+\
                 'Std: faster - fine for max likelihood calculations, not as pretty'})

# Beam parameters
@dataclass
class BeamParams(myDataClass):
    thresh: int = field(
        default=0,
        metadata={'help': '??'})
    method: str = field(
        default='Std',
        metadata={'help': 'Method for calculation. Full=more detailed and more time. Std=fast'})
    Nbeams: List[int] = field(  # can use in list in 3.9
        default_factory=list,
        #default_factory=[5,5,5,10],
        metadata={'help': '???'})
    Wbins: int = field(
        default=5,
        metadata={'help': '???'})
    Wscale: int = field(
        default=3.5,
        metadata={'help': '???'})

    def __post_init__(self):
        self.Nbeams = [5,5,5,10]

# Cosmology parameters
@dataclass
class CosmoParams(myDataClass):
    current_H0: float = field(
        default=Planck18.H0.value,
        metadata={'help': "Current H0 in the analysis"})
    H0: float = field(
        default=Planck18.H0.value,
        metadata={'help': "Hubble's constant (km/s/Mpc)"})
    Omega_k: float = field(
        default=0.,
        metadata={'help': 'photo density. Ignored here (we do not go back far enough)'})
    Omega_lambda: float = field(
        default=Planck18.Ode0,
        metadata={'help': 'dark energy / cosmological constant (in current epoch)'})
    Omega_m: float = field(
        default=Planck18.Om0,
        metadata={'help': 'matter density in current epoch'})
    Omega_b: float = field(
        default=Planck18.Ob0,
        metadata={'help': 'baryon density'})
    Omega_b_h2: float = field(
        default=Planck18.Ob0 * (Planck18.H0.value/100.)**2,
        metadata={'help': 'baryon density weight by h_100**2'})

# FRB Demographics
@dataclass
class FRBDemoParams(myDataClass):
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
    gamma: float = field(
        default = -1.16,
        metadata={'help': 'slope of luminosity distribution function'})
    sfr_n: float = field(
        default = 1.77,
        metadata={'help': 'scaling with star-formation rate'})
    lC: float = field(
        default = 4.19,
        metadata={'help': 'log10 constant in number per Gpc^-3 yr^-1 at z=0'})

# Galactic parameters
@dataclass
class MWParams(myDataClass):
    DMhalo: float = field(
        default=50.,
        metadata={'help': 'DM for the Galactic halo in units of pc/cm^3'})

# Host parameters
@dataclass
class HostParams(myDataClass):
    lmean: float = field(
        default=2.16,
        metadata={'help': 'log10 mean of DM host contribution in pc cm^-3'})
    lsigma: float = field(
        default=0.51,
        metadata={'help': 'log10 sigma of DM host contribution in pc cm^-3'})

# FRB intrinsic width parameters
@dataclass
class WidthParams(myDataClass):
    logmean: float = field(
        default = 1.70267, 
        metadata={'help': 'Intrinsic width log of mean'})
    logsigma: float = field(
        default = 0.899148,
        metadata={'help': 'Intrinsic width log of sigma'})

# FRB Energetics
@dataclass
class EnergeticsParams(myDataClass):
    lEmin: float = field(
        default = 30.,
        metadata={'help': 'Minimum energy.  log10 in erg'})
    lEmax: float = field(
        default = 41.84,
        metadata={'help': 'Maximum energy.  log10 in erg'})
    alpha: float = field(
        default = 1.54,
        metadata={'help': 'spectral index. WARNING: here F(nu)~nu^-alpha in the code, opposite to the paper!'})

def init_parameters():

    # Begin
    param_dict = {}

    # Wrap em together
    param_dict['width'] = WidthParams()
    param_dict['MW'] = MWParams()
    param_dict['analysis'] = AnalysisParams()
    param_dict['beam'] = BeamParams()
    param_dict['FRBdemo'] = FRBDemoParams()
    param_dict['cosmo'] = CosmoParams()
    param_dict['host'] = HostParams()
    param_dict['energy'] = EnergeticsParams()

    return param_dict


def unpack_pset(params:dict, mode:str='H0_std'):
    if mode == 'H0_std':
        return [
            params['energy'].lEmin,
            params['energy'].lEmax,
            params['energy'].alpha,
            params['FRBdemo'].gamma,
            params['FRBdemo'].sfr_n,
            params['host'].lmean,
            params['host'].lsigma,
            params['FRBdemo'].lC,
            params['cosmo'].current_H0,
        ]
    else:
        raise IOError('Bad mode')

def vet_param(obj, dmodel:dict, verbose=True):
    """ Vet the input object against its data model

    Args:
        obj (dict or pandas.DataFrame):  Instance of the data model
        dmodel (dict): Data model
        verbose (bool): Print when something doesn't check

    Returns:
        tuple: chk (bool), disallowed_keys (list), badtype_keys (list)
    """

    chk = True
    # Loop on the keys
    disallowed_keys = []
    badtype_keys = []
    for key in obj.keys():
        # In data model?
        if not key in dmodel.keys():
            disallowed_keys.append(key)
            chk = False
            if verbose:
                print("Disallowed key: {}".format(key))

        # Check data type
        iobj = obj[key].values if isinstance(obj, pandas.DataFrame) else obj[key]
        if not isinstance(iobj,
                          dmodel[key]['dtype']):
            badtype_keys.append(key)
            chk = False        
            if verbose:
                print("Bad key type: {}".format(key))
    # Return
    return chk, disallowed_keys, badtype_keys