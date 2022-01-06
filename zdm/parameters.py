from IPython.terminal.embed import embed
import numpy as np
from dataclasses import dataclass, field, asdict
from typing import IO, List
from astropy.cosmology import Planck18

import pandas
import json

from zdm import io

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
    method: int = field(
        default=2,
        metadata={'help': 'Method for calculation. See beams.py:simplify_beam() for options'})
    Wbins: int = field(
        default=5,
        metadata={'help': '???'})
    Wscale: int = field(
        default=3.5,
        metadata={'help': '???'})

    #def __post_init__(self):
    #    self.Nbeams = [5,5,5,10]

# Cosmology parameters
@dataclass
class CosmoParams(myDataClass):
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
    sfr_n: float = field(
        default = 1.77,
        metadata={'help': 'scaling with star-formation rate',
                  'unit': '',
                  'Notation': 'n',
                  })
    lC: float = field(
        default = 4.19,
        metadata={'help': 'log10 constant in number per Gpc^-3 yr^-1 at z=0'})

# Galactic parameters
@dataclass
class MWParams(myDataClass):
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
class IGMParams(myDataClass):
    F: float = field(
        default=0.32,
        metadata={'help': 'F parameter in DM$_{\\rm cosmic}$ PDF for the Cosmic web',
                  'unit': '',
                  'Notation': 'F',
        })

# Host parameters -- host
@dataclass
class HostParams(myDataClass):
    lmean: float = field(
        default=2.16,
        metadata={'help': 'log10 mean of DM host contribution in pc cm^-3',
                  'unit': '',
                  'Notation': '\mu_{\\rm host}',
        })
    lsigma: float = field(
        default=0.51,
        metadata={'help': 'log10 sigma of DM host contribution in pc cm^-3',
                  'unit': '',
                  'Notation': '\sigma_{\\rm host}',
        })

# FRB intrinsic width parameters
@dataclass
class WidthParams(myDataClass):
    logmean: float = field(
        default = 1.70267, 
        metadata={'help': 'Intrinsic width log of mean',
                  'unit': 'ms',
                  'Notation': '\mu_{w}',
                  })
    logsigma: float = field(
        default = 0.899148,
        metadata={'help': 'Intrinsic width log of sigma',
                  'unit': 'ms',
                  'Notation': '\sigma_{w}',
                  })

# FRB Energetics -- energy
@dataclass
class EnergeticsParams(myDataClass):
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

class State:
    """ Initialize the full state for the analysis 
    with the default parameters

    Returns:
        dict: [description]
    """
    def __init__(self):

        self.width = WidthParams()
        self.MW = MWParams()
        self.analysis = AnalysisParams()
        self.beam = BeamParams()
        self.FRBdemo = FRBDemoParams()
        self.cosmo = CosmoParams()
        self.host = HostParams()
        self.IGM = IGMParams()
        self.energy = EnergeticsParams()

        # Look-up table or convenience
        self.params = {}
        for dc_key in self.__dict__.keys():
            if dc_key == 'params':
                continue
            for param in self[dc_key].__dict__.keys():
                self.params[param] = dc_key

    def __getitem__(self, attrib:str):
        """Enables dict like access to the state

        Args:
            attrib (str): [description]

        Returns:
            [type]: [description]
        """
        return getattr(self, attrib)

    def update_param_dict(self, params:dict):
        for key in params.keys():
            idict = params[key]
            for ikey in idict.keys():
                # Set
                self.update_param(ikey, params[key][ikey])

    def update_params(self, params:dict):
        for key in params.keys():
            self.update_param(key, params[key])

    def update_param(self, param:str, value):
        DC = self.params[param]
        setattr(self[DC], param, value)
        # Special treatment
        if DC == 'cosmo' and param == 'H0':
            if self.cosmo.fix_Omega_b_h2:
                self.cosmo.Omega_b = self.cosmo.Omega_b_h2/(
                    self.cosmo.H0/100.)**2

    '''
    def unpack_pset(self, mode:str='H0_std'):
        if mode == 'H0_std':
            return [
                params['energy'].lEmin,
                params['energy'].lEmax,
                params['energy'].alpha,
                params['energy'].gamma,
                params['FRBdemo'].sfr_n,
                params['host'].lmean,
                params['host'].lsigma,
                params['FRBdemo'].lC,
                params['cosmo'].H0,
            ]
        else:
            raise IOError('Bad mode')
    '''

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

    def to_dict(self):
        items = []
        for key in self.params.keys():
            items.append(self.params[key])
        uni_items = np.unique(items)
        #
        state_dict = {}
        for uni_item in uni_items:
            state_dict[uni_item] = asdict(getattr(self, uni_item))
        # Return
        return state_dict

    def write(self, outfile):
        state_dict = self.to_dict()
        io.savejson(outfile, state_dict, overwrite=True, easy_to_read=True)

    def vet(self, obj, dmodel:dict, verbose=True):
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

    def __repr__(self) -> str:
        return json.dumps(self.to_dict(), 
                   sort_keys=True, indent=4,
                   separators=(',', ': '))