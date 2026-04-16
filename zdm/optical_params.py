"""
Classes for optical properties

This philosophy here is to have a class of key parameters that relates to
a single class object contained within optical.py. The dataclasses are
used to set parameters that initialise their parent classes, which is
where all the complicated calculations are performed.
"""

from dataclasses import dataclass, field

from zdm import data_class
import numpy as np



# Simple SFR model
@dataclass
class SimpleParams(data_class.myDataClass):
    """
    Data class to hold the generic host galaxy class with no
    pre-specified model
    
    """
    # None of the fields should start with an X
    Absmin: float = field( 
        default=-25, 
        metadata={'help': "Minimum host absolute magnitude", 
                  'unit': 'M_r^{min}', 
                  'Notation': '',
                  })
    Absmax: float = field( 
        default=-15., 
        metadata={'help': "Maximum host absolute magnitude", 
                  'unit': 'M_r^{max}', 
                  'Notation': '',
                  })
    NAbsBins: int = field( 
        default=1000, 
        metadata={'help': "Number of absolute magnitude bins used internally",
                  'unit': '', 
                  'Notation': '',
                  })
    NModelBins: int = field( 
        default=10, 
        metadata={'help': "Number of absolute magnitude bins used in the model",
                  'unit': '', 
                  'Notation': '',
                  })
    AbsPriorMeth: int = field( 
        default=0, 
        metadata={'help': "Model for abs mag prior and function description. 0: uniform distribution. Others to be implemented.",
                  'unit': '', 
                  'Notation': '',
                  })
    AppModelID: int = field( 
        default=0, 
        metadata={'help': "Model for converting absolute to apparent magnitudes. 0: no k-correction. 1: with k-correctionOthers to be implemented.",
                  'unit': '', 
                  'Notation': '',
                  })
    AbsModelID: int = field( 
        default=1, 
        metadata={'help': "Model for describing absolute magnitudes. 0: Simple histogram of absolute magnitudes. 1: linear interpolation, 2: spline interpolation of histogram, 3: spline interpolation in log space",
                  'unit': '', 
                  'Notation': '',
                  })
    k: float = field( 
        default=0., 
        metadata={'help': "k-correction",
                  'unit': '', 
                  'Notation': 'k',
                  })


# Nick Loudas's SFR model. Actually, we never really need many/any of these, it's all contained in optical
@dataclass
class LoudasParams(data_class.myDataClass):
    """
    Data class to hold the SFR model from Nick, which models
    FRBs as some fraction of the star-formation rate.
    """
    fSFR: float = field( 
        default=0.5, 
        metadata={'help': "Fraction of FRBs associated with star-formation", 
                  'unit': '', 
                  'Notation': '',
                  })
    NzBins: int = field( 
        default=10, 
        metadata={'help': "Number of redshift bins over which the histograms are calculated",
                  'unit': '', 
                  'Notation': '',
                  })
    zmin: float = field( 
        default=0., 
        metadata={'help': "Minimum redshift over which pmag is calculated",
                  'unit': '', 
                  'Notation': '',
                  })
    zmax: float = field( 
        default=0., 
        metadata={'help': "Maximum redshift over which pmag is calculated",
                  'unit': '', 
                  'Notation': '',
                  })
    NMrBins: int = field( 
        default=0., 
        metadata={'help': "Number of magnitude bins",
                  'unit': '', 
                  'Notation': '',
                  })
    Mrmin: float = field( 
        default=0., 
        metadata={'help': "Minimum absolute magnitude over which pmag is calculated",
                  'unit': '', 
                  'Notation': '',
                  })
    Mrmax: float = field( 
        default=0., 
        metadata={'help': "Maximum magnitude over which pmag is calculated",
                  'unit': '', 
                  'Notation': '',
                  })


@dataclass
class Identification(data_class.myDataClass):
    """
    Data class for holding parameters related to identifying galaxies in an image.
    These describe the mean and deviation for the function pUgm in optical.py
    that parameterises p(U|mr)
    """
    pU_mean: float = field( 
        default=26.176, 
        metadata={'help': "Magnitude at which pU|mr is 0.5", 
                  'unit': '', 
                  'Notation': '',
                  })
    pU_width: float = field( 
        default=0.342, 
        metadata={'help': "Width of pU|mr distribution in ln space", 
                  'unit': '', 
                  'Notation': '',
                  })

@dataclass
class Apparent(data_class.myDataClass):
    """
    # parameters for apparent mags - used by wrapper
    """
    Appmin: float = field( 
        default=10, 
        metadata={'help': "Minimum host apparent magnitude", 
                  'unit': 'm_r^{min}', 
                  'Notation': '',
                  })
    Appmax: float = field( 
        default=35, 
        metadata={'help': "Maximum host apparent magnitude", 
                  'unit': 'm_r^{max}', 
                  'Notation': '',
                  })
    NAppBins: int = field( 
        default=250, 
        metadata={'help': "Number of apparent magnitude bins",
                  'unit': '', 
                  'Notation': '',
                  })
    iModel: int = field(
        default = 0,
        metadata={'help': "Id of optical model to use. 0,1,2 = Marnoch, Loudas, Naive",
                  'unit': '',
                  'Notation': '',
                  })



@dataclass
class Path(data_class.myDataClass):
    """
    parameters to pass along to PATH analysis.
    """
    Scale: float = field( 
        default=0.5, 
        metadata={'help': "Exponential scale of radial distribution", 
                  'unit': '}', 
                  'Notation': '',
                  })
    
class OpticalState(data_class.myData):
    """Initialize the full optical state dataset
    with the default parameters

    """

    def __init__(self):
        self.set_dataclasses()
        self.set_params()

    def set_dataclasses(self):
        self.simple = SimpleParams()
        self.loudas = LoudasParams()
        self.app = Apparent()
        self.id = Identification()
        self.path = Path()
        

    def update_param(self, param:str, value):
        # print(self.params)
        DC = self.params[param]
        setattr(self[DC], param, value)
        
