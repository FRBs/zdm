""" Classes for optical properties """

from dataclasses import dataclass, field

from zdm import data_class
import numpy as np

@dataclass
class Hosts(data_class.myDataClass):
    """
    Data class to hold the generic host galaxy class with no
    pre-specified model
    
    """
    # None of the fields should start with an X
    Absmin: float = field( 
        default=-30, 
        metadata={'help': "Minimum host absolute magnitude", 
                  'unit': 'M_r^{min}', 
                  'Notation': '',
                  })
    Absmax: float = field( 
        default=0., 
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
    AbsPriorMeth: int = field( 
        default=0, 
        metadata={'help': "Model for abs mag prior and function description. 0: uniform distribution. Others to be implemented.",
                  'unit': '', 
                  'Notation': '',
                  })
    AppModelID: int = field( 
        default=0, 
        metadata={'help': "Model for converting absolute to apparent magnitudes. 0: no k-correction. Others to be implemented.",
                  'unit': '', 
                  'Notation': '',
                  })
    AbsModelID: int = field( 
        default=0, 
        metadata={'help': "Model for describing absolute magnitudes. 0: Simple histogram of absolute magnitudes. 1: spline interpolation of histogram.",
                  'unit': '', 
                  'Notation': '',
                  })

class SFRmodel(data_class.myDataClass):
    """
    Data class to hold the SFR model from Nick, which models
    FRBs as some fraction of the star-formation rate
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
