""" Classes for optical properties """

from dataclasses import dataclass, field

from zdm import data_class
import numpy as np

@dataclass
class Hosts(data_class.myDataClass):
    # None of the fields should start with an X
    Absmin: float = field( 
        default=-30, 
        metadata={'help': "Minimum host absolute magnitude", 
                  'unit': 'L*', 
                  'Notation': '',
                  })
    Absmax: float = field( 
        default=0., 
        metadata={'help': "Maximum host absolute magnitude", 
                  'unit': 'L*', 
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
                  'unit': 'L*', 
                  'Notation': '',
                  })
    Appmax: float = field( 
        default=100, 
        metadata={'help': "Maximum host apparent magnitude", 
                  'unit': 'L*', 
                  'Notation': '',
                  })
    NAppBins: int = field( 
        default=300, 
        metadata={'help': "Number of apparent magnitude bins",
                  'unit': '', 
                  'Notation': '',
                  })
    AbsPriorMeth: int = field( 
        default=0, 
        metadata={'help': "Model for abs mag prior and function description",
                  'unit': '', 
                  'Notation': '',
                  })
    AppModelID: int = field( 
        default=0, 
        metadata={'help': "Model for converting to apparent magnitudes",
                  'unit': '', 
                  'Notation': '',
                  })
    AbsModelID: int = field( 
        default=0, 
        metadata={'help': "Model for describing absolute magnitudes",
                  'unit': '', 
                  'Notation': '',
                  })
