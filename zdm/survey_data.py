""" Survey data """

from dataclasses import dataclass, field

from zdm import data_class

@dataclass
class FRB(data_class.myDataClass):
    # None of the fields should start with an X
    BW: float = field( 
        default=0., 
        metadata={'help': "Mean Frequency (observed)", 
                  'unit': 'MHz', 
                  'Notation': '',
                  })
    DM: float = field( 
        default=0., 
        metadata={'help': "Measured DM",
                  'unit': 'pc/cm**3', 
                  'Notation': '',
                  })
    DMG: float = field( 
        default=0., 
        metadata={'help': "Galactic contribution to DM",
                  'unit': 'pc/cm**3', 
                  'Notation': '',
                  })
    FBAR: float = field( 
        default=0., 
        metadata={'help': "Mean frequency",
                  'unit': 'MHz', 
                  'Notation': '',
                  })
    FRES: float = field( 
        default=1., 
        metadata={'help': "Frequency resolution",
                  'unit': 'MHz', 
                  'Notation': '',
                  })
    Gb: float = field( 
        default=1., 
        metadata={'help': "Galactic latitude",
                  'unit': 'deg', 
                  'Notation': '',
                  })
    Gl: float = field( 
        default=1., 
        metadata={'help': "Galactic longitude",
                  'unit': 'deg', 
                  'Notation': '',
                  })
    SNR: float = field( 
        default=0., 
        metadata={'help': "S/N", 
                  'unit': '', 
                  'Notation': '',
                  })
    SNRTHRESH: float = field( 
        default=0., 
        metadata={'help': "S/N threshold to detect an FRB", 
                  'unit': '', 
                  'Notation': '',
                  })
    THRESH: float = field( 
        default=1., 
        metadata={'help': "Threshold fluence used to detect an FRB", 
                  'unit': 'Jy ms', 
                  'Notation': '',
                  })
    TNS: str = field(
        default='', 
        metadata={'help': "TNS Name", 
                  })
    TRES: float = field( 
        default=0., 
        metadata={'help': "Time resolution",
                  'unit': 'ms', 
                  'Notation': '',
                  })
    WIDTH: float = field( 
        default=0.1, 
        metadata={'help': "Width of the event (intrinsic??)", 
                  'unit': 'ms', 
                  'Notation': '',
                  })
    Z: float = field( 
        default=-1., 
        metadata={'help': "redshift; -1 means unlocalised", 
                  'unit': '', 
                  'Notation': '',
                  })

@dataclass
class Telescope(data_class.myDataClass):
    BEAM: str = field(
        default='', 
        metadata={'help': "Beam file", 
                  })
    BW: float = field( 
        default=336., 
        metadata={'help': "Mean Frequency (observed)", 
                  'unit': 'MHz', 
                  'Notation': '',
                  })
    DIAM: float = field(
        default=0., 
        metadata={'help': "Individual antenna diameter", 
                  'unit': 'm', 
                  'Notation': '',
                  })
    DMG: float = field( 
        default=30., 
        metadata={'help': "Galactic contribution to DM",
                  'unit': 'pc/cm**3', 
                  'Notation': '',
                  })
    NBEAMS: int = field(
        default=1, 
        metadata={'help': "Number of beams/antennae", 
                  'unit': '', 
                  'Notation': '',
                  })
    NBINS: int = field(
        default=0, 
        metadata={'help': "Number of bins for width analysis", 
                  'unit': '', 
                  'Notation': '',
                  })
    WMETHOD: int = field(
        default=2, 
        metadata={'help': "Method of calculating FRB widths; 0 ignore, 1 std, 2 includes scattering", 
                  'unit': '', 
                  'Notation': '',
                  })
    WBIAS: str = field(
        default="Quadrature", 
        metadata={'help': "Method to calculate width bias", 
                  'unit': '', 
                  'Notation': '',
                  })
    BMETHOD: int = field(
        default=2, 
        metadata={'help': "Method for beam calculation. See beams.py:simplify_beam()", 
                  'unit': '', 
                  'Notation': '',
                  })
    BTHRESH: float = field(
        default=0.0,
        metadata={'help': 'Minimum value of beam sensitivity to consider',
                  'unit': '',
                  'Notation': 'B_{\rm min}'})
    THRESH: float = field( 
        default=1., 
        metadata={'help': "Threshold fluence used to detect an FRB", 
                  'unit': 'Jy ms', 
                  'Notation': '',
                  })
    FBAR: float = field( 
        default=1300., 
        metadata={'help': "Mean frequency",
                  'unit': 'MHz', 
                  'Notation': '',
                  })
    FRES: float = field( 
        default=1., 
        metadata={'help': "Frequency resolution",
                  'unit': 'MHz', 
                  'Notation': '',
                  })
    TRES: float = field( 
        default=1.26, 
        metadata={'help': "Time resolution",
                  'unit': 'ms', 
                  'Notation': '',
                  })
    WIDTH: float = field( 
        default=0.1, 
        metadata={'help': "Intrinsic width of the event", 
                  'unit': 'ms', 
                  'Notation': '',
                  })
@dataclass
class Observing(data_class.myDataClass):
    NORM_FRB: int = field(
        default=0, 
        metadata={'help': "Number of FRBs for TOBS", 
                  'unit': '', 
                  'Notation': '',
                  })
    TOBS: float = field(
        default=0., 
        metadata={'help': "Total observing time", 
                  'unit': 'hours', 
                  'Notation': '',
                  })

class SurveyData(data_class.myData):
    """ Hold the SurveyData in a convenient object

    """

    def set_dataclasses(self):

        self.observing = Observing()
        self.telescope = Telescope()
        # FRBs -- Will need one per FRB

        # Finish init
