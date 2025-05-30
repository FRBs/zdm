""" Survey data """

from dataclasses import dataclass, field

from zdm import data_class
import numpy as np

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
        default=None, 
        metadata={'help': "Galactic latitude",
                  'unit': 'deg', 
                  'Notation': '',
                  })
    Gl: float = field( 
        default=None, 
        metadata={'help': "Galactic longitude",
                  'unit': 'deg', 
                  'Notation': '',
                  })
    RA: float = field( 
        default=None, 
        metadata={'help': "Right ascension in J2000 coordinates",
                  'unit': 'deg', 
                  'Notation': '',
                  })
    DEC: float = field( 
        default=None, 
        metadata={'help': "Declination in J2000 coordinates",
                  'unit': 'deg', 
                  'Notation': '',
                  })
    NREP: np.int64 = field( 
         default=1, 
         metadata={'help': "Number of repetitions detected", 
                   'unit': '', 
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
        metadata={'help': "Width of the event", 
                  'unit': 'ms', 
                  'Notation': '',
                  })
    TAU: float = field( 
        default=-1., 
        metadata={'help': "Scattering timescale of the event", 
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
        metadata={'help': "Code for width method. 0: ignore it (all 1ms), 1: intrinsic lognormal, 2: include scattering, 3: scat & z-dependence, 4: specific FRB", 
                  'unit': '', 
                  'Notation': '',
                  })
    WDATA: int = field(
        default=2, 
        metadata={'help': "What does the WIDTH column include? 0 intrinsic, 1: also scattering, 2: also DM smearing",
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
    DRIFT_SCAN: int = field(
        default=1,
        metadata={'help': '1: beam represents solid angle viewed at each value of b, for time Tfield \
                           2: (Drift scan) beam represents time (in days) spent on any \
                               given source at sensitivity level b. \
                              Tfield is solid angle. Nfields then becomes a multiplier of the time.',
                  'unit': '',
                  'Notation': ''
                  })
    BTHRESH: float = field(
        default=1.e-3,
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
    NORM_REPS: int = field(
        default=None, 
        metadata={'help': "Number of repeaters for TOBS", 
                  'unit': '', 
                  'Notation': '',
                  })
    NORM_SINGLES: int = field(
        default=None, 
        metadata={'help': "Number of singles for TOBS", 
                  'unit': '', 
                  'Notation': '',
                  })
    TOBS: float = field(
        default=None, 
        metadata={'help': "Total observing time", 
                  'unit': 'days', 
                  'Notation': '',
                  })
    TFIELD: float = field(
        default=None, 
        metadata={'help': "Observing time per field", 
                  'unit': 'days', 
                  'Notation': '',
                  })
    NFIELDS: int = field(
        default=None,
        metadata={'help': "Number of observing fields",
                  'unit': '',
                  'Notation': '',
                  })
    MAX_DM: float = field(
        default=None, 
        metadata={'help': "Maximum searched DM", 
                  'unit': 'pc/cm**3', 
                  'Notation': '',
                  })
    MAX_IDT: int = field(
        default=None,
        metadata={'help': "Maximum number of time samples seaarched (4096 for CRAFT ICS)",
                  'unit': '',
                  'Notation': '',
                  })
    MAX_IW: int = field(
        default=None,
        metadata={'help': "Maximum width of FRB search in units of tres (12 for CRAFT ICS)",
                  'unit': '',
                  'Notation': '',
                  })
    MAXWMETH: int = field(
        default=0,
        metadata={'help': "Method for treating FRBs with width > max width. 0: do nothing, 1: ignore them, 2: reduce sensitivity to 1/w",
                  'unit': '',
                  'Notation': '',
                  })
    MAX_LOC_DMEG: int = field(
        default=-1,
        metadata={'help': "Ignore zs with DMEG larger than 'x'. \n-1: Use all zs \n0: 'x' = smallest DMEG for an FRB without a z \n>0: 'x' = this value",
                  'unit': 'pc/cm**3',
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
