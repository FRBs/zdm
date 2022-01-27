""" Survey data """

from dataclasses import dataclass, field, asdict

from zdm import data_class

@dataclass
class TimeFrequency(data_class.myDataClass):
    BW: float = field(
        default=0., 
        metadata={'help': "Mean Frequency (observed)", 
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
        default=0., 
        metadata={'help': "Time resolution",
                  'unit': 'ms', 
                  'Notation': '',
                  })

@dataclass
class Telescope(data_class.myDataClass):
    DIAM: float = field(
        default=0., 
        metadata={'help': "Individual antenna diameter", 
                  'unit': 'm', 
                  'Notation': '',
                  })
    NBEAMS: int = field(
        default=0, 
        metadata={'help': "Number of beams/antennae", 
                  'unit': '', 
                  'Notation': '',
                  })
    BEAM: str = field(
        default='', 
        metadata={'help': "Beam file", 
                  })
    THRESH: float = field(
        default=0., 
        metadata={'help': "Threshold fluence", 
                  'unit': 'Jy ms', 
                  'Notation': '',
                  })
    SNRTHRESH: float = field(
        default=0., 
        metadata={'help': "S/N threshold", 
                  'unit': '', 
                  'Notation': '',
                  })

@dataclass
class Observing(data_class.myDataClass):
    TOBS: float = field(
        default=0., 
        metadata={'help': "Total observing time", 
                  'unit': 'hours', 
                  'Notation': '',
                  })

class SurveyData(data_class.myData):
    """ Hold the SurveyData in a convenient
    object

    """
    def set_dataclasses(self):

        self.timefrequency = TimeFrequency()
        self.observing = Observing()
        self.telescope = Telescope()
