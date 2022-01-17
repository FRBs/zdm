""" Survey data """

from dataclasses import dataclass, field, asdict

from zdm import data_class

@dataclass
class Frequency(data_class.myDataClass):
    FBAR: float = field(
        default=0.,
        metadata={'help': "Mean Frequency (observed)", 
                  'unit': 'MHz', 
                  'Notation': '',
                  })
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

class SurveyData(data_class.myData):
    """ Initialize the full state for the analysis 
    with the default parameters

    Returns:
        dict: [description]
    """
    def set_dataclasses(self):

        self.frequency = Frequency()
