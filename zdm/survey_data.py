""" Survey data """

from dataclasses import dataclass, field, asdict

from zdm import data_class

@dataclass
class Frequency(data_class.myDataClass):
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
    """ Hold the SurveyData in a convenient
    object

    """
    def set_dataclasses(self):

        self.frequency = Frequency()
