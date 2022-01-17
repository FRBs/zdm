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

class SurveyData:
    """ Initialize the full state for the analysis 
    with the default parameters

    Returns:
        dict: [description]
    """
    def __init__(self):

        self.frequency = Frequency()

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
