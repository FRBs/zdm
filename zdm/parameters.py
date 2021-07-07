import numpy as np
from dataclasses import dataclass, field, fields

# Add a few methods to be shared by them all
@dataclass
class myDataClass:
    def meta(self, attribute_name):
        return self.__dataclass_fields__[attribute_name].metadata
    def chk_options(self, attribute_name):
        options = self.__dataclass_fields__[attribute_name].metadata['options']

# Beam parameters
@dataclass
class BeamParams(myDataClass):
    thresh: int = field(metadata={'help': '??'})
    method: int


# FRB intrinsic width parameters
@dataclass
class WidthParams(myDataClass):
    logmean: float = field(metadata={'help': 'Intrinsic width log of mean'})
    logsigma: float = field(metadata={'help': 'Intrinsic width log of sigma'})


# Milky Way parameters
MW_dmodel = {
    'DMhalo': dict(dtype=(float),
                help='DM of Milky Way halo in units of pc/cm^3'),
}

# FRB demongraphics
frbdemo_dmodel = {
    'source_evolution': 
        dict(dtype=(int), 
             options=[0,1], 
             help='Integer flag specifying the function used.  '+\
                 '0: SFR^n; 1: (1+z)^(2.7n)'), 
    'alpha_method': 
        dict(dtype=(int), 
             options=[0,1], 
             help='Integer flag specifying the nature of scaling. '+\
                 '0: spectral index interpretation: includes k-correction. Slower to update ' +\
                 '1: rate interpretation: extra factor of (1+z)^alpha in source evolution'),
        }

# Cosmology

# Analysis
analysis_dmodel = {
    'sprefix': 
        dict(dtype=(str), 
             options=['Std', 'Full'],
             help='?? '+\
                 'Full:  more detailed estimates. Takes more space and time \n'+\
                 'Std: faster - fine for max likelihood calculations, not as pretty'),
    'NewGrids': dict(dtype=bool,
                help='Generate new z, DM grids?'),
}


def init_parameters():

    # Begin
    param_dict = {}

    # FRB demographics
    FRBdemo_dict = dict(source_evolution=0, 
                        alpha_method=1)
    vet_param(FRBdemo_dict, frbdemo_dmodel)

    # Beam
    beam_dict = dict(thresh=0,
                     method=2)
    vet_param(beam_dict, beam_dmodel)

    # Width
    width_dict = dict(logmean=1.70267, 
                      logsigma=0.899148)
    vet_param(width_dict, width_dmodel)

    # Milky Way
    MW_dict = dict(DMhalo=50.)
    vet_param(MW_dict, MW_dmodel)

    # Analysis
    analysis_dict = dict(NewGrids=True, 
                         sprefix='Std')
    vet_param(analysis_dict, analysis_dmodel)
    
    # Wrap em together
    param_dict['width'] = width_dict
    param_dict['MW'] = MW_dict
    param_dict['analysis'] = analysis_dict
    param_dict['beam'] = beam_dict

    return param_dict


def vet_param(obj, dmodel:dict, verbose=True):
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

def vet_param():
    pass