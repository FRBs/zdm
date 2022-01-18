""" Items related to dataclasses """
import numpy as np
import json
import pandas

from zdm import io

from dataclasses import dataclass, field, asdict
# Add a few methods to be shared by them all
@dataclass
class myDataClass:
    def meta(self, attribute_name):
        return self.__dataclass_fields__[attribute_name].metadata
    def chk_options(self, attribute_name):
        options = self.__dataclass_fields__[attribute_name].metadata['options']


class myData:
    def __init__(self):
        self.set_dataclasses()

        self.set_params()

    @classmethod
    def from_dict(cls, param_dict:dict):
        slf = cls()
        # Fill em up
        slf.update_param_dict(param_dict)
        #
        return slf

    @classmethod
    def from_jsonfile(cls, jfile:str):
        json_dict = io.process_jfile(jfile)
        return cls.from_dict(json_dict)

    def set_dataclasses(self):
        pass

    def set_params(self):
        """ Generate a simple dict for parameters
        """
        # Look-up dict or convenience
        self.params = {}
        for dc_key in self.__dict__.keys():
            if dc_key == 'params':
                continue
            for param in self[dc_key].__dict__.keys():
                self.params[param] = dc_key

    def __getitem__(self, attrib:str):
        """Enables dict like access to the state

        Args:
            attrib (str): name of the attribute

        Returns:
            ?:  Value of the attribute requested
        """
        return getattr(self, attrib)

    def update_param_dict(self, params:dict):
        for key in params.keys():
            idict = params[key]
            for ikey in idict.keys():
                # Set
                self.update_param(ikey, params[key][ikey])

    def update_params(self, params:dict):
        """ Update the state parameters using the input dict

        Args:
            params (dict): New parameters+values
        """
        for key in params.keys():
            self.update_param(key, params[key])

    def update_param(self, param:str, value):
        """ Update the value of a single parameter

        Args:
            param (str): name of the parameter
            value (?): value
        """
        DC = self.params[param]
        setattr(self[DC], param, value)

    def to_dict(self):
        """ Generate a dict holding all of the object parameters

        Returns:
            dict: [description]
        """
        items = []
        for key in self.params.keys():
            items.append(self.params[key])
        uni_items = np.unique(items)
        #
        state_dict = {}
        for uni_item in uni_items:
            state_dict[uni_item] = asdict(getattr(self, uni_item))
        # Return
        return state_dict

    def write(self, outfile:str):
        """ Write the parameters to a JSON file

        Args:
            outfile (str): name of output file
        """
        state_dict = self.to_dict()
        io.savejson(outfile, state_dict, overwrite=True, easy_to_read=True)

    def vet(self, obj, dmodel:dict, verbose=True):
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

    def __repr__(self) -> str:
        return json.dumps(self.to_dict(), 
                   sort_keys=True, indent=4,
                   separators=(',', ': '))