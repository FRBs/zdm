"""
This script generates MC samples

"""

from pkg_resources import resource_filename
from zdm import survey
from zdm import parameters
from zdm import misc_functions
import numpy as np

from astropy.coordinates import SkyCoord

from astropy.table import Table

import json

def modify():
    survey_names = ['MC_CRAFT_ICS_1300_1000'] #['CRAFT_ICS_892', 'CRAFT_ICS_1300', 'CRAFT_ICS_1632']

    for survey_name in survey_names:
        state = parameters.State()
        state.update_params({'min_lat': None})
        zDMgrid, zvals, dmvals = misc_functions.get_zdm_grid(
            state, datdir=resource_filename('zdm', 'GridData'))
        s = survey.load_survey(survey_name, state, dmvals)

        # Dec = s.XDec
        # RA = s.XRA

        # coords = SkyCoord(ra=RA, dec=Dec, frame='icrs', unit="deg")
        # gcoords = coords.galactic

        # Gls = gcoords.l
        # Gbs = gcoords.b

        # Do meta data for survey
        t = Table()
        t.meta = s.meta
        # t.meta['survey_data'] = json.dumps(t.meta['survey_data'])

        for key, val in s.frbs.items():
            if isinstance(val.values[0], str):
                t[key] = val.values.astype(str)
                if key == 'TNS':
                    t[key] = (val.values.astype(int) + 9000).astype(int)
            else:
                t[key] = val.values
        
        # t['Gl'] = Gls
        # t['Gb'] = Gbs
        # t['TNS'] = str(s.frbs.items()['TNS'].values.astype(int) + 1000)

        t.write(survey_name + '_mod.ecsv', format='ascii.ecsv')

modify()