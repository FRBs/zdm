"""  Refactor the original Survey files into ECSV files 
"""
import os
from pkg_resources import resource_filename

from zdm import survey 

#
if __name__ == "__main__":

    sdir = os.path.join(resource_filename('zdm', 'data'), 
                        'Surveys', 'Original')

    # CRAFT FE
    survey.refactor_old_survey_file(
        'CRAFT/FE', 'CRAFT_class_I_and_II.ecsv', 
        clobber=False, sdir=sdir)