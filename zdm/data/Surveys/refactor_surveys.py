"""  Refactor the original Survey files into ECSV files 
"""
import os
from pkg_resources import resource_filename

from zdm import survey 

#
if __name__ == "__main__":

    # PKS/Mb
    survey.refactor_old_survey_file(
        'PKS/Mb', 'parkes_mb_class_I_and_II.ecsv',
        clobber=False)

    # CRAFT ICS
    survey.refactor_old_survey_file(
        'CRAFT/ICS', 'CRAFT_ICS.ecsv', 
        clobber=False)

    # CRAFT FE
    survey.refactor_old_survey_file(
        'CRAFT/FE', 'CRAFT_class_I_and_II.ecsv', 
        clobber=False)

    # CRAFT 892
    survey.refactor_old_survey_file(
        'CRAFT/ICS892', 'CRAFT_ICS_892.ecsv', 
        clobber=False)

    # CRAFT 1632
    survey.refactor_old_survey_file(
        'CRAFT/ICS1632', 'CRAFT_ICS_1632.ecsv', 
        clobber=False)