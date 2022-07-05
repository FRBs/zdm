from zdm.craco import loading

fiducial_survey = "CRACO_std_May2022"


def craco_mc_survey_grid():
    """ Load the defaul MonteCarlo survey+grid for CRACO """
    survey, grid = loading.survey_and_grid(
        survey_name=fiducial_survey, NFRB=100, lum_func=2, iFRB=100
    )
    return survey, grid
