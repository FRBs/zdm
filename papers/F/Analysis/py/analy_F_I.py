from zdm.craco import loading

fiducial_survey = "../MC_F/Surveys/F_0.32_survey"


def craco_mc_survey_grid(iFRB=100):
    """ Load the defaul MonteCarlo survey+grid for CRACO """
    survey, grid = loading.survey_and_grid(
        survey_name=fiducial_survey, NFRB=100, lum_func=2, iFRB=iFRB
    )
    return survey, grid
