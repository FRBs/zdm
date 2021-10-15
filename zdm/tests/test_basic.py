
import pytest

from pkg_resources import resource_filename
import os
import pickle

from zdm import cosmology as cos
from zdm import misc_functions
from zdm import parameters
from zdm import survey

from IPython import embed

def test_make_grids():

    ############## Initialise parameters ##############
    params = parameters.init_parameters()
    params['cosmo'].current_H0 = 70.


    ############## Initialise cosmology ##############
    cos.set_cosmology(params)
    cos.init_dist_measures()

    # get the grid of p(DM|z). See function for default values.
    # set new to False once this is already initialised
    zDMgrid, zvals,dmvals, _ = misc_functions.get_zdm_grid(
        params['cosmo'].current_H0,
        new=True, plot=False, method='analytic')


    sdir = os.path.join(resource_filename('zdm', 'data'), 'Surveys')

    surveys = []
    for ss, dfile in zip(range(4),
                         ['CRAFT_class_I_and_II.dat',
                         'CRAFT_ICS.dat',
                         'CRAFT_ICS_892.dat', 
                         'parkes_mb_class_I_and_II.dat']): 

        srvy=survey.survey()
        srvy.process_survey_file(os.path.join(sdir, dfile))
        srvy.init_DMEG(params['MW'].DMhalo)
        srvy.init_beam(nbins=params['beam'].Nbeams[ss],
                    method=2, plot=False,
                    thresh=params['beam'].thresh) # tells the survey to use the beam file
        pwidths,pprobs=survey.make_widths(srvy, 
                                      params['width'].logmean,
                                      params['width'].logsigma,
                                      params['beam'].Wbins,
                                      scale=params['beam'].Wscale)
        _ = srvy.get_efficiency_from_wlist(dmvals,pwidths,pprobs)

        # Append
        surveys.append(srvy)

    
    # generates zdm grids for the specified parameter set
    if params['beam'].method =='Full':
        gprefix='best'
    elif params['beam'].method =='Std':
        gprefix='Std_best'
    
    if params['analysis'].NewGrids:
        print("Generating new grids, set NewGrids=False to save time later")
        grids=misc_functions.initialise_grids(
            surveys,zDMgrid, zvals, dmvals, params,
            wdist=True)#, source_evolution=source_evolution, alpha_method=alpha_method)
        with open('Pickle/'+gprefix+'grids.pkl', 'wb') as output:
            pickle.dump(grids, output, pickle.HIGHEST_PROTOCOL)
    else:
        print("Loading grid ",'Pickle/'+gprefix+'grids.pkl')
        with open('Pickle/'+gprefix+'grids.pkl', 'rb') as infile:
            grids=pickle.load(infile)
    glat50=grids[0]
    gICS=grids[1]
    gICS892=grids[2]
    gpks=grids[3]
    print("Initialised grids")
    
    