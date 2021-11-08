
import pytest

from pkg_resources import resource_filename
import os
import copy
import pickle

from astropy.cosmology import Planck18

from zdm import cosmology as cos
from zdm import misc_functions
from zdm import parameters
from zdm import survey

from IPython import embed

def make_grids():

    ############## Initialise parameters ##############
    state = parameters.State()

    # Variable parameters
    vparams = {}
    vparams['cosmo'] = {}
    vparams['cosmo']['H0'] = 67.74
    vparams['cosmo']['Omega_lambda'] = 0.685
    vparams['cosmo']['Omega_m'] = 0.315
    vparams['cosmo']['Omega_b'] = 0.044

    # Update state
    state.update_param_dict(vparams)

    ############## Initialise cosmology ##############
    cos.set_cosmology(state)
    cos.init_dist_measures()

    # get the grid of p(DM|z). See function for default values.
    # set new to False once this is already initialised
    zDMgrid, zvals,dmvals = misc_functions.get_zdm_grid(
        state, new=True, plot=False, method='analytic')

    surveys = []
    names = ['CRAFT/FE', 'CRAFT/ICS', 'CRAFT/ICS892', 'PKS/Mb']
    for survey_name in names:
        surveys.append(survey.load_survey(survey_name, state, dmvals))

    # generates zdm grids for the specified parameter set
    if state.beam.method =='Full':
        gprefix='best'
    elif state.beam.method =='Std':
        gprefix='Std_best'
    
    if state.analysis.NewGrids:
        print("Generating new grids, set NewGrids=False to save time later")
        grids=misc_functions.initialise_grids(
            surveys,zDMgrid, zvals, dmvals, state, wdist=True)#, source_evolution=source_evolution, alpha_method=alpha_method)
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

    Location='Plots'
    if not os.path.isdir(Location):
        os.mkdir(Location)
    prefix='bestfit_'
    
    do2DPlots=True
    if do2DPlots:
        # Unpack for convenience
        lat50,ICS,ICS892,pks = surveys
        #muDM=10**state.host.lmean
        Macquart=grids[0].state
        # plots zdm distribution
        misc_functions.plot_grid_2(gpks.rates,gpks.zvals,gpks.dmvals,zmax=3,DMmax=3000,
                             name=os.path.join(Location,prefix+'nop_pks_optimised_grid.pdf'),
                             norm=2,log=True,label='$\\log_{10} p({\\rm DM}_{\\rm EG},z)$',
                             project=False,FRBDM=pks.DMEGs,FRBZ=None,Aconts=[0.01,0.1,0.5],
                             Macquart=Macquart)
        misc_functions.plot_grid_2(gICS.rates,gICS.zvals,gICS.dmvals,zmax=1,DMmax=2000,
                             name=os.path.join(Location,prefix+'nop_ICS_optimised_grid.pdf'),
                             norm=2,log=True,label='$\\log_{10} p({\\rm DM}_{\\rm EG},z)$',
                             project=False,FRBDM=ICS.DMEGs,FRBZ=ICS.frbs["Z"],Aconts=[0.01,0.1,0.5],Macquart=Macquart)
        misc_functions.plot_grid_2(gICS892.rates,gICS892.zvals,gICS892.dmvals,zmax=1,DMmax=2000,
                             name=os.path.join(Location,prefix+'nop_ICS892_optimised_grid.pdf'),
                             norm=2,log=True,label='$\\log_{10} p({\\rm DM}_{\\rm EG},z)$',
                             project=False,FRBDM=ICS892.DMEGs,FRBZ=ICS892.frbs["Z"],Aconts=[0.01,0.1,0.5],Macquart=Macquart)
        
        misc_functions.plot_grid_2(glat50.rates,glat50.zvals,glat50.dmvals,zmax=0.6,DMmax=1500,
                             name=os.path.join(Location,prefix+'nop_lat50_optimised_grid.pdf'),
                             norm=2,log=True,label='$\\log_{10} p({\\rm DM}_{\\rm EG},z)$',
                             project=False,FRBDM=lat50.DMEGs,FRBZ=None,Aconts=[0.01,0.1,0.5],Macquart=Macquart)
        
        # plots zdm distribution, including projections onto z and DM axes
        misc_functions.plot_grid_2(gpks.rates,gpks.zvals,gpks.dmvals,zmax=3,DMmax=3000,
                             name=os.path.join(Location,prefix+'pks_optimised_grid.pdf'),
                             norm=2,log=True,label='$\\log_{10} p({\\rm DM}_{\\rm EG},z)$',
                             project=True,FRBDM=pks.DMEGs,FRBZ=None,Aconts=[0.01,0.1,0.5],Macquart=Macquart)
        misc_functions.plot_grid_2(gICS.rates,gICS.zvals,gICS.dmvals,zmax=1,DMmax=2000,
                             name=os.path.join(Location,prefix+'ICS_optimised_grid.pdf'),
                             norm=2,log=True,label='$\\log_{10} p({\\rm DM}_{\\rm EG},z)$',
                             project=True,FRBDM=ICS.DMEGs,FRBZ=ICS.frbs["Z"],Aconts=[0.01,0.1,0.5],Macquart=Macquart)
        misc_functions.plot_grid_2(gICS892.rates,gICS892.zvals,gICS892.dmvals,zmax=1,DMmax=2000,
                             name=os.path.join(Location,prefix+'ICS892_optimised_grid.pdf'),
                             norm=2,log=True,label='$\\log_{10} p({\\rm DM}_{\\rm EG},z)$',
                             project=True,FRBDM=ICS892.DMEGs,FRBZ=ICS892.frbs["Z"],Aconts=[0.01,0.1,0.5],Macquart=Macquart)
        misc_functions.plot_grid_2(glat50.rates,glat50.zvals,glat50.dmvals,zmax=0.5,DMmax=1000,
                             name=os.path.join(Location,prefix+'lat50_optimised_grid.pdf'),
                             norm=2,log=True,label='$\\log_{10} p({\\rm DM}_{\\rm EG},z)$',
                             project=True,FRBDM=lat50.DMEGs,FRBZ=None,Aconts=[0.01,0.1,0.5],Macquart=Macquart)

make_grids()        