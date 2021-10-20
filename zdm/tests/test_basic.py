
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

    # Compare to main branch
    DEF_Omega_k=0.
    # dark energy / cosmological constant (in current epoch)
    DEF_Omega_lambda=0.685
    # matter density in current epoch
    DEF_Omega_m=0.315 #Plank says 0.315
    # baryon density
    DEF_Omega_b=0.044
    DEF_Omega_b_h2=0.0224 #Planck says 0.0224, WMAP 0.02264
    # hubble constant in current epoch
    DEF_H0 = igm.Planck15.H0.value #km s^-1 Mpc^-1 #planck15 used in frb.igm



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
    

    Location='Plots'
    if not os.path.isdir(Location):
        os.mkdir(Location)
    prefix='bestfit_'
    
    do2DPlots=True
    if do2DPlots:
        # Unpack for convenience
        lat50,ICS,ICS892,pks = surveys
        muDM=10**params['host'].lmean
        Macquart=muDM
        # plots zdm distribution
        misc_functions.plot_grid_2(gpks.rates,gpks.zvals,gpks.dmvals,zmax=3,DMmax=3000,
                             name=os.path.join(Location,prefix+'nop_pks_optimised_grid.pdf'),
                             norm=2,log=True,label='$\\log_{10} p({\\rm DM}_{\\rm EG},z)$',
                             project=False,FRBDM=pks.DMEGs,FRBZ=None,Aconts=[0.01,0.1,0.5],Macquart=Macquart)
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
        