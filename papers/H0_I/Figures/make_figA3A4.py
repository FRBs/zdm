"""
Generates figure A3 and A4

If "extraPlots" is set to true, this will produce a LARGE
number of extra figures for your viewing pleasure!

"""

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
from zdm import iteration as it

from IPython import embed

from pathlib import Path as dirpath
from matplotlib import pyplot as plt
import numpy as np

import matplotlib

defaultsize=14
ds=4
font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : defaultsize}
matplotlib.rc('font', **font)

def make_grids(extraPlots=False):
    
    ##### working sub-directory for this analysis #####
    directory = "./FigureA3_A4/"
    dirpath(directory).mkdir(parents=True, exist_ok=True)
    
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
    
    lat50=surveys[0]
    ICS=surveys[1]
    ICS892=surveys[2]
    pks=surveys[3]
    
    # generates zdm grids for the specified parameter set
    if state.beam.Bmethod == 3: #'Full':
        gprefix='best'
    elif state.beam.Bmethod == 2: #'Std':
        gprefix='Std_best'
    else:
        raise ValueError("Bad beam method!")
    
    ## check directory exists ##
    dirpath("./Pickle").mkdir(parents=True, exist_ok=True)
    
    ############## loop through Galactic DM contributions ###########
    
    # initiualise a plot of mean efficiencies
    plt.figure()
    plt.xlabel("DM$_{\\rm EG}$")
    plt.ylabel("Efficiency, $\\bar{\\epsilon} = \\sum_i \\epsilon(w_i) p(w_i)$")
    # make a list of plotting styles (one for each survey)
    style=["-","--","-.",":"] # four surveys
    colours=["red","green","blue","orange","purple","cyan"]
    
    # plots initial efficiency
    for j,s in enumerate(surveys):
        efficiencies = s.efficiencies
        weighted = (efficiencies.T).dot(s.wplist)
        plt.plot(dmvals,weighted,linestyle=style[j],color="black",label=names[j])
    
    # list of trial Galacic DM contributions
    DMGs=[0,100,200,300,400,500]
    
    if extraPlots:
        for i,DMG in enumerate(DMGs):
            # set DMG artificially in surveys
            for j,s in enumerate(surveys):
                # sets Galactic contributions to specific values
                s.frbs["DMG"][:]=DMG
                # sets survey "mean efficiency
                
                # survey already has width list and weights initialised
                # just need to re-calculate relative efficiencies
                efficiencies = s.get_efficiency_from_wlist(dmvals,s.wlist,s.wplist)
                weighted = (efficiencies.T).dot(s.wplist)
                plt.plot(dmvals,weighted,linestyle=style[j],color=colours[i])#,label=names[j]+", DMG= "+str(DMG))
        plt.legend(loc="upper right")
        plt.xlim(0,2000)
        plt.tight_layout()
        plt.savefig(directory+"efficiencies.pdf")
        plt.close()
    
    # initialises arrays to hold DM, z, and total info
    nDMGs = len(DMGs)
    nSurveys = len(surveys)
    nz = zvals.size
    ndm = dmvals.size
    totals = np.zeros([nDMGs,nSurveys])
    dmprojs = np.zeros([nDMGs,nSurveys,ndm])
    zprojs = np.zeros([nDMGs,nSurveys,nz])
    
    # test if we need to perform new calculations, or just plot it
    if (os.path.exists(directory+"totals.npy") and os.path.exists(directory+"z_proj.npy")
        and os.path.exists(directory+"dm_proj.npy")):
        new_DMG_calcs = False
    else:
        new_DMG_calcs = True
    
    
    for i,DMG in enumerate(DMGs):
        if not new_DMG_calcs:
            break
        # set DMG artificially in surveys
        for j,s in enumerate(surveys):
            # sets Galactic contributions to specific values
            s.frbs["DMG"][:]=DMG
            # sets survey "mean efficiency
            
            # survey already has width list and weights initialised
            efficiencies = s.get_efficiency_from_wlist(dmvals,s.wlist,s.wplist)
            
        grids=misc_functions.initialise_grids(
            surveys,zDMgrid, zvals, dmvals, state, wdist=True)
        # gets total number, z-projections, and DM-projectstions
        
        for j,g in enumerate(grids):
            rates = g.rates
            dm_proj = np.sum(rates,axis=0)
            z_proj = np.sum(rates,axis=1)
            tot = np.sum(dm_proj)
            totals[i,j] = tot
            dmprojs[i,j,:] = dm_proj
            zprojs[i,j,:] = z_proj
        glat50=grids[0]
        gICS=grids[1]
        gICS892=grids[2]
        gpks=grids[3]
        # plots zdm distribution, including projections onto z and DM axes
        prefix="DMG"+str(DMG)+"_"
        Location=directory
        Macquart=grids[0].state
        if extraPlots:
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
        
    if new_DMG_calcs:
        np.save(directory+"totals.npy",totals)
        np.save(directory+"z_proj.npy",zprojs)
        np.save(directory+"dm_proj.npy",dmprojs)
    else:
        totals = np.load(directory+"totals.npy")
        zprojs = np.load(directory+"z_proj.npy")
        dmprojs = np.load(directory+"dm_proj.npy")
    
    ##### effects on total rate #######
    
    styles=["-","--",":","-."]
    
    names=["CRAFT/FE","CRAFT/ICS (1.3 GHz)","CRAFT/ICS (900 MHz)","Parkes/Mb"]
    
    plt.figure()
    plt.xlabel('DM$_{\\rm ISM}$ [pc cm$^{-3}$]')
    plt.ylabel('Relative detection rate')
    plt.ylim(0,1)
    plt.xlim(0,500)
    for i in np.arange(nSurveys):
        plt.plot(DMGs,totals[:,i]/totals[0,i],label=names[i],linestyle=styles[i],linewidth=3)
    
    plt.legend()
    plt.tight_layout()
    plt.savefig(directory+"FigureA3.pdf")
    plt.close()
    
    if extraPlots:
        ##### effects on DM distribution #######
        plt.figure()
        plt.xlabel('DM$_{\\rm EG}$')
        plt.ylabel('p(DM$_{\\rm EG}$)')
        plt.xlim(0,2000)
        for i in np.arange(nSurveys):
            for j in np.arange(nDMGs):
                norm=np.sum(dmprojs[j,i,:])*(dmvals[1]-dmvals[0])
                plt.plot(dmvals,dmprojs[j,i,:]/norm,color=colours[j],linestyle=style[i])
         
        #plt.legend()
        plt.tight_layout()
        plt.savefig(directory+"effect_on_dm_distributions.pdf")
        plt.close()
    
    ##### effects on z distribution #######
    zmeans = np.zeros(totals.shape)
    xmaxs=[0.5,1,1,2]
    savenames = ['CRAFT_FE', 'CRAFT_ICS', 'CRAFT_ICS892', 'PKS_Mb']
    for i in np.arange(nSurveys):
        plt.figure()
        plt.xlabel('$z$')
        plt.ylabel('$p(z) (a.u.)$')
        plt.xlim(0,xmaxs[i])
        for j in np.arange(nDMGs):
            #norm=np.sum(zprojs[j,i])*(zvals[1]-zvals[0])
            plt.plot(zvals,zprojs[j,i,:],color=colours[j],linestyle=style[i],label=str(DMGs[j]))
            zmean=np.sum(zprojs[j,i,:]*zvals)/np.sum(zprojs[j,i,:])
            zmeans[j,i]=zmean
        plt.legend()
        plt.tight_layout()
        if extraPlots:
            plt.savefig(directory+savenames[i]+"effect_on_z_distributions.pdf")
        plt.close()
    
    
    ### plots mean redshift ###
    plt.figure()
    plt.xlabel('${\\rm DM_{ISM}}$ [pc cm$^{-3}$]')
    plt.ylabel('$\\dfrac{\\bar{z}({\\rm DM_{ISM}})}{\\bar{z}({\\rm DM_{ISM}}=0)}$')
    plt.xlim(0,500)
    plt.ylim(0,1)
    for i in np.arange(nSurveys):
        plt.plot(DMGs,zmeans[:,i]/zmeans[0,i],label=names[i],linestyle=styles[i],linewidth=3)
    plt.legend()
    plt.tight_layout()
    
    if extraPlots:
        plt.savefig(directory+"mean_z.pdf")
    
    
    ### plots zoomed version ###
    plt.ylim(0.8,1)
    plt.xlim(0,500)
    plt.yticks([0.8,0.85,0.9,0.95,1.0])
    #plt.tight_layout()
    plt.savefig(directory+"FigureA4.pdf")
    plt.close()
    
    
        
make_grids()        
