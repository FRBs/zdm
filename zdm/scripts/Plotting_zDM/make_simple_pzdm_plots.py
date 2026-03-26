""" 
This script is intended to illustrate how to plot common results
from a zDM grid (ignoring repetition).

It creates the following plots:
    - 2D p(z,DM) plot
    - 2D p(DMEG|z) plot
    - 2D p(DMcosmic|z) plot
    - 1D p(DM) plot
    - 1D p(z) plot
illustrates how to plot various products from it.

Please see the "zdm/scripts/PlotIndividualExperiments"
directory for the nuanced plotting of particular surveys.

"""
import os

from zdm import cosmology as cos
from zdm import figures
from zdm import parameters
from zdm import survey
from zdm import pcosmic
from zdm import iteration as it
from zdm import loading
from zdm import io
from pkg_resources import resource_filename
import numpy as np
from zdm import survey
from matplotlib import pyplot as plt

import matplotlib

defaultsize=14
ds=4
font = {'family' : 'Helvetica',
        'weight' : 'normal',
        'size'   : defaultsize}
matplotlib.rc('font', **font)

def main():

    # in case you wish to switch to another output directory
    opdir = "Plots/"
    if not os.path.exists(opdir):
        os.mkdir(opdir)
    
    # directory where the survey files are located. The below is the default - 
    # you can leave this out, or change it for a different survey file location.
    sdir = os.path.join(resource_filename('zdm', 'data'), 'Surveys')
    
    survey_name = "CRAFT_ICS_1300"
    
    # make this into a list to initialise multiple surveys art once
    names = [survey_name]
    
    # Write True if you want to do repeater grids - see "plot_repeaters.py" to make repeater plots
    surveys, grids = loading.surveys_and_grids(survey_names = names,\
                        repeaters=False, sdir=sdir,nz=700,ndm=1400)
    
    # shortcuts to the returned survey and grid objects
    s = surveys[0]
    g = grids[0]
    
    zmax=3.
    DMmax=3000
    
    ###### We not plot a standrad grid of p(z,DM) #######
    # this grid is *not* normalised in any meaningful way
    
    # determines if there are localised FRBs to plot
    if s.zlist is not None:
        OK = s.zlist
        FRBDMs=s.DMEGs[s.zlist]
        FRBZs=s.Zs[s.zlist]
    else:
        FRBDMs=None
        FRBZs=None
    
    ######### Create a plot of p(z,DM) and show FRBs ##################
    # in the below, set "project" to False to create only a single 2D plot
    figures.plot_grid(
                g.rates,
                g.zvals,
                g.dmvals,
                FRBDMs=FRBDMs,
                FRBZs=FRBZs,
                name=opdir+survey_name+"_pzdm.png",
                norm=3,
                log=True,
                label='$\\log_{10} p({\\rm DM}_{\\rm cosmic} + {\\rm DM}_{\\rm host},z)$ [a.u.]',
                project=True,
                ylabel='${\\rm DM}_{\\rm cosmic} + {\\rm DM}_{\\rm host}$',
                zmax=zmax,DMmax=DMmax,
                Aconts=[0.01, 0.1, 0.5]
                )
    
    ######### Create a plot of p(DMEG|z) and show FRBs ##################
    figures.plot_grid(
                g.smear_grid,
                g.zvals,
                g.dmvals,
                FRBDMs=FRBDMs,
                FRBZs=FRBZs,
                name=opdir+survey_name+"_pdmeg_given_z.png",
                norm=3,
                log=True,
                label='$\\log_{10} p({\\rm DM}_{\\rm cosmic} + {\\rm DM}_{\\rm host}|z)$ [a.u.]',
                project=False,
                ylabel='${\\rm DM}_{\\rm cosmic} + {\\rm DM}_{\\rm host}$',
                zmax=zmax,DMmax=DMmax,
                conts=[0.16,0.5,0.84]
                )
    
    
    ######### Create a plot of p(DMcosmic|z) and show FRBs ##################
    figures.plot_grid(
                g.grid,
                g.zvals,
                g.dmvals,
                FRBDMs=FRBDMs,
                FRBZs=FRBZs,
                name=opdir+survey_name+"_pdmcosmic_given_z.png",
                norm=4,
                log=True,
                label='$\\log_{10} p({\\rm DM}_{\\rm cosmic}|z)$ [a.u.]',
                project=False,
                ylabel='${\\rm DM}_{\\rm cosmic}$',
                zmax=zmax,DMmax=DMmax,
                conts=[0.16,0.5,0.84]
                )
    
    
    ######### Create a 1D p(z) plot ##########
    # there is no fancy pre-existing routine for this one
    pz = np.sum(g.rates,axis=1)
    pz /= np.max(pz) # just to give it a reasonable normalisation
    plt.ylim(0,1)
    plt.xlim(0,zmax) # limits to this redshift range
    plt.plot(g.zvals,pz)
    plt.xlabel("Redshift, z")
    plt.ylabel("p(z) [a.u.]")
    plt.tight_layout()
    plt.savefig(opdir+survey_name+"_pz.png")
    plt.close()
    
    
    ######### Create a 1D p(z) plot ##########
    # there is no fancy pre-existing routine for this one
    pdm = np.sum(g.rates,axis=0)
    pdm /= np.max(pdm) # just to give it a reasonable normalisation
    plt.ylim(0,1)
    plt.xlim(0,DMmax) # limits to this redshift range
    plt.plot(g.dmvals,pdm)
    plt.xlabel("DM [pc cm$^{-3}$]")
    plt.ylabel("p(DM) [a.u.]")
    plt.tight_layout()
    plt.savefig(opdir+survey_name+"_pDM.png")
    plt.close()
    
main()
