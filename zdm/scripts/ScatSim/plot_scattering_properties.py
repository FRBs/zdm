""" 
This script is intended to illustrate the different
ways in which scattering can be implemented in zDM

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
    
    
    # initialise surveys and grids with different state values
    imethods = np.arange(4)
    methnames = ["_const","_width", "_scat","_zscat"]
    for i,Method in enumerate(imethods):
        
        survey_dict = {"WMETHOD": Method}
        # Write True if you want to do repeater grids - see "plot_repeaters.py" to make repeater plots
        surveys, grids = loading.surveys_and_grids(survey_names = names,\
                        repeaters=False, sdir=sdir,nz=70,ndm=140,
                        survey_dict = survey_dict)
        g=grids[0]
        s=surveys[0]
        
        llsum = it.calc_likelihoods_1D(g,s,pNreps=False)
        print("For scattering method ",methnames[i],", 1D likelihoods ",llsum)
        
        llsum = it.calc_likelihoods_2D(g,s,pNreps=False)
        print("For scattering method ",methnames[i],", 2D likelihoods are ",llsum)
        
        # extracts weights from survey and plots as function of z
        if i==3:
            weights = s.wplist
            widths = s.wlist[:,0]
            nw,nz = weights.shape
            plt.figure()
            plt.xlabel('z')
            plt.ylabel('weight')
            for iw in np.arange(nw):
                plt.plot(g.zvals,weights[iw,:],label="width = "+str(widths[iw])[0:5])
            total = np.sum(weights,axis=0)
            plt.plot(g.zvals,total,label="total",color="black")
            plt.yscale("log")
            plt.legend()
            plt.tight_layout()
            plt.savefig(opdir+"z_dependent_weights.png")
            plt.close()
            
        figures.plot_grid(
                g.rates,
                g.zvals,
                g.dmvals,
                name=opdir+"pzdm_scat"+methnames[i]+".png",
                norm=3,
                log=True,
                label='$\\log_{10} p({\\rm DM}_{\\rm cosmic} + {\\rm DM}_{\\rm host},z)$ [a.u.]',
                project=True,
                ylabel='${\\rm DM}_{\\rm cosmic} + {\\rm DM}_{\\rm host}$',
                zmax=2,DMmax=2000,
                Aconts=[0.01, 0.1, 0.5]
                )
    
    
main()
