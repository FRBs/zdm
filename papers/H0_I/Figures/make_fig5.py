""" 
This script generates figure 5 from the H0 paper (James, Ghosh, Prochaska et al)

It also generates imilar figures for each of the three frequency ranges.

"""
import os

from zdm import cosmology as cos
from zdm import misc_functions
from zdm import parameters
from zdm import survey
from zdm import pcosmic
from zdm import iteration as it
from zdm.MC_sample import loading
from zdm import io

import numpy as np
from zdm import survey
from matplotlib import pyplot as plt

def main():
    
    # in case you wish to switch to another output directory
    opdir='Figure5/'
    if not os.path.exists(opdir):
        os.mkdir(opdir)
    
    # Initialise surveys and grids
    
    # The below is for private, unpublished FRBs. You will NOT see this in the repository!
    names = ['CRAFT_ICS','CRAFT_ICS_892','CRAFT_ICS_1632']
    sdir='zdm/data/Surveys/'
    
    # if True, this generates a summed histogram of all the surveys, weighted by
    # the observation time
    sumit=True
    
    # approximate best-fit values from recent analysis
    vparams = {}
    vparams['H0'] = 73
    vparams['lEmax'] = 41.3
    vparams['gamma'] = -0.9
    vparams['alpha'] = 1
    vparams['sfr_n'] = 1.15
    vparams['lmean'] = 2.25
    vparams['lsigma'] = 0.55
    
    zvals=[]
    dmvals=[]
    grids=[]
    surveys=[]
    nozlist=[]
    for i,name in enumerate(names):
        s,g = loading.survey_and_grid(
            survey_name=name,NFRB=None,sdir=sdir) # should be equal to actual number of FRBs, but for this purpose it doesn't matter
        grids.append(g)
        surveys.append(s)
        
        # set up new parameters
        g.update(vparams)
        
        # gets cumulative rate distribution
        if i==0:
            rtot = np.copy(g.rates)*s.TOBS
        else:
            rtot += g.rates*s.TOBS
        
        if name=='Arecibo':
            # remove high DM vals from rates as per ALFA survey limit
            delete=np.where(g.dmvals > 2038)[0]
            g.rates[:,delete]=0.
        
        for iFRB in s.zlist:
            zvals.append(s.Zs[iFRB])
            dmvals.append(s.DMEGs[iFRB])
            for dm in s.DMEGs[s.nozlist]:
                nozlist.append(dm)
        
        ############# do 2D plots ##########
        misc_functions.plot_grid_2(g.rates,g.zvals,g.dmvals,
            name=opdir+name+'.pdf',norm=3,log=True,label='$\\log_{10} p({\\rm DM}_{\\rm EG},z)$  [a.u.]',
            project=False,FRBDM=s.DMEGs,FRBZ=s.frbs["Z"],Aconts=[0.01,0.1,0.5],zmax=1.5,
            DMmax=1500)#,DMlines=s.DMEGs[s.nozlist])
        
    if sumit:
        # does the final plot of all data
        frbzvals=np.array(zvals)
        frbdmvals=np.array(dmvals)
        ############# do 2D plots ##########
        misc_functions.plot_grid_2(g.rates,g.zvals,g.dmvals,
            name=opdir+'Fig5_combined_localised_FRBs.pdf',norm=3,log=True,
            label='$\\log_{10} p({\\rm DM}_{\\rm EG},z)$ [a.u.]',
            project=False,FRBDM=frbdmvals,FRBZ=frbzvals,Aconts=[0.01,0.1,0.5],
            zmax=1.5,DMmax=2000,DMlines=nozlist)
    
    
main()
