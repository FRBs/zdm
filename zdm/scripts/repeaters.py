""" 
This script shows how to use repeating FRB grids.

It produces four outputs in the "Repeaters" directory,
showing zDM for:
- 1: All bursts (single bursts, and bursts from repeating sources)
- 2: FRBs expected as single bursts
- 3: Repeating FRBs (each source counts once)
- 4: Bursts from repeaters (each source counts Nburst times)

We expect 1 = 2+4 (if not, it's a bug!)

"""
import os
from pkg_resources import resource_filename
from zdm import misc_functions
from zdm import parameters
from zdm import iteration as it
from zdm import loading as loading
import numpy as np

def main():
    
    # in case you wish to switch to another output directory
    opdir='Repeaters/'
    if not os.path.exists(opdir):
        os.mkdir(opdir)
    
    # standard 1.4 GHz CRAFT data
    # name = 'CRAFT_ICS'
        
    name = 'CHIME_decbin_3_of_6'
    
    state = parameters.State()
    
    sdir='../data/Surveys/CHIME/'
    # use loading.survey_and_grid for proper estimates
    # remove loading for width-based estimates
    # the below is hard-coded for a *very* simplified analysis!
    # using loading. gives 5 beams and widths, ignoring that gives a single beam
    ss,gs = loading.surveys_and_grids(survey_names=[name],repeaters=True,sdir=sdir)
    s = ss[0]
    g = gs[0]

    # updates survey to have single beam value and weights
    
    # set up new parameters
    # vparams = {}
    # vparams['H0'] = 73
    # vparams['lEmax'] = 41.3
    # vparams['gamma'] = -0.95
    # vparams['alpha'] = 1.03
    # vparams['sfr_n'] = 1.15
    # vparams['lmean'] = 2.23
    # vparams['lsigma'] = 0.57
    # vparams['lC'] = 1.963

    # g.update(vparams)
    
    newC,llC=it.minimise_const_only(None,[g],[s])
    
    ############# do 2D plots ##########
    misc_functions.plot_grid_2(g.rates,g.zvals,g.dmvals,
        name=opdir+name+'all_frbs.pdf',norm=3,log=True,label='$\\log_{10} p({\\rm DM}_{\\rm EG},z)$  [a.u.]',
        project=False,FRBDM=s.DMEGs,FRBZ=s.frbs["Z"],Aconts=[0.01,0.1,0.5],zmax=1.5,
        DMmax=1500)
    
    misc_functions.plot_grid_2(g.exact_singles,g.zvals,g.dmvals,
        name=opdir+name+'single_frbs.pdf',norm=3,log=True,label='$\\log_{10} p({\\rm DM}_{\\rm EG},z)$  [a.u.]',
        project=False,FRBDM=s.DMEGs,FRBZ=s.frbs["Z"],Aconts=[0.01,0.1,0.5],zmax=1.5,
        DMmax=1500)
    
    misc_functions.plot_grid_2(g.exact_reps,g.zvals,g.dmvals,
        name=opdir+name+'repeating_sources.pdf',norm=3,log=True,label='$\\log_{10} p({\\rm DM}_{\\rm EG},z)$  [a.u.]',
        project=False,FRBDM=s.DMEGs,FRBZ=s.frbs["Z"],Aconts=[0.01,0.1,0.5],zmax=1.5,
        DMmax=1500)
    
    misc_functions.plot_grid_2(g.exact_rep_bursts,g.zvals,g.dmvals,
        name=opdir+name+'bursts_from_repeaters.pdf',norm=3,log=True,label='$\\log_{10} p({\\rm DM}_{\\rm EG},z)$  [a.u.]',
        project=False,FRBDM=s.DMEGs,FRBZ=s.frbs["Z"],Aconts=[0.01,0.1,0.5],zmax=1.5,
        DMmax=1500)
    
    print(np.max(g.rates - g.exact_singles - g.exact_rep_bursts))
    
main()
