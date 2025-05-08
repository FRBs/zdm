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
from zdm import figures
from zdm import parameters
from zdm import iteration as it
from zdm import loading as loading
import numpy as np

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
    opdir='RepeaterPlots/'
    if not os.path.exists(opdir):
        os.mkdir(opdir)
    
    # we choose a CHIME declination bin, because this survey has a significant
    # effect due to repeaters
    survey_name = 'CHIME_decbin_3_of_6'
    
    # make this into a list to initialise multiple surveys art once
    names = [survey_name]
    
    state = parameters.State()
    
    # directory where the survey files are located. The below is the default - 
    # you can leave this out, or change it for a different survey file location.
    sdir = resource_filename('zdm', 'data/Surveys/CHIME/')
    
    # use loading.survey_and_grid for proper estimates
    # remove loading for width-based estimates
    # the below is hard-coded for a *very* simplified analysis!
    # using loading. gives 5 beams and widths, ignoring that gives a single beam
    ss,gs = loading.surveys_and_grids(survey_names=names,repeaters=True,sdir=sdir)
    s = ss[0]
    g = gs[0]
    
    # plotting limits
    zmax=3.
    DMmax=3000.
    
    # fits the constant total number of FRBs to this survey
    #newC,llC=it.minimise_const_only(None,[g],[s])
    
    ############# do 2D plots ##########
    figures.plot_grid(g.rates,g.zvals,g.dmvals,
        name=opdir+survey_name+'all_frbs.pdf',norm=3,log=True,label='$\\log_{10} p({\\rm DM}_{\\rm EG},z)$  [a.u.]',
        project=False,FRBDMs=s.DMEGs,FRBZs=s.frbs["Z"],Aconts=[0.01,0.1,0.5],zmax=1.5,
        DMmax=1500)
    
    figures.plot_grid(g.exact_singles,g.zvals,g.dmvals,
        name=opdir+survey_name+'single_frbs.pdf',norm=3,log=True,label='$\\log_{10} p({\\rm DM}_{\\rm EG},z)$  [a.u.]',
        project=False,FRBDMs=s.DMEGs,FRBZs=s.frbs["Z"],Aconts=[0.01,0.1,0.5],zmax=1.5,
        DMmax=1500)
    
    figures.plot_grid(g.exact_reps,g.zvals,g.dmvals,
        name=opdir+survey_name+'repeating_sources.pdf',norm=3,log=True,label='$\\log_{10} p({\\rm DM}_{\\rm EG},z)$  [a.u.]',
        project=False,FRBDMs=s.DMEGs,FRBZs=s.frbs["Z"],Aconts=[0.01,0.1,0.5],zmax=1.5,
        DMmax=1500)
    
    figures.plot_grid(g.exact_rep_bursts,g.zvals,g.dmvals,
        name=opdir+survey_name+'bursts_from_repeaters.pdf',norm=3,log=True,label='$\\log_{10} p({\\rm DM}_{\\rm EG},z)$  [a.u.]',
        project=False,FRBDMs=s.DMEGs,FRBZs=s.frbs["Z"],Aconts=[0.01,0.1,0.5],zmax=1.5,
        DMmax=1500)
    
    
    ####### generate 1D plots of p(z) ########
    pztot = np.sum(g.rates,axis=1)* s.TOBS * 10**g.state.FRBdemo.lC # weight by Tobs wrst repeaters
    pzsingles = np.sum(g.exact_singles,axis=1) * g.Rc
    pzreps = np.sum(g.exact_reps,axis=1) * g.Rc
    pzbursts = np.sum(g.exact_rep_bursts,axis=1) * g.Rc
    pzsources = pzsingles+pzreps
    
    plt.figure()
    plt.plot(g.zvals,pztot,label="Total",linestyle="-",linewidth=2)
    plt.plot(g.zvals,pzsingles,label="Single bursts",linestyle="--")
    plt.plot(g.zvals,pzreps,label="Repeaters",linestyle="-")
    plt.plot(g.zvals,pzbursts,label="Bursts from repeaters",linestyle="-.")
    plt.plot(g.zvals,pzsources,label="Unique sources",linestyle=":")
    plt.xlabel('z')
    plt.ylabel('p(z) [a.u.]')
    plt.xlim(0,zmax)
    plt.ylim(0,None)
    plt.legend()
    plt.tight_layout()
    plt.savefig(opdir+survey_name+"_repeater_pz.png")
    plt.close()
    
main()
