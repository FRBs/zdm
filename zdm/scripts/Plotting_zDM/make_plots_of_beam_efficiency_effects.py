""" 
This script creates example plots beyond those produced
in make_simple_pzdm_plots.py

In particular, it plots p(z,DM) for the following special cases:


pEG_perfect_survey.png: shows p(DM,z) including
    - Standard effects:
        - source evolution
        - FRB luminosity function
        - cosmological volume calculation
        - detection threshold in Jyms
    - BUT:
        - Assumes B=1
        - Ignores DM and width smearing

pEG_efficiency_effect.png: shows p(DM,z) as pEG_luminosity, adding in
    detection efficiency losses from FRB width, DM smearing etc

pEG_beam_effect.pdf: shows p(DM,z) as with efficiency above,
    adding in antenna beamshape

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

import numpy as np
from zdm import survey
from matplotlib import pyplot as plt
from pkg_resources import resource_filename

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
    
    # set plotting limits
    zmax = 3
    DMmax = 3000
    ######## plots while ignoring beam, bias against DM ######
    
    # saves some data
    orig_beam_b=g.beam_b
    orig_beam_o=g.beam_o
    
    # sets up fake "efficiencies". Equivalent of no penalty to detection
    # due to DM smearing, FRB width, etc
    # not quite 'perfect' - "1" is relative to a width of 1ms, while
    # in theory an FRB with width 0.01 ms would be detectable at
    # efficiency of 10 (1% of noise, same fluence) by the proper system
    fake=np.full([g.dmvals.size],1.)
    weights=None
    g.calc_thresholds(s.meta['THRESH'],fake)
    
    # calculates the detection probability assuming a 'perfect' beam
    # i.e. 100% of beamshape has 100% of max sensitivity
    g.calc_pdv(beam_b=np.array([1.]),beam_o=np.array([1.]))
    g.calc_rates()
    
    
    figures.plot_grid(g.rates,g.zvals,g.dmvals,
        name=opdir+survey_name+'_pEG_perfect_survey.pdf',norm=3,log=True,
        label='$\\log_{10} p({\\rm DM}_{\\rm EG},z|B=1,\\epsilon=1)$ [a.u.]',
        project=False,ylabel='${\\rm DM}_{\\rm EG} = {\\rm DM}_{\\rm cosmic} + {\\rm DM}_{\\rm host}$',
        zmax=zmax,DMmax=DMmax)
               
    # adds in effect of FRB width and detection efficiency
    # does a false grid with no penalty to DM efficiency
    #zmax=1.5
    #DMmax=2000
    
    ####### Applies beamshape effect only #######
    g.calc_pdv(beam_b=orig_beam_b,beam_o=orig_beam_o)
    g.calc_rates()
    figures.plot_grid(g.rates,g.zvals,g.dmvals,
        name=opdir+survey_name+'_pEG_beam_effect.pdf',norm=3,log=True,
        label='$\\log_{10} p({\\rm DM}_{\\rm EG},z|\\epsilon=1)$ [a.u.]',
        project=False,ylabel='${\\rm DM}_{\\rm EG} = {\\rm DM}_{\\rm cosmic} + {\\rm DM}_{\\rm host}$',
        zmax=zmax,DMmax=DMmax)
    
    
    ##### plots the p(z,DM) grid with correct efficiencies, but ignoring beamshape #####
    g.calc_thresholds(s.meta['THRESH'],s.efficiencies,weights=s.wplist)
    g.calc_pdv(beam_b=np.array([1.]),beam_o=np.array([1.]))
    g.calc_rates()
    figures.plot_grid(g.rates,g.zvals,g.dmvals,
        name=opdir+survey_name+'_pEG_efficiency_effect.pdf',norm=3,log=True,
        label='$\\log_{10} p({\\rm DM}_{\\rm IGM} + {\\rm DM}_{\\rm host},z|b=1)$ [a.u.]',
        project=False,ylabel='${\\rm DM}_{\\rm IGM} + {\\rm DM}_{\\rm host}$',
        zmax=zmax,DMmax=DMmax)
    
    
main()
