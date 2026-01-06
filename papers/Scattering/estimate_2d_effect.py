"""
This script estimates the effect of bias in 2D vs 1D,
e.g. what is the difference between
efficiency = f(z,DM) and  f1(z)f2(DM)
"""

import os

from zdm import cosmology as cos
from zdm import figures
from zdm import parameters
from zdm import survey
from zdm import pcosmic
from zdm import misc_functions as mf
from zdm import iteration as it
from zdm import loading
from zdm import io
from pkg_resources import resource_filename
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
    """
    See main description at top of file.
    """
    
    Load = False
    
    # in case you wish to switch to another output directory
    
    opdir = "2Deffect/"
    names = ["CRAFT_ICS_1300"] # for example
    
    if not os.path.exists(opdir):
        os.mkdir(opdir)
    
    # directory where the survey files are located. The below is the default - 
    # you can leave this out, or change it for a different survey file location.
    sdir = os.path.join(resource_filename('zdm', 'data'), 'Surveys')
    
    # make this into a list to initialise multiple surveys art once. But here, we just use the standard
    # CRAFT ICS survey
    #names = ["CRAFT_ICS_1300"] # for example
    
    
    # gets single case of relevance
    survey_dict,state_dict = get_defined_states()
    
    # We ignore repetition... for now!
    repeaters=False
    
    # sets limits for calculations - both plotting, nad intrinsic. We want things to go quickly!
    zmax = 2.
    dmmax = 2000
    ndm=200
    nz=200
    
    if not Load:
        surveys, grids = loading.surveys_and_grids(survey_names = names,\
                        repeaters=repeaters, sdir=sdir,nz=nz,ndm=ndm,zmax=zmax,dmmax=dmmax,
                        survey_dict = survey_dict, state_dict = state_dict)
        s = surveys[0]
        g = grids[0]
        
        rates = g.rates
        np.save(opdir+"rates.npy",rates)
        dmvals = g.dmvals
        zvals = g.zvals
        np.save(opdir+"dmvals.npy",dmvals)
        np.save(opdir+"zvals.npy",zvals)
        np.save(opdir+"efficiencies.npy",s.efficiencies)
        
        
    else:
        rates = np.load(opdir+"rates.npy")
        dmvals = np.load(opdir+"dmvals.npy")
        zvals = np.load(opdir+"zvals.npy")
        efficiencies = np.load("efficiencies.npy")
    
        
    
    # now we have rates, let's look at the effects!
    
    labels=["No width/scattering","CHIME Catalogue 1","Lognormals (this work)","Best fit (this work)","(numerical approx)"]
    styles=[":","-.","--","-",":"]
    
    pzvals = [0.01,0.5,1,1.5,2]#0.5,1,1.5,2]
    colors=[]
    jnorm=3
    
    #pzvals=[0.01,0.5,1,1.5,2]
    pzvals=[0.01,2,1.5,1,0.5]
    plt.figure()
    weffs=[]
    for i,z in enumerate(pzvals):
        iz = np.where(zvals <= z)[0][-1]
        
        # weights this over all wieght bins
        weighted_eff = np.array(s.efficiencies[:,iz,:]).T*np.array(s.wplist[:,iz])
        #now we sum this over those bins
        weighted_eff = np.sum(weighted_eff,axis=1) # summed over weight bins
        weffs.append(weighted_eff)
    
    for i,z in enumerate(pzvals):
        if i==0:
            continue
        plt.plot(g.dmvals,weffs[i]/weffs[0],label="z = "+str(z)[0:3],linestyle=styles[i-1])
    
    plt.xlabel("DM [pc cm$^{-3}$]")
    plt.ylabel("$\\frac{\\epsilon({\\rm DM}|z)}{\\epsilon({\\rm DM}|z=0)}$")
    
    plt.xlim(0,2000)
    plt.ylim(1.08,1.2)
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(opdir+"efficiency.png")
    plt.close()
    
    

def get_defined_states():
    """
    This function just contains hard-coded state and survey dicts
    corresponding to four cases of interest
    """
    # we now set up three examples. These are:
    #1: no width/scattering distributions. Treat width=1ms (nominal)
    #2: CHIME width/scattering distributions (no z-dependence_
    #3: bestfit lognormal function of this work (with z-dependence)
    #4: bestfit functions of this work (with z-dependence)
    
    # sets up survey dictionaries for width methods
    survey_dict4 = {"WMETHOD": 3} # z-dependence
    
    
    state_dict4 = {}
    state_dict4["scat"] = {}
    state_dict4["width"] = {}
    state_dict4["width"]["WNbins"] = 100 # set to large number for this analysis
    state_dict4["width"]["WidthFunction"] = 2 # half-lognormal
    state_dict4["width"]["ScatFunction"] = 0 # log-uniform true value 0
    state_dict4["width"]["Wlogmean"] = -0.29    # -0.29
    state_dict4["width"]["Wlogsigma"] = 0.65    #0.65
    state_dict4["scat"]["Slogmean"] = -1.38 # true valiue -1.38
    state_dict4["scat"]["Slogsigma"] = 0.01 # not actually used in the log-constant distribution
    state_dict4["scat"]["Sfnorm"] = 1000 # 1 GHz normalisation
    
    
    return survey_dict4,state_dict4

main()
