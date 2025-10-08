"""
This script compares the p(z,DM) distributions
created for the CRAFT ICS survey for different
treatments of scattering/width.

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
    
    Load = True
    
    # in case you wish to switch to another output directory
    
    if False:
        opdir = "FastComparisons/"
        names = ["FAST"]
    else:
        opdir = "Comparisons/"
        names = ["CRAFT_ICS_1300"] # for example
    
    if not os.path.exists(opdir):
        os.mkdir(opdir)
    
    # directory where the survey files are located. The below is the default - 
    # you can leave this out, or change it for a different survey file location.
    sdir = os.path.join(resource_filename('zdm', 'data'), 'Surveys')
    
    # make this into a list to initialise multiple surveys art once. But here, we just use the standard
    # CRAFT ICS survey
    #names = ["CRAFT_ICS_1300"] # for example
    
    
    # gets four cases of relevance
    survey_dicts,state_dicts = get_defined_states()
    
    # We ignore repetition... for now!
    repeaters=False
    
    # sets limits for calculations - both plotting, nad intrinsic. We want things to go quickly!
    zmax = 5.
    dmmax = 5000
    ndm=1000
    nz=1000
    
    rates = []
    for i,state_dict in enumerate(state_dicts):
        if Load:
            continue
        survey_dict = survey_dicts[i]
    
        surveys, grids = loading.surveys_and_grids(survey_names = names,\
                        repeaters=repeaters, sdir=sdir,nz=nz,ndm=ndm,zmax=zmax,dmmax=dmmax,
                        survey_dict = survey_dict, state_dict = state_dict)
        s = surveys[0]
        g = grids[0]
        
        rates.append(g.rates)
    
    if Load:
        rates = np.load(opdir+"rates.npy")
        dmvals = np.load(opdir+"dmvals.npy")
        zvals = np.load(opdir+"zvals.npy")
    else:
        np.save(opdir+"rates.npy",rates)
        dmvals = g.dmvals
        zvals = g.zvals
        np.save(opdir+"dmvals.npy",dmvals)
        np.save(opdir+"zvals.npy",zvals)
    
    # now we have rates, let's look at the effects!
    
    labels=["No width/scattering","CHIME Catalogue 1","Lognormals (this work)","Best fit (this work)","(numerical approx)"]
    styles=[":","-.","--","-",":"]
    
    # generates a plot of p(z)
    plt.figure()
    ax1 = plt.gca()
    
    # generates a plot of p(dm)
    plt.figure()
    ax2 = plt.gca()
    
    
    # generates a plot of p(z) relative to "truth"
    plt.figure()
    ax3 = plt.gca()
    
    # generates a plot of p(dm) relativer to "truth"
    plt.figure()
    ax4 = plt.gca()
    
    dz = zvals[1]-zvals[0]
    ddm = dmvals[1]-dmvals[0]
    
    zdists=[]
    dmdists=[]
    for i,r in enumerate(rates):
        
        r /= np.sum(r)
        zdist = np.sum(r,axis=1)
        dmdist = np.sum(r,axis=0)
        zdists.append(zdist)
        dmdists.append(dmdist)
    
    
    inorm=3 # this is the "correct" one
    for i,r in enumerate(rates):
        zdist = zdists[i]
        dmdist = dmdists[i]
        
        ax1.plot(zvals,zdist/dz,label=labels[i],linestyle=styles[i])
        ax2.plot(dmvals,dmdist/ddm,label=labels[i],linestyle=styles[i])
        
        ax3.plot(zvals,zdist/zdists[inorm],label=labels[i],linestyle=styles[i])
        ax4.plot(dmvals,dmdist/dmdists[inorm],label=labels[i],linestyle=styles[i])
    
    plt.sca(ax1)
    plt.legend()
    plt.xlabel("z")
    plt.ylabel("p(z)")
    plt.yscale("log")
    #plt.ylim(0,1e-3)
    plt.xlim(0,2)
    plt.tight_layout()
    plt.savefig(opdir+"zdists.png")
    plt.close()
    
    plt.sca(ax2)
    plt.legend()
    plt.xlabel("DM [pc cm$^{-3}$]")
    plt.ylabel("p(DM)")
    plt.yscale("log")
    plt.tight_layout()
    plt.savefig(opdir+"dmdists.png")
    plt.close()
    
    
    #### relative plots ####
    plt.sca(ax3)
    plt.legend()
    plt.xlabel("$z$")
    plt.ylabel("$p(z)/p_{\\rm best}(z)$")
    #plt.yscale("log")
    #plt.ylim(0,1e-3)
    plt.xlim(0,2)
    plt.tight_layout()
    plt.savefig(opdir+"rel_zdists.png")
    plt.close()
    
    plt.sca(ax4)
    plt.legend()
    plt.xlabel("DM [pc cm$^{-3}$]")
    plt.ylabel("p(DM)")
    plt.tight_layout()
    plt.savefig(opdir+"rel_dmdists.png")
    plt.close()  
    
    
    ##### Plots of p(DM) for specific z, illustrating the z-dependence of DM bias #####
    plt.figure()
    
    pzvals = [0.01,1]#0.5,1,1.5,2]
    colors=[]
    jnorm=3
    for i,z in enumerate(pzvals):
        # gets index
        iz = np.where(zvals <= z)[0][-1]
        
        for j,g in enumerate(rates):
            
            if i==0:
                label=labels[j]
            else:
                label=None
            pdm = rates[j][iz,:]
            relpdm = pdm/rates[jnorm][iz,:]
            if i==0:
                color=None
            else:
                color=colors[j]
            p=plt.plot(dmvals,relpdm+i*0.4,label=label,linestyle=styles[j],color=color)
            if i==0:
                colors.append(p[-1].get_color())
    plt.xlabel("DM [pc cm$^{-3}$]")
    plt.ylabel("$p({\\rm DM})/p_{\\rm best}({\\rm DM})$")
    plt.legend(fontsize=6)
    plt.tight_layout()
    plt.savefig(opdir+"pdm_vs_z_methods.png")
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
    survey_dict1 = {"WMETHOD": 0} # ignore it - just 1ms
    survey_dict2 = {"WMETHOD": 2} # scattering and width
    survey_dict3 = {"WMETHOD": 3} # z-dependence
    survey_dict4 = {"WMETHOD": 3} # z-dependence
    survey_dict5 = {"WMETHOD": 3} # z-dependence
    
    
    # now sets up state dictionaries
    state_dict1 = {}
    state_dict1["scat"] = {}
    state_dict1["width"] = {}
    
    state_dict2 = {}
    state_dict2["scat"] = {}
    state_dict2["width"] = {}
    state_dict2["width"]["WNbins"] = 100 # set to large number for this analysis
    state_dict2["width"]["WidthFunction"] = 1 # lognormal
    state_dict2["width"]["ScatFunction"] = 1 # lognormal
    state_dict2["width"]["Wlogmean"] = 0. # 1ms
    state_dict2["width"]["Wlogsigma"] = 0.42 # 0.97 in ln
    state_dict2["scat"]["Slogmean"] = 0.3 # 2.02 ms#
    state_dict2["scat"]["Slogsigma"] = 0.74 # 1.72 in ln
    state_dict2["scat"]["Sfnorm"] = 600 # 600 MHz normalisation - for clarity
    
    state_dict3 = {}
    state_dict3["scat"] = {}
    state_dict3["width"] = {}
    state_dict3["width"]["WNbins"] = 100 # set to large number for this analysis
    state_dict3["width"]["WidthFunction"] = 1 # lognormal
    state_dict3["width"]["ScatFunction"] = 1 # lognormal
    state_dict3["width"]["Wlogmean"] = 0.22
    state_dict3["width"]["Wlogsigma"] = 0.88
    state_dict3["scat"]["Slogmean"] = 0.36
    state_dict3["scat"]["Slogsigma"] = 1.05
    state_dict3["scat"]["Sfnorm"] = 1000 # 1 GHz normalisation
    
    if False:
        state_dict3 = {}
        state_dict3["scat"] = {}
        state_dict3["width"] = {}
        state_dict3["width"]["WNbins"] = 100 # set to large number for this analysis
        state_dict3["width"]["WidthFunction"] = 2 # half-lognormal
        state_dict3["width"]["ScatFunction"] = 2 # log-uniform true value 0
        state_dict3["width"]["Wlogmean"] = -0.29    # -0.29
        state_dict3["width"]["Wlogsigma"] = 0.65    #0.65
        state_dict3["scat"]["Slogmean"] = -0.38 # true valiue -1.38
        state_dict3["scat"]["Slogsigma"] = 0.01 # not actually used in the log-constant distribution
        state_dict3["scat"]["Sfnorm"] = 1000 # 1 GHz normalisation
    
    
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
    
    state_dict5 = {}
    state_dict5["scat"] = {}
    state_dict5["width"] = {}
    state_dict5["width"]["WNbins"] = 5 # set to large number for this analysis
    state_dict5["width"]["WidthFunction"] = 2 # half-lognormal
    state_dict5["width"]["ScatFunction"] = 0 # log-uniform true value 0
    state_dict5["width"]["Wlogmean"] = -0.29    # -0.29
    state_dict5["width"]["Wlogsigma"] = 0.65    #0.65
    state_dict5["scat"]["Slogmean"] = -1.38 # true valiue -1.38
    state_dict5["scat"]["Slogsigma"] = 0.01 # not actually used in the log-constant distribution
    state_dict5["scat"]["Sfnorm"] = 1000 # 1 GHz normalisation
    
    
    state_dicts = [state_dict1, state_dict2, state_dict3, state_dict4, state_dict5]
    survey_dicts = [survey_dict1, survey_dict2, survey_dict3, survey_dict4, survey_dict5]
    
    return survey_dicts,state_dicts

main()
