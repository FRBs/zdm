"""
Used to generate final fitted P(m|DM) figures

"""

#standard Python imports
import os
import numpy as np
from matplotlib import pyplot as plt

# imports from the "FRB" series
from zdm import optical as opt
from zdm import optical_params as op
from zdm import optical_numerics as on
from zdm import loading
from zdm import cosmology as cos
from zdm import parameters
from zdm import loading
from zdm import frb_lists as lists

import astropath.priors as pathpriors

import matplotlib

defaultsize=14
ds=4
font = {'family' : 'Helvetica',
        'weight' : 'normal',
        'size'   : defaultsize}
matplotlib.rc('font', **font)

def calc_path_priors():
    """
    Loops over all ICS FRBs
    """
    
    opdir = "Plots/"
    if not os.path.exists(opdir):
        os.mkdir(opdir)
    ##### performs the following calculations for the below combinations ######
    
    ######## initialises optical-independent info ########
    #frblist is a hard-coded list of FRBs for which we have optical PATH data
    frblist = lists.icslist
    NFRB = len(frblist)
    
    
    state = parameters.State()
    cos.set_cosmology(state)
    cos.init_dist_measures()
    
    ##### makes a plot of host priors for the simple model ######
    
    # simple host model
    # Case of simple host model
    opstate1 = op.OpticalState()
    # sets optical state to use simple linear interpolation
    opstate1.simple.AbsModelID = 1 # linear interpolation
    opstate1.simple.AppModelID = 1 # include k-correction
    opstate1.simple.NModelBins = 6
    opstate1.simple.Absmin = -25
    opstate1.simple.Absmax = -15
    model1 = opt.simple_host_model(opstate1)
    # this is from an initial estimate. Currently, no way to enter this into the opstate. To do.
    xbest = np.load("simple_output/best_fit_params.npy")
    model1.init_args(xbest)
    
    
    model2=opt.loudas_model()
    xbest = np.load("loudas_output/best_fit_params.npy")
    model2.init_args(xbest) # best-fit arguments
    
    # set up basic histogram of p(mr) distribution
    mrbins = np.linspace(0,40,401)
    mrvals=(mrbins[:-1]+mrbins[1:])/2.
    dmr = mrbins[1]-mrbins[0]
    
    model3 = opt.marnoch_model()
    
    ######### Plots apparent mag distribution for all models as function of z #######
    styles=["-",":","--","-."]
    
    plt.figure()
    flist=[1,3] #normal distributions, and best fit
    
    for i,z in enumerate([0.1,1.0]):
        
        # simple model
        pmr = model1.get_pmr_gz(mrbins,z)
        pmr /= np.sum(pmr)
        
        if i==0:
            plt.plot(mrvals,pmr/dmr,label="Naive",linestyle=styles[0])
        else:
            plt.plot(mrvals,pmr/dmr,linestyle=styles[0],\
                color=plt.gca().lines[0].get_color())
        
        pmr = model3.get_pmr_gz(mrbins,z)
        if i==0:
            plt.plot(mrvals,pmr/dmr,label = "Marnoch23",linestyle=styles[3])
        else:
            plt.plot(mrvals,pmr/dmr,linestyle=styles[3],color=plt.gca().lines[1].get_color())
        
        
        # Loudas model dependencies
        for j,fsfr in enumerate(flist):
            model2.init_args(fsfr)
            pmr = model2.get_pmr_gz(mrbins,z)
            pmr /= np.sum(pmr)
            if i==0:
                plt.plot(mrvals,pmr/dmr,label = "Loudas25 ($f_{\\rm sfr}$ = "+str(fsfr)+")",
                    linestyle=styles[j+1])
            else:
                plt.plot(mrvals,pmr/dmr,linestyle=styles[j+1],\
                        color=plt.gca().lines[j+2].get_color())
        
        
        
    plt.xlabel("Apparent magnitude $m_r$")
    plt.ylabel("$P(m_r|z)$")
    plt.text(17.5,0.285,"$z=0.1$")
    plt.text(22,0.285,"$z=1.0$")
    plt.xlim(12.5,30)
    plt.ylim(0,0.4)
    plt.legend(loc="upper right",ncol=2)
    #plt.legend(loc=[25,0.35])
    
    plt.tight_layout()
    plt.savefig(opdir+"all_model_apparent_mags.png")
    plt.close()
    
    

if __name__ == "__main__":
    
    calc_path_priors()

    
    
