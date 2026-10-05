"""
This file fits the simple (naive) model to CRAFT ICS observations.
It fits a model of absolute galaxy magnitude distributions,
uses zDM to predict redshifts and hence apparent magntidues,
runs PATH using that prior, and tries to get priors to match posteriors.

It also geenrates host z-mr plots

"""


#standard Python imports
import os
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import minimize

# imports from the "FRB" series
from zdm import optical as opt
from zdm import optical_params as op
from zdm import loading
from zdm import cosmology as cos
from zdm import parameters
from zdm import loading
from zdm import optical_numerics as on
from zdm import states

from importlib import resources

# other FRB library imports
import astropath.priors as pathpriors

import matplotlib

defaultsize=14
ds=4
font = {'family' : 'Helvetica',
        'weight' : 'normal',
        'size'   : defaultsize}
matplotlib.rc('font', **font)


def main():
    """
    Main function
    Contains outer loop to iterate over parameters
    
    """
    ########## Sets optical parameters #########
    
    opdir="Plots/"
    # Case of simple host model
    opstate = op.OpticalState()
    
    # sets optical state to use simple model
    opstate.simple.AbsModelID = 1 # linear interpolation
    opstate.simple.AppModelID = 1 # include k-correction
    opstate.simple.NModelBins = 6
    opstate.simple.Absmin = -25
    opstate.simple.Absmax = -15
    
    # sets the parameters of the P(O|m) function
    TELs=["Pan-STARRS","Legacy Surveys","VLT/FORS2"]
    TelMeans = [21.8,24.0,26.4]
    TelSigmas = [0.54,0.55,0.28]
    
    
    # Initlisation of zDM grid
    # Eventually, this should be part of the loop, i.e. host IDs should
    # be re-fed into FRB surveys. However, it will be difficult to do this
    # with very limited redshift estimates. That might require posterior
    # estimates of redshift given the observed galaxies. Maybe.
    state = states.load_state("HoffmannHalo25",scat=None,rep=None)
    #state = parameters.State()
    
    labels=['(a) ASKAP/ICS 900 MHz','(b) CHIME ($\delta=60^{\circ}$)','(c) MeerKAT coherent','(d) DSA-110']
    tags=['ASKAP','CHIME','MeerKAT','DSA']
    names=['CRAFT_ICS_892','CHIME/CHIME_decbin_3_of_6','MeerTRAPcoherent','DSA']#,'CRAFT_ICS_1300','CRAFT_ICS_1632']
    ss,gs = loading.surveys_and_grids(survey_names=names,init_state=state)
    
    
    # initialise figure
    styles=[":","--","-"]
    NDM=20
    DMlist = np.linspace(50,1950,NDM)
    locs = ["upper left","upper center","upper right"]
    
    for j,g in enumerate(gs):
        plt.figure()
        plt.xlabel("$\\rm DM_{\\rm EG}$ [pc cm$^{-3}$]")
        plt.ylabel("$P(U|{\\rm DM_{\\rm EG}})$")
        tag = tags[j]
        label = labels[j]
        plt.text(-250,1.04,label)
        lines = []
        legends = []
        for i in np.arange(3):
            
            opstate.id.pU_mean=TelMeans[i]
            opstate.id.pU_width=TelSigmas[i]
        
            PUs = get_PUs(opstate,g,DMlist)
            
            
            if i==0:
                l1,=plt.plot(DMlist,PUs[:,0],label="Marnoch23",linestyle=styles[i])
                l2,=plt.plot(DMlist,PUs[:,1],label="Loudas25",linestyle=styles[i])
                l3,=plt.plot(DMlist,PUs[:,2],label="Naive",linestyle=styles[i])
                
            else:
                l1,=plt.plot(DMlist,PUs[:,0],label="Marnoch23",linestyle=styles[i],color=l1.get_color())
                l2,=plt.plot(DMlist,PUs[:,1],label="Loudas25",linestyle=styles[i],color=l2.get_color())
                l3,=plt.plot(DMlist,PUs[:,2],label="Naive",linestyle=styles[i],color=l3.get_color())
            lines.append([l1,l2,l3])
            if j==0:
                leg = plt.legend(labels = ["Marnoch23","Loudas25","Naive"], handles = [l1,l2,l3],title=TELs[i],loc=locs[i],fontsize=12)
                plt.setp(leg.get_title(),fontsize='12')
                legends.append(leg)
            
            #if i==0:
            #    l1,=plt.plot(DMlist,PUs[:,0],linestyle=styles[i])
            #    l2,=plt.plot(DMlist,PUs[:,1],linestyle=styles[i])
            #    l3,=plt.plot(DMlist,PUs[:,2],linestyle=styles[i])
            #elif i==2:
            #    plt.plot(DMlist,PUs[:,0],label="Marnoch23",linestyle=styles[i],color=l1.get_color())
            #    plt.plot(DMlist,PUs[:,1],label="Loudas25",linestyle=styles[i],color=l2.get_color())
            #    plt.plot(DMlist,PUs[:,2],label="Naive",linestyle=styles[i],color=l3.get_color())
            #else:
            #    plt.plot(DMlist,PUs[:,0],linestyle=styles[i],color=l1.get_color())
            #    plt.plot(DMlist,PUs[:,1],linestyle=styles[i],color=l2.get_color())
            #    plt.plot(DMlist,PUs[:,2],linestyle=styles[i],color=l3.get_color())
            # plot kind of optical observation
        for i in np.arange(3):
            Tlabel=TELs[i]
            plt.plot([--100,-50],[-10,-10],color="black",label=Tlabel,linestyle=styles[i])
        
        plt.ylim(0,1)
        plt.xlim(0,2000)
        
        if j==0:
        
            plt.gca().add_artist(legends[0])
            plt.gca().add_artist(legends[1])
            #plt.gca().add_artist(legends[2])
            #plt.legend(fontsize=12)
        plt.tight_layout()
        plt.savefig(opdir+tag+"pu_all.png")
        plt.close()
    
    
def get_PUs(opstate,g,DMlist):
    """
    Returns P(U) array for three models
    """
    ######### Initialise models ###########
    
    # model 1: Marnoch
    model1 = opt.marnoch_model(opstate)
    
    # Loudas
    model2 = opt.loudas_model(opstate)
    xbest = np.load("loudas_output/best_fit_params.npy")
    model2.init_args(xbest)
    
    
    model3 = opt.simple_host_model(opstate)
    
    # retrieve default initial arguments in vector form
    xbest = np.load("simple_output/best_fit_params.npy")
    model3.init_args(xbest)
    
    
    wrapper1 = opt.model_wrapper(model1,g.zvals)
    wrapper2 = opt.model_wrapper(model2,g.zvals)
    wrapper3 = opt.model_wrapper(model3,g.zvals)
    wrappers = [wrapper1,wrapper2,wrapper3]
    
    # iterates of DM list
    NM=len(wrappers)
    NDM = len(DMlist)
    PUs = np.zeros([NDM,NM])
    for i,DMEG in enumerate(DMlist):
        for j,wrapper in enumerate(wrappers):
            wrapper.init_path_raw_prior_Oi(DMEG,g)
            PU = wrapper.estimate_unseen_prior()
            PUs[i,j]=PU
        #print(DMEG,PU)
    
    return PUs
    
main()
