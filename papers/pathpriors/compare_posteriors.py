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
from zdm import frb_lists as lists

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
    
    
    ######### List of all ICS FRBs for which we can run PATH #######
    # hard-coded list of FRBs with PATH data in ice paper
    frblist = lists.icslist
    
    # Initlisation of zDM grid
    # Eventually, this should be part of the loop, i.e. host IDs should
    # be re-fed into FRB surveys. However, it will be difficult to do this
    # with very limited redshift estimates. That might require posterior
    # estimates of redshift given the observed galaxies. Maybe.
    state = states.load_state("HoffmannHalo25",scat=None,rep=None)
    #state = parameters.State()
    cos.set_cosmology(state)
    cos.init_dist_measures()
    names=['CRAFT_ICS_892','CRAFT_ICS_1300','CRAFT_ICS_1632']
    ss,gs = loading.surveys_and_grids(survey_names=names,init_state=state)
    
    
    # initialise figure. ax1 is new vs old. ax2 is new-old vs old
    plt.figure()
    ax1=plt.gca()
    plt.xlabel("$P(O_i| \\mathbf{x})$ [original; $P(U)=0.1$]")
    plt.ylabel("$P(O_i| \\mathbf{x},N_O)$ [this work]")
    
    plt.figure()
    ax2=plt.gca()
    plt.xlabel("$P(O_i| \\mathbf{x})$ [original; $P(U)=0.1$]")
    plt.ylabel("$\Delta P(O_i| \\mathbf{x},N_O)$")
    
    
    ##### Begins by calculating the original PATH result #####
    # calculates the original PATH result
    wrappers = None
    NFRB2,AppMags2,AppMagPriors2,ObsMags2,ObsPrior2,ObsPosteriors2,PUprior2,PUobs2,sumPUprior2,sumPUobs2,frbs,dms = \
                                            on.calc_path_priors(frblist,ss,gs,wrappers,verbose=False,usemodel=False,P_U=0.1)
    fObsPosteriors2 = np.array(on.flatten(ObsPosteriors2))
    
    
    with open("posteriors/orig.txt",'w') as f:
        for i,frb in enumerate(frbs):
            f.write(str(i)+"  "+frb+"  "+str(dms[i])[0:5]+" "+str(PUprior2[i])[0:4]+"\n")
            for j,om in enumerate(ObsMags2[i]):
                f.write(str(om)[0:5]+" "+ "%e" % ObsPrior2[i][j]+" "+"%e" % ObsPosteriors2[i][j]+"\n")
            f.write("\n")
    
    
    #### creates some lists to later pass to make_cumulative_plots ####
    NFRBlist = []
    AppMagslist = []
    AppMagPriorslist = []
    ObsMagslist = []
    ObsPosteriorslist = []
    PUpriorlist = []
    PUobslist = []
    sumPUpriorlist = []
    sumPUobslist = []
    
    
    ####### Model 1: Marnoch ########
    
    # model 1: Marnoch
    model = opt.marnoch_model()
    
    wrappers = on.make_wrappers(model,gs)
    NFRB1,AppMags1,AppMagPriors1,ObsMags1,ObsPrior1,ObsPosteriors1,PUprior1,PUobs1,sumPUprior1,sumPUobs1,frbs,dms = on.calc_path_priors(frblist,ss,gs,wrappers,verbose=False)
    
    fObsPosteriors1 = np.array(on.flatten(ObsPosteriors1))
    plt.sca(ax1)
    plt.scatter(fObsPosteriors2,fObsPosteriors1,label="Marnoch23",marker='s')
    plt.sca(ax2)
    plt.scatter(fObsPosteriors2,fObsPosteriors1-fObsPosteriors2,label="Marnoch23",marker='s')
    
    
    with open("posteriors/marnoch.txt",'w') as f:
        for i,frb in enumerate(frbs):
            f.write(str(i)+"  "+frb+"  "+str(dms[i])[0:5]+" "+str(PUprior1[i])[0:4]+"\n")
            for j,om in enumerate(ObsMags1[i]):
                f.write(str(om)[0:5]+" "+ "%e" % ObsPrior1[i][j]+" "+"%e" % ObsPosteriors1[i][j]+"\n")
            f.write("\n")
    
    ####### Model 2: Loudas ########
    
    model = opt.loudas_model()
    xbest = np.load("loudas_output/best_fit_params.npy")
    model.init_args(xbest) # best-fit arguments
    wrappers = on.make_wrappers(model,gs)
    NFRB3,AppMags3,AppMagPriors3,ObsMags3,ObsPrior3,ObsPosteriors3,PUprior3,PUobs3,sumPUprior3,sumPUobs3,frbs,dms = on.calc_path_priors(frblist,ss,gs,wrappers,verbose=False)
    
    fObsPosteriors3 = np.array(on.flatten(ObsPosteriors3))
    plt.sca(ax1)
    plt.scatter(fObsPosteriors2,fObsPosteriors3,label="Loudas25",marker='x')
    plt.sca(ax2)
    plt.scatter(fObsPosteriors2,fObsPosteriors3-fObsPosteriors2,label="Loudas25",marker='x')
    
    with open("posteriors/loudas.txt",'w') as f:
        for i,frb in enumerate(frbs):
            f.write(str(i)+"  "+frb+"  "+str(dms[i])[0:5]+" "+str(PUprior3[i])[0:4]+"\n")
            for j,om in enumerate(ObsMags3[i]):
                f.write(str(om)[0:5]+" "+ "%e" % ObsPrior3[i][j]+" "+"%e" % ObsPosteriors3[i][j]+"\n")
            f.write("\n")
    
    
    
    ####### Model 3: Simple ########
    
    # Case of simple host model
    opstate = op.OpticalState()
    # sets optical state to use simple linear interpolation
    opstate.simple.AbsModelID = 1 # linear interpolation
    opstate.simple.AppModelID = 1 # include k-correction
    opstate.simple.NModelBins = 6
    opstate.simple.Absmin = -25
    opstate.simple.Absmax = -15
    model = opt.simple_host_model(opstate)
    
    # retrieve default initial arguments in vector form
    xbest = np.load("simple_output/best_fit_params.npy")
    #x = [-2.28795519,  0.,  0. , 0. , 0.11907231,0.84640048,0.99813815 , 0., 0. , 0. , 0. ]
    
    # initialises best-fit arguments
    model.init_args(xbest)
    
    ############# Generate a KS-like plot showing the best fits ####################
    wrappers = on.make_wrappers(model,gs)
    NFRB4,AppMags4,AppMagPriors4,ObsMags4,ObsPrior4,ObsPosteriors4,PUprior4,PUobs4,sumPUprior4,sumPUobs4,frbs,dms = on.calc_path_priors(frblist,ss,gs,wrappers,verbose=False)
    
    fObsPosteriors4 = np.array(on.flatten(ObsPosteriors4))
    plt.sca(ax1)
    plt.scatter(fObsPosteriors2,fObsPosteriors4,label="Naive",marker='o',s=20)
    plt.sca(ax2)
    plt.scatter(fObsPosteriors2,fObsPosteriors4-fObsPosteriors2,label="Naive",marker='o',s=20)
    
    # format and save ax1
    plt.sca(ax1)
    plt.legend(loc="lower right")
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.plot([0,1],[0,1],color="black",linestyle=":")
    plt.tight_layout()
    plt.savefig("posteriors/pox_comparison.png")
    plt.close()
    
    # format and save ax2
    plt.sca(ax2)
    plt.legend(loc="upper left")
    plt.xlim(0,1)
    plt.ylim(-0.2,0.2)
    plt.plot([0,1],[0,0],color="black",linestyle=":")
    plt.text(0.4,0.01,"no change")
    plt.plot([0.8,1],[0.2,0],color="black",linestyle=":")
    plt.text(0.85,0.08,"$P(O_i| \\mathbf{x})=1$",rotation=-60)
    
    plt.plot([0,0.2],[0,-0.2],color="black",linestyle=":")
    plt.text(0.03,-0.18,"$P(O_i| \\mathbf{x})=0$",rotation=-60)
    
    plt.tight_layout()
    plt.savefig("posteriors/delta_pox_comparison.png")
    plt.close()
    
    with open("posteriors/naive.txt",'w') as f:
        for i,frb in enumerate(frbs):
            f.write(str(i)+"  "+frb+"  "+str(dms[i])[0:5]+" "+str(PUprior4[i])[0:4]+"\n")
            for j,om in enumerate(ObsMags4[i]):
                f.write(str(om)[0:5]+" "+ "%e" % ObsPrior4[i][j]+" "+"%e" % ObsPosteriors4[i][j]+"\n")
            f.write("\n")
    
    all_candidates = on.get_cand_properties(frblist)
    
    # now iterates through galaxies and writes relevant info
    for i in np.arange(NFRB1):
        string1="\multicolumn{5}{c|}{"+frbs[i]+"} & "
        string1 += f"{PUprior2[i]:.3f} &  {PUobs2[i]:.3f} & "
        string1 += f"{PUprior1[i]:.3f} &  {PUobs1[i]:.3f} & "
        string1 += f"{PUprior3[i]:.3f} &  {PUobs3[i]:.3f} & "
        string1 += f"{PUprior4[i]:.3f} &  {PUobs4[i]:.3f} \\\\ "
        print("\\hline")
        print(string1)
        print("\\hline")
        
        
        
        for j,mag in enumerate(ObsMags4[i]):
            # check if we print this one at all
            if ObsPosteriors1[i][j] < 1e-4 and ObsPosteriors2[i][j] < 1e-4 \
                and ObsPosteriors3[i][j] < 1e-4 and ObsPosteriors4[i][j] < 1e-4:
                
                continue
            
            string2 = f"{all_candidates[i]['ra'][j]:.4f} & {all_candidates[i]['dec'][j]:.4f} &"
            string2 += f" {all_candidates[i]['separation'][j]:.2f} &"
            string2 += f" {all_candidates[i]['ang_size'][j]:.2f} & {all_candidates[i]['mag'][j]:.2f} &"
            
            
            string2 += f"{ObsPrior2[i][j]:.3f} &  {ObsPosteriors2[i][j]:.3f} & "
            string2 += f"{ObsPrior1[i][j]:.3f} &  {ObsPosteriors1[i][j]:.3f} & "
            string2 += f"{ObsPrior3[i][j]:.3f} &  {ObsPosteriors3[i][j]:.3f} & "
            string2 += f"{ObsPrior4[i][j]:.3f} &  {ObsPosteriors4[i][j]:.3f} \\\\ "
            print(string2)
    
    
    
    ######## Makes cumulative distribution KS-style plots
    
    # loads various marnoch models
    model = opt.loudas_model()
    xbest = np.load("loudas_output/best_fit_params.npy")
    for f_sfr in [0,1,xbest]:
        x=[f_sfr]
        model.init_args(x)
        wrappers = on.make_wrappers(model,gs)
        NFRB,AppMags,AppMagPriors,ObsMags,ObsPriors,ObsPosteriors,PUprior,PUobs,sumPUprior,sumPUobs,frbs,dms = on.calc_path_priors(frblist,ss,gs,wrappers,verbose=False)
        
        NFRBlist.append(NFRB)
        AppMagslist.append(AppMags)
        AppMagPriorslist.append(AppMagPriors)
        ObsMagslist.append(ObsMags)
        ObsPosteriorslist.append(ObsPosteriors)
        PUpriorlist.append(PUprior)
        PUobslist.append(PUobs)
        sumPUpriorlist.append(sumPUprior)
        sumPUobslist.append(sumPUobs)
    
    # loads naive model
    NFRBlist.append(NFRB4)
    AppMagslist.append(AppMags4)
    AppMagPriorslist.append(AppMagPriors4)
    ObsMagslist.append(ObsMags4)
    ObsPosteriorslist.append(ObsPosteriors4)
    PUpriorlist.append(PUprior4)
    PUobslist.append(PUobs4)
    sumPUpriorlist.append(sumPUprior4)
    sumPUobslist.append(sumPUobs4)
    
    
    # loads Marnoch model
    NFRBlist.append(NFRB1)
    AppMagslist.append(AppMags1)
    AppMagPriorslist.append(AppMagPriors1)
    ObsMagslist.append(ObsMags1)
    ObsPosteriorslist.append(ObsPosteriors1)
    PUpriorlist.append(PUprior1)
    PUobslist.append(PUobs1)
    sumPUpriorlist.append(sumPUprior1)
    sumPUobslist.append(sumPUobs1)
    
    
    plotlabels=["Loudas25: $f_{\\rm sfr} = 0$", "                  $f_{\\rm sfr} = 1$",
                     "                  $f_{\\rm sfr} = 3$","Naive","Marnoch23"]
    plotfile="Plots/all_cumulative.png"
    NMODELS=5
    on.make_cumulative_plots(NMODELS,NFRBlist,AppMagslist,AppMagPriorslist,ObsMagslist,ObsPosteriorslist,
                            PUobslist,PUpriorlist,plotfile,plotlabels,POxcut=None,onlyobs=0,addpriorlabel=False)
    
main()
