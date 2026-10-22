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
    results2 = on.calc_path_priors(frblist,ss,gs,wrappers,verbose=False,usemodel=False,P_U=0.1)
    fObsPosteriors2 = np.array(on.flatten(results2["POx"]))
    
    
    with open("posteriors/orig.txt",'w') as f:
        for i,frb in enumerate(results2["frbs"]):
            f.write(str(i)+"  "+frb+"  "+str(results2["dms"][i])[0:5]+" "+str(results2["PU"][i])[0:4]+"\n")
            for j,om in enumerate(results2["ObsMags"][i]):
                f.write(str(om)[0:5]+" "+ "%e" % results2["PO"][i][j]+" "+"%e" % results2["POx"][i][j]+"\n")
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
    results1 = on.calc_path_priors(frblist,ss,gs,wrappers,verbose=False)
    
    fObsPosteriors1 = np.array(on.flatten(results1["POx"]))
    plt.sca(ax1)
    plt.scatter(fObsPosteriors2,fObsPosteriors1,label="Marnoch23",marker='s')
    plt.sca(ax2)
    plt.scatter(fObsPosteriors2,fObsPosteriors1-fObsPosteriors2,label="Marnoch23",marker='s')
    
    
    with open("posteriors/marnoch.txt",'w') as f:
        for i,frb in enumerate(results1["frbs"]):
            f.write(str(i)+"  "+frb+"  "+str(results1["dms"][i])[0:5]+" "+str(results1["PU"][i])[0:4]+"\n")
            for j,om in enumerate(results1["ObsMags"][i]):
                f.write(str(om)[0:5]+" "+ "%e" % results1["PO"][i][j]+" "+"%e" % results1["POx"][i][j]+"\n")
            f.write("\n")
    
    
    ####### Model 2: Loudas ########
    
    model = opt.loudas_model()
    xbest = np.load("loudas_output/best_fit_params.npy")
    model.init_args(xbest) # best-fit arguments
    wrappers = on.make_wrappers(model,gs)
    results3 = on.calc_path_priors(frblist,ss,gs,wrappers,verbose=False)
    
    fObsPosteriors3 = np.array(on.flatten(results3["POx"]))
    plt.sca(ax1)
    plt.scatter(fObsPosteriors2,fObsPosteriors3,label="Loudas25",marker='x')
    plt.sca(ax2)
    plt.scatter(fObsPosteriors2,fObsPosteriors3-fObsPosteriors2,label="Loudas25",marker='x')
    
    with open("posteriors/loudas.txt",'w') as f:
        for i,frb in enumerate(results3["frbs"]):
            f.write(str(i)+"  "+frb+"  "+str(results3["dms"][i])[0:5]+" "+str(results3["PU"][i])[0:4]+"\n")
            for j,om in enumerate(results3["ObsMags"][i]):
                f.write(str(om)[0:5]+" "+ "%e" % results3["PO"][i][j]+" "+"%e" % results3["POx"][i][j]+"\n")
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
    
    # initialises best-fit arguments
    model.init_args(xbest)
    
    ############# Generate a KS-like plot showing the best fits ####################
    wrappers = on.make_wrappers(model,gs)
    results4 = on.calc_path_priors(frblist,ss,gs,wrappers,verbose=False)
    
    fObsPosteriors4 = np.array(on.flatten(results4["POx"]))
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
        for i,frb in enumerate(results4["frbs"]):
            f.write(str(i)+"  "+frb+"  "+str(results4["dms"][i])[0:5]+" "+str(results4["PU"][i])[0:4]+"\n")
            for j,om in enumerate(results4["ObsMags"][i]):
                f.write(str(om)[0:5]+" "+ "%e" % results4["PO"][i][j]+" "+"%e" % results4["POx"][i][j]+"\n")
            f.write("\n")
    
    all_candidates = on.get_cand_properties(frblist)
    
    
    po1 = results1["PO"]
    po2 = results2["PO"]
    po3 = results3["PO"]
    po4 = results4["PO"]
    
    pox1 = results1["POx"]
    pox2 = results2["POx"]
    pox3 = results3["POx"]
    pox4 = results4["POx"]
    
    pu1 = results1["PU"]
    pu2 = results2["PU"]
    pu3 = results3["PU"]
    pu4 = results4["PU"]
    
    pux1 = results1["PUx"]
    pux2 = results2["PUx"]
    pux3 = results3["PUx"]
    pux4 = results4["PUx"]
    
    # now iterates through galaxies and writes relevant info
    for i in np.arange(results1["NFRB"]):
        string1="\multicolumn{5}{c|}{"+results1["frbs"][i]+"} & "
        string1 += f"{pu2[i]:.3f} &  {pux2[i]:.3f} & "
        string1 += f"{pu1[i]:.3f} &  {pux1[i]:.3f} & "
        string1 += f"{pu3[i]:.3f} &  {pux3[i]:.3f} & "
        string1 += f"{pu4[i]:.3f} &  {pux4[i]:.3f} \\\\ "
        print("\\hline")
        print(string1)
        print("\\hline")
        
        
        
        for j,mag in enumerate(results4["ObsMags"][i]):
            # check if we print this one at all
            if results1["POx"][i][j] < 1e-4 and results2["POx"][i][j] < 1e-4 \
                and results3["POx"][i][j] < 1e-4 and results4["POx"][i][j] < 1e-4:
                
                continue
            
            string2 = f"{all_candidates[i]['ra'][j]:.4f} & {all_candidates[i]['dec'][j]:.4f} &"
            string2 += f" {all_candidates[i]['separation'][j]:.2f} &"
            string2 += f" {all_candidates[i]['ang_size'][j]:.2f} & {all_candidates[i]['mag'][j]:.2f} &"
            
            
            
            
            string2 += f"{po2[i][j]:.3f} &  {pox2[i][j]:.3f} & "
            string2 += f"{po1[i][j]:.3f} &  {pox1[i][j]:.3f} & "
            string2 += f"{po3[i][j]:.3f} &  {pox3[i][j]:.3f} & "
            string2 += f"{po4[i][j]:.3f} &  {pox4[i][j]:.3f} \\\\ "
            print(string2)
    
    
    
    ######## Makes cumulative distribution KS-style plots
    
    # loads various loudas models
    model = opt.loudas_model()
    xbest = np.load("loudas_output/best_fit_params.npy")
    for f_sfr in [0,1,xbest]:
        x=[f_sfr]
        model.init_args(x)
        wrappers = on.make_wrappers(model,gs)
        results = on.calc_path_priors(frblist,ss,gs,wrappers,verbose=False)
        
        NFRBlist.append(results["NFRB"])
        AppMagslist.append(results["AppMags"])
        AppMagPriorslist.append(results["AppMagPriors"])
        ObsMagslist.append(results["ObsMags"])
        ObsPosteriorslist.append(results["POx"])
        PUpriorlist.append(results["PU"])
        PUobslist.append(results["PUx"])
        sumPUpriorlist.append(results["sumPU"])
        sumPUobslist.append(results["sumPUx"])
    
    # loads naive model
    NFRBlist.append(results4["NFRB"])
    AppMagslist.append(results4["AppMags"])
    AppMagPriorslist.append(results4["AppMagPriors"])
    ObsMagslist.append(results4["ObsMags"])
    ObsPosteriorslist.append(results4["POx"])
    PUpriorlist.append(results4["PU"])
    PUobslist.append(results4["PUx"])
    sumPUpriorlist.append(results4["sumPU"])
    sumPUobslist.append(results4["sumPUx"])
    
    
    # loads Marnoch model
    NFRBlist.append(results1["NFRB"])
    AppMagslist.append(results1["AppMags"])
    AppMagPriorslist.append(results1["AppMagPriors"])
    ObsMagslist.append(results1["ObsMags"])
    ObsPosteriorslist.append(results1["POx"])
    PUpriorlist.append(results1["PU"])
    PUobslist.append(results1["PUx"])
    sumPUpriorlist.append(results1["sumPU"])
    sumPUobslist.append(results1["sumPUx"])
    
    
    plotlabels=["Loudas25: $f_{\\rm sfr} = 0$", "                  $f_{\\rm sfr} = 1$",
                     "                  $f_{\\rm sfr} = 3$","Naive","Marnoch23"]
    plotfile="Plots/all_cumulative.png"
    NMODELS=5
    on.make_cumulative_plots(NMODELS,NFRBlist,AppMagslist,AppMagPriorslist,ObsMagslist,ObsPosteriorslist,
                            PUobslist,PUpriorlist,plotfile,plotlabels,POxcut=None,onlyobs=0,addpriorlabel=False)
    
main()
