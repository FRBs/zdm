"""
This file fits the simple (naive) model to CRAFT ICDS observations.
It fits a model of absolute galaxy magnitude distributions,
uses zDM to predict redshifts and hence apparent magntidues,
runs PATH using that prior, and tries to get priors to match posteriors.

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
    frblist=opt.frblist
    frblist.remove('FRB20230731A') # too reddened
    
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
    
    
    # initialise figure
    plt.figure()
    plt.xlabel("$P(O_i| \\mathbf{x})$ [original; $P(U)=0.1$]")
    plt.ylabel("$P(O_i| \\mathbf{x},N_O)$ [this work]")
    
    
    ##### Begins by calculating the original PATH result #####
    # calculates the original PATH result
    wrappers = None
    NFRB2,AppMags2,AppMagPriors2,ObsMags2,ObsPrior2,ObsPosteriors2,PUprior2,PUobs2,sumPUprior2,sumPUobs2,frbs,dms = \
                                            on.calc_path_priors(frblist,ss,gs,wrappers,verbose=False,usemodel=False,P_U=0.1)
    fObsPosteriors2 = on.flatten(ObsPosteriors2)
    
    
    with open("posteriors/orig.txt",'w') as f:
        for i,frb in enumerate(frbs):
            f.write(str(i)+"  "+frb+"  "+str(dms[i])[0:5]+" "+str(PUprior2[i])[0:4]+"\n")
            for j,om in enumerate(ObsMags2[i]):
                f.write(str(om)[0:5]+" "+ "%e" % ObsPrior2[i][j]+" "+"%e" % ObsPosteriors2[i][j]+"\n")
            f.write("\n")
    
    
    ####### Model 1: Marnoch ########
    
    # model 1: Marnoch
    model = opt.marnoch_model()
    
    wrappers = on.make_wrappers(model,gs)
    NFRB1,AppMags1,AppMagPriors1,ObsMags1,ObsPrior1,ObsPosteriors1,PUprior1,PUobs1,sumPUprior1,sumPUobs1,frbs,dms = on.calc_path_priors(frblist,ss,gs,wrappers,verbose=False)
    
    fObsPosteriors1 = on.flatten(ObsPosteriors1)
    plt.scatter(fObsPosteriors2,fObsPosteriors1,label="Marnoch",marker='s')
    
    
    with open("posteriors/marnoch.txt",'w') as f:
        for i,frb in enumerate(frbs):
            f.write(str(i)+"  "+frb+"  "+str(dms[i])[0:5]+" "+str(PUprior1[i])[0:4]+"\n")
            for j,om in enumerate(ObsMags1[i]):
                f.write(str(om)[0:5]+" "+ "%e" % ObsPrior1[i][j]+" "+"%e" % ObsPosteriors1[i][j]+"\n")
            f.write("\n")
    
    ####### Model 2: Loudas ########
    
    model = opt.loudas_model()
    model.init_args([1.49]) # best-fit arguments
    wrappers = on.make_wrappers(model,gs)
    NFRB3,AppMags3,AppMagPriors3,ObsMags3,ObsPrior3,ObsPosteriors3,PUprior3,PUobs3,sumPUprior3,sumPUobs3,frbs,dms = on.calc_path_priors(frblist,ss,gs,wrappers,verbose=False)
    
    fObsPosteriors3 = on.flatten(ObsPosteriors3)
    plt.scatter(fObsPosteriors2,fObsPosteriors3,label="Loudas",marker='x')
    
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
    opstate
    model = opt.simple_host_model(opstate)
    
    # retrieve default initial arguments in vector form
    x = [-2.28795519,  0.,  0. , 0. , 0.11907231,0.84640048,0.99813815 , 0., 0. , 0. , 0. ]
    
    # initialises best-fit arguments
    model.init_args(x)
    
    ############# Generate a KS-like plot showing the best fits ####################
    wrappers = on.make_wrappers(model,gs)
    NFRB4,AppMags4,AppMagPriors4,ObsMags4,ObsPrior4,ObsPosteriors4,PUprior4,PUobs4,sumPUprior4,sumPUobs4,frbs,dms = on.calc_path_priors(frblist,ss,gs,wrappers,verbose=False)
    
    fObsPosteriors4 = on.flatten(ObsPosteriors4)
    plt.scatter(fObsPosteriors2,fObsPosteriors4,label="Naive",marker='o',s=20)
    
    plt.legend(loc="lower right")
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.plot([0,1],[0,1],color="black",linestyle=":")
    plt.tight_layout()
    plt.savefig("posteriors/pox_comparison.png")
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
            
            string2 = f"{all_candidates[i]['ra'][j]:.4f} & ${all_candidates[i]['dec'][j]:.4f}$ &"
            string2 += f" {all_candidates[i]['separation'][j]:.2f} &"
            string2 += f" {all_candidates[i]['ang_size'][j]:.2f} & {all_candidates[i]['mag'][j]:.2f} &"
            
            
            string2 += f"{ObsPrior2[i][j]:.3f} &  {ObsPosteriors2[i][j]:.3f} & "
            string2 += f"{ObsPrior1[i][j]:.3f} &  {ObsPosteriors1[i][j]:.3f} & "
            string2 += f"{ObsPrior3[i][j]:.3f} &  {ObsPosteriors3[i][j]:.3f} & "
            string2 += f"{ObsPrior4[i][j]:.3f} &  {ObsPosteriors4[i][j]:.3f} \\\\ "
            print(string2)
main()
