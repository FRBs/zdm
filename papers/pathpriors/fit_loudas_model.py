"""
This file fits and generates plots for the model of Loudas et al
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
    
    
    ######## Determnine which statistic to use in optimisation ########
    # setting istat=0 means using a ks statistic to fit p(m_r)
    # setting istat=1 means using a maximum likelihood estimator
    istat=1
    # determines which model to use
    modelname = "loudas"
    
    opdir = modelname+"_output/"
    POxcut = None # set to e.g. 0.9 to reject FRBs with lower posteriors when doing model comparisons
    
    if not os.path.exists(opdir):
        os.mkdir(opdir)
    
    model = opt.loudas_model()
    # set f_sfr starting value to 0.5
    x0 = [0.5]
    bounds=[(-3,3)] # large range - physical region is 0 to 1
    
    # initialise aguments to minimisation function
    args=[frblist,ss,gs,model,POxcut,istat]
    
    # turn minimise on to re-perform the minimusation. But it's already been done
    minimise=False
    if minimise:
        result = minimize(on.function,x0 = x0,args=args,bounds = bounds)
        print("Best fit result is f_sfr = ",result.x)
        x = result.x
        # saves result
        np.save(opdir+"/best_fit_params.npy",x)
    else:
        # replace later
        x=[1.68]
        print("using previous result of f_sfr = ",x)
    
    # initialises arguments
    model.init_args(x)
    
    outfile = opdir+"best_fit_apparent_magnitudes.png"
    wrappers = on.make_wrappers(model,gs)
    NFRB,AppMags,AppMagPriors,ObsMags,ObsPosteriors,PUprior,PUobs,sumPUprior,sumPUobs = on.calc_path_priors(frblist,ss,gs,wrappers,verbose=False)
    llstat = on.calculate_likelihood_statistic(NFRB,AppMags,AppMagPriors,ObsMags,ObsPosteriors,PUobs,PUprior,plotfile=outfile)
    
    ksstat = on.calculate_ks_statistic(NFRB,AppMags,AppMagPriors,ObsMags,ObsPosteriors,sumPUobs,sumPUprior,plotfile=outfile)
    
    print("Best-fit stats of the Loudas model are ll=",llstat," ks = ",ksstat)
    
    # makes a ks stat plot comparing three scenarios - SFR, M*, and best fit
    
    NFRBlist = []
    AppMagslist = []
    AppMagPriorslist = []
    ObsMagslist = []
    ObsPosteriorslist = []
    PUpriorlist = []
    PUobslist = []
    sumPUpriorlist = []
    sumPUobslist = []
    
    for f_sfr in [1.68,0,1]:
        x=[f_sfr]
        model.init_args(x)
        wrappers = on.make_wrappers(model,gs)
        NFRB,AppMags,AppMagPriors,ObsMags,ObsPosteriors,PUprior,PUobs,sumPUprior,sumPUobs = on.calc_path_priors(frblist,ss,gs,wrappers,verbose=False)
        
        NFRBlist.append(NFRB)
        AppMagslist.append(AppMags)
        AppMagPriorslist.append(AppMagPriors)
        ObsMagslist.append(ObsMags)
        ObsPosteriorslist.append(ObsPosteriors)
        PUpriorlist.append(PUprior)
        PUobslist.append(PUobs)
        sumPUpriorlist.append(sumPUprior)
        sumPUobslist.append(sumPUobs)
    
    NMODELS = 3
    plotlabels=["$f_{\\rm sfr} = 1.68$","$f_{\\rm sfr} = 0$", "$f_{\\rm sfr} = 1$"]
    plotfile = opdir+"loudas_f0_1_best_comparison.png"
    on.make_cumulative_plots(NMODELS,NFRBlist,AppMagslist,AppMagPriorslist,ObsMagslist,ObsPosteriorslist,
                            PUobslist,PUpriorlist,plotfile,plotlabels,POxcut=None,onlyobs=0,abc="(b)")
    
    
    NSFR=31
    stats = np.zeros([NSFR])
    SFRs = np.linspace(0,3,NSFR)
    for istat,sfr in enumerate(SFRs):
        
        model.init_args([sfr])
        wrappers = on.make_wrappers(model,gs)
        NFRB,AppMags,AppMagPriors,ObsMags,ObsPosteriors,PUprior,\
        PUobs,sumPUprior,sumPUobs = on.calc_path_priors(frblist,ss,gs,wrappers,verbose=False)
        stat = on.calculate_likelihood_statistic(NFRB,AppMags,AppMagPriors,ObsMags,ObsPosteriors,PUobs,
                        PUprior,plotfile=outfile,POxcut=POxcut)
        stats[istat] = stat
    outfile = opdir+"scan_sfr.png"
    plt.figure()
    plt.plot(SFRs,stats,marker="o")
    plt.xlabel("$f_{\\rm sfr}$")
    plt.ylabel("$\\log_{10} \\mathcal{L}(f_{\\rm sfr})$")
    plt.xlim(0,3)
    plt.ylim(46,52)
    plt.tight_layout()
    plt.savefig(outfile)
    plt.close()



main()
