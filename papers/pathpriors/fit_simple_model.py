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
    
    # initialise cosmology
    cos.set_cosmology(state)
    cos.init_dist_measures()
    
    names=['CRAFT_ICS_892','CRAFT_ICS_1300','CRAFT_ICS_1632']
    ss,gs = loading.surveys_and_grids(survey_names=names,init_state=state)
    
    
    ######## Determnine which statistic to use in optimisation ########
    # setting istat=0 means using a ks statistic to fit p(m_r)
    # setting istat=1 means using a maximum likelihood estimator
    istat=1
    # dok=True means use the k-correction
    dok = True
    # we are using the simple model
    modelname = "simple"
    # set to e.g. 0.9 to reject FRBs with lower posteriors when doing model comparisons
    POxcut = None
    
    opdir = modelname+"_output/"
    
    
    if not os.path.exists(opdir):
        os.mkdir(opdir)
    
    # Case of simple host model
    # Initialisation of model
    # simple host model
    opstate = op.OpticalState()
    # sets optical state to use simple linear interpolation
    opstate.simple.AbsModelID = 1
    
    # sets up initial bounds on variables
    if dok:
        opstate.simple.AppModelID = 1 # k-correction
        Nparams = opstate.simple.NModelBins+1
        opstate.simple.AppModelID = 1 # sets to include k-correction
        opstate.simple.k = 0.5 # for some reason, this just doesn't make much difference to results
        bounds = [(-25,25)]+[(0,1)]*(Nparams-1)
    else:
        Nparams = opstate.simple.NModelBins
        # bins now give log-space values, hence -5,2 is range of 10^7
        if opstate.simple.AbsModelID == 3:
            base=(-5,2) # log space
        else:
            base=(0,1) # linear space
        bounds = [base]*(Nparams)
        opstate.simple.AppModelID = 0 # no k-correction
        
    model = opt.simple_host_model(opstate)
    
    # retrieve default initial arguments in vector form
    x0 = model.get_args()
    
    # initialise aguments to minimisation function
    args=[frblist,ss,gs,model,POxcut,istat]
    
    # set to false to just use hard-coded best fit parameters
    minimise=False
    if minimise:
        result = minimize(on.function,x0 = x0,args=args,bounds = bounds)
        print("Best fit result is ",result.x)
        x = result.x
        # saves result
        np.save(opdir+"/best_fit_params.npy",x)
    else:
        # hard-coded best fit parameters from running optimisation
        if not dok:
            print("Need to re-run this!")
            exit()
        else:
            x = [-2.28795519,0.,0.,0.,0.11907231,0.84640048,0.99813815,0.,0.,0.,0.]
    # initialises best-fit arguments
    model.init_args(x)
    
    ############# Generate a KS-like plot showing the best fits ####################
    outfile = opdir+"simple_best_fit_apparent_magnitudes.png"
    wrappers = on.make_wrappers(model,gs)
    NFRB,AppMags,AppMagPriors,ObsMags,ObsPriors,ObsPosteriors,PUprior,PUobs,sumPUprior,sumPUobs,frbs,dms = on.calc_path_priors(frblist,ss,gs,wrappers,verbose=False)
    
    llstat = on.calculate_likelihood_statistic(NFRB,AppMags,AppMagPriors,ObsMags,ObsPosteriors,PUobs,PUprior,plotfile=outfile)
    ksstat = on.calculate_ks_statistic(NFRB,AppMags,AppMagPriors,ObsMags,ObsPosteriors,sumPUobs,
                                    sumPUprior,plotfile=outfile,abc="(c)",tag="naive: ",)
    
    print("Best-fit stats of the naive model are ll=",llstat," ks = ",ksstat)
    
    
    ############ k-correction figure ############3
    # we generate a plot showing the convergence on k, i.e. how/why we get a best fit
    nk=21
    kvals = np.linspace(-5,5,nk)
    stats = np.zeros([nk])
    for i,kcorr in enumerate(kvals):
        x[0] = kcorr
        model.init_args(x)
        wrappers = on.make_wrappers(model,gs)
        NFRB,AppMags,AppMagPriors,ObsMags,ObsPriors,ObsPosteriors,PUprior,PUobs,sumPUprior,sumPUobs,frbs,dms = on.calc_path_priors(frblist,ss,gs,wrappers,verbose=False)
        stat = on.calculate_likelihood_statistic(NFRB,AppMags,AppMagPriors,ObsMags,ObsPosteriors,PUobs,PUprior)
        stats[i] = stat
    
    plt.figure()
    plt.plot(kvals,stats)
    #plt.yscale('log')
    plt.xlabel('$k$')
    plt.ylabel('$\\log_{10} \\mathcal{L} (k)$')
    plt.tight_layout()
    plt.savefig(opdir+'/pkvalue.png')
    plt.close()
    
    
    # calculates the original PATH result
    outfile = opdir+"original_fit_apparent_magnitudes.png"
    NFRB2,AppMags2,AppMagPriors2,ObsMags2,ObsPriors2,ObsPosteriors2,PUprior2,PUobs2,sumPUprior2,sumPUobs2,frbs,dms = on.calc_path_priors(frblist,ss,gs,wrappers,verbose=False,usemodel=False)
    
    # flattens lists of lists
    ObsPosteriors = [x for xs in ObsPosteriors for x in xs]
    ObsPosteriors2 = [x for xs in ObsPosteriors2 for x in xs]
    
    # plots original vs updated posteriors
    plt.figure()
    plt.xlabel("Original P")
    plt.ylabel("Best fit P")
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.scatter(ObsPosteriors2,ObsPosteriors,label="Hosts",marker='x')
    plt.scatter(PUobs2,PUobs,label="Unobserved",marker='+')
    plt.legend()
    plt.tight_layout()
    plt.savefig(opdir+"Scatter_plot_comparison.png")
    plt.close()
    
    ####### Plots that only make sense for specific models ########3
    
    plt.figure()
    plt.xlabel("Absolute magnitude, $M_r$")
    plt.ylabel("$p(M_r)$")
    plt.plot(model.AbsMags,model.AbsMagWeights/np.max(model.AbsMagWeights),label="interpolation")
    
    if dok:
        toplot = x[1:]
    else:
        toplot = x
    
    if model.AbsModelID == 3:
        toplot = 10**toplot
    plt.plot(model.ModelBins,toplot/np.max(toplot),marker="o",linestyle="",label="Model Parameters")
    plt.legend()
    plt.tight_layout()
    plt.savefig(opdir+"best_fit_absolute_magnitudes.png")
    plt.close()

    


main()
