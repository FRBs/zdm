"""
This file illustrates how to optimise the host prior
distribution by fitting to CRAFT ICS optical observations.
It fits a model of absolute galaxy magnitude distributions,
uses zDM to predict redshifts and hence apparent magntidues,
runs PATH using that prior, and tries to get priors to match posteriors.

WARNING: this is NOT the optimal method! That would require using
a catalogue of galaxies to sample from to generate fake optical fields.
But nonetheless, this tests the power of estimating FRB host galaxy
contributions using zDM to set priors for apparent magnitudes.

WARNING2: To do this properly also requires inputting the posterior POx
for host galaxies into zDM! This simulation does not do that either.

WARNING3: this program can take a while to run, if optimising the simple model.
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
    
    cos.set_cosmology(state)
    cos.init_dist_measures()
    
    # loads zDM grids
    names=['CRAFT_ICS_892','CRAFT_ICS_1300','CRAFT_ICS_1632']
    ss,gs = loading.surveys_and_grids(survey_names=names,init_state=state)
    
    
    ######## Determnine which statistic to use in optimisation ########
    # setting istat=0 means using a ks statistic to fit p(m_r)
    # setting istat=1 means using a maximum likelihood estimator
    istat=1
    dok = True # turn on k-correction or not
    
    # determines which model to use
    #modelname = "loudas"
    modelname = "simple"
    
    opdir = modelname+"_output/"
    POxcut = None # set to e.g. 0.9 to reject FRBs with lower posteriors when doing model comparisons
    
    if not os.path.exists(opdir):
        os.mkdir(opdir)
    
    # Case of simple host model
    # Initialisation of model
    # simple host model
    if modelname=="simple":
        opstate = op.OpticalState()
        
        if dok:
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
        x0 = model.get_args()
        
        
    elif modelname=="loudas":
        #### case of Loudas model
        model = opt.loudas_model()
        x0 = [0.5]
        bounds=[(-3,3)] # large range
    else:
        print("Unrecognised host model ", modelname)
    
    
    # initialise aguments to minimisation function
    args=[frblist,ss,gs,model,POxcut,istat]
    
    # "function" is the function that performs the comparison of
    # predictions to outcomes. It's where all the magic happens
    
    minimise=True
    if minimise:
        result = minimize(on.function,x0 = x0,args=args,bounds = bounds)
        print("Best fit result is ",result.x)
        x = result.x
        # saves result
        np.save(opdir+"/best_fit_params.npy",x)
    else:
        x = np.load(opdir+"/best_fit_params.npy")
    
    # initialises arguments
    model.init_args(x)
    
    outfile = opdir+"best_fit_apparent_magnitudes.png"
    wrappers = on.make_wrappers(model,gs)
    NFRB,AppMags,AppMagPriors,ObsMags,ObsPriors,ObsPosteriors,PUprior,PUobs,sumPUprior,sumPUobs,frbs,dms = on.calc_path_priors(frblist,ss,gs,wrappers,verbose=False)
    
    # calculates a maximum-likelihood statistic
    stat = on.calculate_likelihood_statistic(NFRB,AppMags,AppMagPriors,ObsMags,ObsPosteriors,PUobs,PUprior,plotfile=outfile)
    
    # calculates a KS-like statistic
    stat = on.calculate_ks_statistic(NFRB,AppMags,AppMagPriors,ObsMags,ObsPosteriors,sumPUobs,sumPUprior,plotfile=outfile)
    
    
    # calculates the original PATH result
    outfile = opdir+"original_fit_apparent_magnitudes.png"
    NFRB2,AppMags2,AppMagPriors2,ObsMags2,ObsPriors2,ObsPosteriors2,PUprior2,PUobs2,sumPUprior2,sumPUobs2,frbs,dms = on.calc_path_priors(frblist,ss,gs,wrappers,verbose=False,usemodel=False)
    # currently, calculating KS statistics does not work/make sense for original path. need to re-think this
    #stat = on.calculate_ks_statistic(NFRB,AppMags,AppMagPriors2,ObsMags2,ObsPosteriors2,sumPUobs2,sumPUprior2,plotfile=outfile)
    
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
    

def make_cdf_for_plotting(xvals,weights=None):
    """
    Creates a cumulative distribution function
    
    xvals,yvals: values of data points
    """
    N = xvals.size
    cx = np.zeros([2*N])
    cy = np.zeros([2*N])
    
    order = np.argsort(xvals)
    xvals = xvals[order]
    if weights is None:
        weights = np.linspace(0.,1.,N+1)
    else:
        weights = weights[order]
        weights = np.cumsum(weights)
        weights /= weights[-1]
        
    for i,x in enumerate(xvals):
        cx[2*i] = x
        cx[2*i+1] = x
        cy[2*i] = weights[i]
        cy[2*i+1] = weights[i+1]
    return cx,cy


    


main()
