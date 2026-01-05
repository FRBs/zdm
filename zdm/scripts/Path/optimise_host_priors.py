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

WARNING3: this program takes O~1 hr to run
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

# other FRB library imports
import astropath.priors as pathpriors


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
    state = parameters.State()
    cos.set_cosmology(state)
    cos.init_dist_measures()
    names=['CRAFT_ICS_892','CRAFT_ICS_1300','CRAFT_ICS_1632']
    ss,gs = loading.surveys_and_grids(survey_names=names)
    
    modelname = "loudas"
    opdir = modelname+"_output/"
    if not os.path.exists(opdir):
        os.mkdir(opdir)
    
    # Case of simple host model
    # Initialisation of model
    # simple host model
    if modelname=="simple":
        model = opt.simple_host_model()
        x0 = model.AbsPrior
    elif modelname=="loudas":
        #### case of Loudas model
        model = opt.loudas_model()
        x0 = [0.5]
    else:
        print("Unrecognised host model ", modelname)
    
    
    # initialise aguments to minimisation function
    args=[frblist,ss,gs,model]
    Nparams = len(x0)
    bounds = [(0,1)]*Nparams
    
    # "function" is the function that performs the comparison of
    # predictions to outcomes. It's where all the magic happens
    result = minimize(on.function,x0 = x0,args=args,bounds = bounds)
    
    # Recording the current spline best-fit here
    #x = [0.00000000e+00 0.00000000e+00 7.05155614e-02 8.39235326e-01
    #    3.27794398e-01 1.00182186e-03 0.00000000e+00 3.46702511e-04
    #    2.17040011e-03 9.72472750e-04]
    
    # recording the current non-spline best fit here
    #x = [ 1.707e-04,  8.649e-02,  9.365e-01,  9.996e-01,  2.255e-01,\
    #         3.493e-02,  0.000e+00,  0.000e+00,  0.000e+00,  1.000e-01]
    #x = np.array(x)
    
    print("Best fit result is ",result.x)
    x = result.x
    
    # analyses final result
    if modelname == "simple":
        # renormalise distribution in parameters
        x /= np.sum(x)
    
    # initialises arguments
    model.init_args(x)
    
    outfile = opdir+"best_fit_apparent_magnitudes.png"
    wrappers = on.make_wrappers(model,gs)
    NFRB,AppMags,AppMagPriors,ObsMags,ObsPosteriors,PUprior,PUobs,sumPUprior,sumPUobs = on.calc_path_priors(frblist,ss,gs,wrappers,verbose=False)
    stat = on.calculate_goodness_statistic(NFRB,AppMags,AppMagPriors,ObsMags,ObsPosteriors,sumPUobs,sumPUprior,plotfile=outfile)
    
    # calculates the original PATH result
    outfile = opdir+"original_fit_apparent_magnitudes.png"
    NFRB2,AppMags2,AppMagPriors2,ObsMags2,ObsPosteriors2,PUprior2,PUobs2,sumPUprior2,sumPUobs2 = on.calc_path_priors(frblist,ss,gs,wrappers,verbose=False,usemodel=False)
    stat = on.calculate_goodness_statistic(NFRB,AppMags,AppMagPriors,ObsMags,ObsPosteriors,sumPUobs,sumPUprior,plotfile=outfile)
    
    
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
    
    if modelname == "simple":
        # plots final result on absolute magnitudes
        plt.figure()
        plt.xlabel("Absolute magnitude, $M_r$")
        plt.ylabel("$p(M_r)$")
        plt.plot(model.AbsMags,model.AbsMagWeights/np.max(model.AbsMagWeights),label="interpolation")
        plt.plot(model.ModelBins,x/np.max(x),marker="o",linestyle="",label="Model Parameters")
        plt.legend()
        plt.tight_layout()
        plt.savefig(opdir+"best_fit_absolute_magnitudes.pdf")
        plt.close()
        
    if modelname == "loudas":
        
        NSFR=41
        stats = np.zeros([NSFR])
        SFRs = np.linspace(0,4,NSFR)
        for istat,sfr in enumerate(SFRs):
            outfile = opdir+"ks_test_sfr_"+str(sfr)+".png"
            #outfile = None
            model.init_args(sfr)
            wrappers = on.make_wrappers(model,gs)
            NFRB,AppMags,AppMagPriors,ObsMags,ObsPosteriors,PUprior,PUobs,sumPUprior,sumPUobs = on.calc_path_priors(frblist,ss,gs,wrappers,verbose=False)
            stat = on.calculate_goodness_statistic(NFRB,AppMags,AppMagPriors,ObsMags,ObsPosteriors,sumPUobs,sumPUprior,plotfile=outfile)
            stats[istat] = stat
        outfile = opdir+"scan_sfr.png"
        plt.figure()
        plt.plot(SFRs,stats,marker="o")
        plt.xlabel("$f_{\\rm sfr}$")
        plt.ylabel("ks statistic (lower is better)")
        plt.tight_layout()
        plt.savefig(outfile)
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
