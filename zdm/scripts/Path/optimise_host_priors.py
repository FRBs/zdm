"""
Optimise FRB host galaxy magnitude priors using zdm predictions and PATH.

This script fits a parametric model of FRB host galaxy absolute magnitude
distributions to the CRAFT ICS optical observations. It works by:

1. Initialising zdm grids for the three CRAFT ICS survey bands (892, 1300,
   and 1632 MHz) using the HoffmannHalo25 parameter state.
2. Constructing a host galaxy model (``simple`` or ``loudas``) that predicts
   apparent r-band magnitudes by convolving the absolute magnitude distribution
   with the zdm p(z|DM_EG) redshift prior, optionally including a k-correction.
3. Running PATH with those zdm-derived apparent magnitude priors to obtain
   posterior host association probabilities P_Ox for each CRAFT ICS FRB.
4. Optimising the model parameters with ``scipy.optimize.minimize`` by
   minimising either a maximum-likelihood statistic or a KS-like goodness-of-fit
   statistic against the observed PATH posteriors.

After optimisation the script:

- Saves the best-fit parameters to ``<modelname>_output/best_fit_params.npy``.
- Plots the predicted vs observed apparent magnitude distributions for the
  best-fit model (``best_fit_apparent_magnitudes.png``).
- Re-runs PATH with the original (flat) priors for comparison and produces a
  scatter plot of best-fit vs original posteriors
  (``Scatter_plot_comparison.png``).

Limitations
-----------
- The optimal approach would sample galaxy candidates from a real photometric
  catalogue to construct proper optical fields; this script uses a parametric
  model instead.
- Host identification posteriors (P_Ox) are not fed back into the zdm
  likelihood; a self-consistent joint fit is not performed.
- Runtime can be significant when optimising the ``simple`` model (10 free
  parameters by default).

Usage
-----
Set ``minimise = True`` (default) to run the optimiser, or ``False`` to load
previously saved parameters from ``<modelname>_output/best_fit_params.npy``.
Switch between host models by changing ``modelname`` to ``"simple"`` or
``"loudas"``.

Requirements
------------
- ``astropath`` package (PATH implementation)
- ``frb`` package (FRB utilities and optical data)
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
    Optimise host galaxy magnitude model parameters and compare with baseline PATH.

    Workflow:

    1. Load the CRAFT ICS FRB list and initialise zdm grids for the 892, 1300,
       and 1632 MHz survey bands using the HoffmannHalo25 cosmological/FRB state.
    2. Select a host magnitude model (``"simple"`` or ``"loudas"``) and configure
       its parameter bounds and initial values.
    3. If ``minimise=True``, call ``scipy.optimize.minimize`` with
       ``on.function`` as the objective, minimising either the maximum-likelihood
       statistic (``istat=1``) or the KS-like statistic (``istat=0``) over all
       CRAFT ICS FRBs. Best-fit parameters are saved to
       ``<modelname>_output/best_fit_params.npy``.
    4. Re-evaluate PATH at the best-fit parameters and compute both the
       likelihood and KS statistics; save the apparent magnitude comparison
       plot to ``<modelname>_output/best_fit_apparent_magnitudes.png``.
    5. Re-run PATH with the original flat priors (``usemodel=False``) and save
       a scatter plot comparing original vs best-fit P_Ox posteriors to
       ``<modelname>_output/Scatter_plot_comparison.png``.

    Configuration knobs (edit at the top of the function body):

    - ``istat``: 0 = KS statistic, 1 = maximum-likelihood statistic.
    - ``dok``: whether to include a k-correction in the apparent magnitude model.
    - ``modelname``: ``"simple"`` for the parametric histogram model or
      ``"loudas"`` for the Loudas single-parameter model.
    - ``POxcut``: optional float (e.g. 0.9) to exclude low-confidence FRBs
      from the model comparison.
    - ``minimise``: set to ``False`` to skip optimisation and load saved
      parameters instead.
    """
    
    ######### List of all ICS FRBs for which we can run PATH #######
    # hard-coded list of FRBs with PATH data in ICE paper
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
    

def make_cdf_for_plotting(xvals, weights=None):
    """
    Build a step-function CDF suitable for plotting.

    Converts an array of data values (and optional weights) into paired
    (x, y) arrays that trace the cumulative distribution as a staircase,
    with two points per input value so that horizontal steps are rendered
    correctly by matplotlib.

    Parameters
    ----------
    xvals : np.ndarray
        1-D array of data values. Will be sorted in ascending order.
    weights : np.ndarray, optional
        1-D array of weights with the same length as ``xvals``. If provided,
        the CDF is computed as the normalised cumulative sum of the sorted
        weights. If ``None``, a uniform CDF over ``N`` points is used,
        with steps at ``0, 1/N, 2/N, ..., 1``.

    Returns
    -------
    cx : np.ndarray
        x-coordinates of the staircase CDF (length ``2 * N``).
    cy : np.ndarray
        y-coordinates of the staircase CDF (length ``2 * N``).
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
