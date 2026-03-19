"""
Plot and compare FRB host galaxy magnitude models against CRAFT ICS PATH posteriors.

This script demonstrates how to load, configure, and visualise the three
available FRB host galaxy magnitude models, then evaluate PATH host association
posteriors for all CRAFT ICS FRBs using each model in turn.

Model comparison
----------------
Three host magnitude models are loaded and plotted:

- **Simple model** (``opt.simple_host_model``): a parametric histogram of
  absolute magnitudes M_r, linearly interpolated, with an optional
  k-correction.  Parameters are set by hand here (not from a fitted result).
- **Loudas model** (``opt.loudas_model``): predicts apparent magnitudes from
  a galaxy luminosity function weighted by stellar mass (``fSFR=0``) or
  star-formation rate (``fSFR=1``), based on Nick Loudas's galaxy model.
  The mixing parameter ``fSFR`` is varied to show sensitivity.
- **Marnoch model** (``opt.marnoch_model``): predicts apparent magnitudes
  from the galaxy spectral evolution model of Marnoch et al. 2023
  (MNRAS 525, 994).

Diagnostic plots produced in ``Plots/``
-----------------------------------------
- ``simple_model_mags.png``: absolute magnitude prior p(M_r) for the simple
  model, showing the interpolated curve and the raw histogram bin values.
- ``loudas_model_mags.png``: apparent magnitude distributions p(m_r) for the
  Loudas model at several redshifts, comparing mass- vs SFR-weighted variants.
- ``loudas_fsfr_interpolation.png``: sensitivity of the Loudas model to the
  ``fSFR`` mixing parameter at z=0.5, illustrating the full allowed range.
- ``all_model_apparent_mags.png``: side-by-side comparison of p(m_r | z) for
  all three models at z = 0.1, 0.5, and 2.0.
- ``all_model_mag_priors_dm.png``: PATH apparent magnitude priors p(m_r | DM)
  for all three models at DM = 200, 600, and 1000 pc/cm³, using the
  CRAFT_ICS_1300 zdm grid to convert DM to a redshift prior.
- ``posterior_comparison.png``: scatter plot of PATH host posteriors P(O|x)
  from the original flat-prior run vs each of the four zdm-informed model
  runs, across all CRAFT ICS FRBs.

Note
----
Parameter values used here are illustrative initial estimates, not best-fit
results from the published analysis. See ``optimise_host_priors.py`` for the
fitting procedure.

Requirements
------------
- ``astropath`` package (PATH implementation)
- ``frb`` package (FRB utilities and optical data)
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
    Generate diagnostic plots for all host models and compare PATH posteriors.

    Workflow:

    1. **Model initialisation**: Loads the simple, Loudas (mass-weighted,
       ``fSFR=0``), and Marnoch host magnitude models with illustrative
       parameter values.

    2. **Intrinsic magnitude plots**: Plots the absolute magnitude prior
       p(M_r) for the simple model and apparent magnitude distributions
       p(m_r) for the Loudas model at several redshifts.

    3. **fSFR sensitivity**: Plots p(m_r | z=0.5) for the Loudas model
       across a wide range of ``fSFR`` values to illustrate model behaviour
       beyond the physically motivated [0, 1] range.

    4. **Model comparison at fixed z**: Compares p(m_r | z) across all three
       models at z = 0.1, 0.5, and 2.0.

    5. **DM-dependent priors**: Loads the CRAFT_ICS_1300 zdm grid and wraps
       each model in a ``model_wrapper`` to compute the PATH apparent magnitude
       prior p(m_r | DM) at DM = 200, 600, and 1000 pc/cm³.

    6. **PATH evaluation over CRAFT ICS FRBs**: For each FRB in
       ``opt.frblist`` that is found in the CRAFT_ICS_1300 survey:

       - Runs PATH with a flat prior (``usemodel=False``, P_U=0.1) as the
         baseline.
       - Runs PATH four more times, one per zdm-informed model variant
         (simple; Loudas mass-weighted; Loudas SFR-weighted; Marnoch), each
         with P_U estimated from ``wrapper.estimate_unseen_prior()``.
       - Prints diagnostic output for candidate host galaxies that flip above
         P_Ox=0.5 relative to the baseline (simple model only).

    7. **Posterior scatter plot**: Produces ``Plots/posterior_comparison.png``
       showing P(O|x) from each zdm-informed model against the flat-prior
       baseline across all FRBs.

    Output files are written to the ``Plots/`` subdirectory, which is created
    if it does not already exist.
    """
    
    opdir = "Plots/"
    if not os.path.exists(opdir):
        os.mkdir(opdir)
    ##### performs the following calculations for the below combinations ######
    
    # loads a default optical state.
    opstate1 = op.OpticalState()
    #opstate1.SimpleParams.AbsPrior = [0,0,0.1,1,0.4,0,0,0,0,0] # this is from an initial estimate
    
    opstate2 = op.OpticalState()
    opstate2.loudas.fSFR=0.
    
    
    ######## initialises optical-independent info ########
    #frblist is a hard-coded list of FRBs for which we have optical PATH data
    frblist = opt.frblist
    NFRB = len(frblist)
    
    
    state = parameters.State()
    cos.set_cosmology(state)
    cos.init_dist_measures()
    
    
    plt.figure()
    
    opstate = op.SimpleParams()
    
    ##### makes a plot of host priors for the simple model ######
    
    # simple host model
    model1 = opt.simple_host_model(opstate1)
    # this is from an initial estimate. Currently, no way to enter this into the opstate. To do.
    absprior = [0,0,0.1,1,0.4,0,0,0,0,0]
    model1.init_args(absprior)
        
    plt.figure()
    plt.plot(model1.AbsMags,model1.AbsMagWeights/np.max(model1.AbsMagWeights),label="Histogram interpolation")
    plt.scatter(model1.ModelBins,model1.AbsPrior/np.max(model1.AbsPrior),label="Simple model points")
    plt.xlabel("Absolute magnitude $M_r$")
    plt.ylabel("$p(M_r)$")
    plt.legend()
    plt.tight_layout()
    plt.savefig(opdir+"simple_model_mags.png")
    plt.close() 
    
    ####### Makes plots for Nick Loudas' model #####
    
    model2=opt.loudas_model(opstate2)
    for i in np.arange(1,20,4):
        plt.plot(model2.rmags,model2.p_mr_mass[i],label="$M_\\odot$, z="+str(model2.zbins[i]),linestyle="-")
        plt.plot(model2.rmags,model2.p_mr_sfr[i],label="SFR, z="+str(model2.zbins[i]),linestyle="--",color=plt.gca().lines[-1].get_color())
    plt.xlabel("apparent magnitude $m_r$")
    plt.ylabel("$p(m_r)$")
    plt.legend()
    plt.tight_layout()
    plt.savefig(opdir+"loudas_model_mags.png")
    plt.close()
    
    
    ###### Gives examples for fsfr at weird values #####
    
    plt.figure()
    plt.xlabel("$m_r$")
    plt.ylabel("$P(m_r | z=0.5)$")
    styles=["-","--",":","-.","-","--",":","-."]
    fsfrs = np.linspace(-2,3,6) # extrapolates to weird values
    z=0.5
    mrbins = np.linspace(0,40,401)
    rbc = (mrbins[1:] + mrbins[:-1])/2.
    for i,fsfr in enumerate(fsfrs):
        model2.init_args(fsfr)
        pmr = model2.get_pmr_gz(mrbins,z)
        pmr /= np.sum(pmr)*(rbc[1]-rbc[0])
        plt.plot(rbc,pmr,label="$f_{\\rm sfr} = $"+str(fsfr)[:5],linestyle=styles[i])
    plt.xlim(15,30)
    plt.ylim(0.,0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(opdir+"loudas_fsfr_interpolation.png")
    plt.close()
    
    # set up basic histogram of p(mr) distribution
    mrbins = np.linspace(0,40,401)
    mrvals=(mrbins[:-1]+mrbins[1:])/2.
    dmr = mrbins[1]-mrbins[0]
    
    model3 = opt.marnoch_model()
    
    ######### Plots apparent mag distribution for all models as function of z #######
    styles=["-","--",":","-."]
    
    plt.figure()
    flist=[0,1]
    
    for i,z in enumerate([0.1,0.5,2.0]):
        
        # simple model
        pmr = model1.get_pmr_gz(mrbins,z)
        pmr /= np.sum(pmr)
        
        plt.plot(mrvals,pmr/dmr,label="z = "+str(z)+"; Naive",linestyle=styles[0])
        
        # Loudas model dependencies
        for i,fsfr in enumerate(flist):
            model2.init_args(fsfr)
            pmr = model2.get_pmr_gz(mrbins,z)
            pmr /= np.sum(pmr)
            plt.plot(mrvals,pmr/dmr,label = "              $f_{\\rm sfr}$ = "+str(fsfr),
                linestyle=styles[i+1],color=plt.gca().lines[-1].get_color())
        
        pmr = model3.get_pmr_gz(mrbins,z)
        plt.plot(mrvals,pmr/dmr,label = "              Marnoch",linestyle=styles[3],
            color=plt.gca().lines[-1].get_color())
        
    plt.xlabel("Apparent magnitude $m_r$")
    plt.ylabel("$p(m_r|z)$")
    plt.xlim(10,40)
    plt.ylim(0,0.35)
    plt.tight_layout()
    plt.legend()
    plt.savefig(opdir+"all_model_apparent_mags.png")
    plt.close()
    
    
    ############################################################################
    #Load a grid. We'll only load data from the ICS 1300 survey. Just to use it
    # as an example calculation for particular DMs
    ############################################################################
    name='CRAFT_ICS_1300'
    ss,gs = loading.surveys_and_grids(survey_names=[name])
    g = gs[0]
    s = ss[0]
    
    
    # wrapper around the optical model. For returning p(m_r|DM)
    wrapper1 = opt.model_wrapper(model1,g.zvals) # simple
    wrapper2 = opt.model_wrapper(model2,g.zvals) # loudas with fsfr=0
    wrapper3 = opt.model_wrapper(model3,g.zvals) # loudas with fsfr=0
    
    # simply illustrates how one might change the probabilities of
    # observaing a galaxy of a given magnitude
    wrapper1.pU_mean = 26.385
    wrapper1.pU_width = 0.279
    
    
    # do this once per "model" objects
    #pathpriors.USR_raw_prior_Oi = wrapper1.path_raw_prior_Oi
    
    
    # how do we change a parameter? We need to pass on the low-level model to the wrapper
    plt.figure()
    
    for i,DM in enumerate([200,600,1000]):
        
        wrapper1.init_path_raw_prior_Oi(DM,g)
        plt.plot(wrapper1.AppMags,wrapper1.priors,label="DM = "+str(DM)+", Simple",linestyle=styles[0])
        
        # this is how we change the parameters of a state
        # we first change the underlying state
        # then we initialise the model
        # then we re-init the wrapper.
        fSFR=0.
        model2.init_args(fSFR)
        wrapper2.init_zmapping(g.zvals)
        wrapper2.init_path_raw_prior_Oi(DM,g)
        plt.plot(wrapper2.AppMags,wrapper2.priors,label="DM = "+str(DM)+", Loudas: $f_{\\rm sfr}$ = 0.0",
            linestyle=styles[1],color=plt.gca().lines[-1].get_color())
        
        fSFR=1.0
        model2.init_args(fSFR)
        wrapper2.init_zmapping(g.zvals)
        wrapper2.init_path_raw_prior_Oi(DM,g)
        plt.plot(wrapper2.AppMags,wrapper2.priors,label="DM = "+str(DM)+", Loudas: $f_{\\rm sfr}$ = 1.0",
            linestyle=styles[2],color=plt.gca().lines[-1].get_color())
        
        wrapper3.init_zmapping(g.zvals)
        wrapper3.init_path_raw_prior_Oi(DM,g)
        plt.plot(wrapper3.AppMags,wrapper3.priors,label="DM = "+str(DM)+", Marnoch",
            linestyle=styles[2],color=plt.gca().lines[-1].get_color())
        
    plt.xlabel("Absolute magnitude $M_r$")
    plt.ylabel("$p(M_r)$")
    plt.legend()
    plt.tight_layout()
    plt.savefig(opdir+"all_model_mag_priors_dm.png")
    plt.close()
    
    # do this only for a particular FRB
    # it gives a prior on apparent magnitude and pz
    #AppMagPriors,pz = model.get_posterior(g,DMlist)
    
    
    maglist = [None,None,None,None,None]
    allPOx = [None,None,None,None,None]
    
    labels=["Orig","Simple","Mass-weighted","SFR weighted","Marnoch"]
    markers=["x","+","s","o","v"]
    
    for i,frb in enumerate(frblist):
        # interates over the FRBs. "Do FRB"
        # P_O is the prior for each galaxy
        # P_Ox is the posterior
        # P_Ux is the posterior for it being unobserved
        # mags is the list of galaxy magnitudes
        
        # determines if this FRB was seen by the survey, and
        # if so, what its DMEG is
        imatch = opt.matchFRB(frb,s)
        if imatch is None:
            print("Could not find ",frb," in survey")
            continue
        else:
            print("Found FRB ",frb)
        
        DMEG = s.DMEGs[imatch]
        
        # original calculation
        P_O1,P_Ox1,P_Ux1,mags1,ptbl = on.run_path(frb,usemodel=False,P_U=0.1)
        
        # record this info
        if maglist[0] is None:
            maglist[0] = mags1
            allPOx[0] = P_Ox1
        else:
            maglist[0] = np.append(maglist[0],mags1)
            allPOx[0] = np.append(allPOx[0],P_Ox1)
        
        
        # simple model
        wrapper1.init_path_raw_prior_Oi(DMEG,g)
        PU2 = wrapper1.estimate_unseen_prior() # might not be correct
        pathpriors.USR_raw_prior_Oi = wrapper1.path_raw_prior_Oi
        P_O2,P_Ox2,P_Ux2,mags2,ptbl = on.run_path(frb,usemodel=True,P_U = PU2)
        
        
        for imag,mag in enumerate(mags2):
            if P_Ox2[imag] > 0.5 and P_Ox1[imag] < 0.5:
                #print(i,frb,mag,P_Ox1[imag],P_Ox2[imag])
                print(frb,P_Ux1,PU2,DMEG)
                for k,x in enumerate(P_Ox1):
                    print(mags1[k],x,P_Ox2[k])
        
        # record this info
        if maglist[1] is None:
            maglist[1] = mags2
            allPOx[1] = P_Ox2
        else:
            maglist[1] = np.append(maglist[1],mags2)
            allPOx[1] = np.append(allPOx[1],P_Ox2)
        
        
        # loudas fsfr = 0.0 (i.e., mass weighted)
        fSFR=0.0
        model2.init_args(fSFR)
        wrapper2.init_zmapping(g.zvals)
        wrapper2.init_path_raw_prior_Oi(DMEG,g)
        PU3 = wrapper2.estimate_unseen_prior() # might not be correct
        pathpriors.USR_raw_prior_Oi = wrapper2.path_raw_prior_Oi
        P_O3,P_Ox3,P_Ux3,mags3,ptbl = on.run_path(frb,usemodel=True,P_U = PU3)
        
        # record this info
        if maglist[2] is None:
            maglist[2] = mags3
            allPOx[2] = P_Ox3
        else:
            maglist[2] = np.append(maglist[2],mags3)
            allPOx[2] = np.append(allPOx[2],P_Ox3)
        
        
        # loudas fsfr = 1.0
        fSFR=1.0
        model2.init_args(fSFR)
        wrapper2.init_zmapping(g.zvals)
        wrapper2.init_path_raw_prior_Oi(DMEG,g)
        PU4 = wrapper2.estimate_unseen_prior() # might not be correct limit
        pathpriors.USR_raw_prior_Oi = wrapper2.path_raw_prior_Oi
        P_O4,P_Ox4,P_Ux4,mags4,ptbl = on.run_path(frb,usemodel=True,P_U = PU4)
        
        # record this info
        if maglist[3] is None:
            maglist[3] = mags4
            allPOx[3] = P_Ox4
        else:
            maglist[3] = np.append(maglist[3],mags4)
            allPOx[3] = np.append(allPOx[3],P_Ox4)
        
        # Marnoch model
        wrapper3.init_zmapping(g.zvals)
        wrapper3.init_path_raw_prior_Oi(DMEG,g)
        PU5 = wrapper3.estimate_unseen_prior() # might not be correct limit
        pathpriors.USR_raw_prior_Oi = wrapper3.path_raw_prior_Oi
        P_O5,P_Ox5,P_Ux5,mags5,ptbl = on.run_path(frb,usemodel=True,P_U = PU5)
        
        # record this info
        if maglist[4] is None:
            maglist[4] = mags5
            allPOx[4] = P_Ox5
        else:
            maglist[4] = np.append(maglist[4],mags5)
            allPOx[4] = np.append(allPOx[4],P_Ox5)
        
        
    # scatter plot of old vs new priors
    plt.figure()
    plt.xlabel("$P(O|x)$ (original)")
    plt.ylabel("$P(O|x)$ (new)")
    for j in np.arange(1,5,1):
        plt.scatter(allPOx[0],allPOx[j],label=labels[j],marker=markers[j])
    plt.legend()
    plt.tight_layout()
    plt.savefig(opdir+"posterior_comparison.png")
    plt.close()
    

 

if __name__ == "__main__":
    
    calc_path_priors()

    
    
