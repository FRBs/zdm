"""
Estimate zdm-informed PATH priors for CRAFT/ICS FRB host galaxies.

This script demonstrates how to incorporate zdm-derived p(z|DM) predictions
as priors for the PATH (Probabilistic Association of Transients to their Hosts)
algorithm applied to CRAFT ICS FRBs.

For each FRB in the CRAFT ICS sample (`opt.frblist`), the script runs PATH
twice and compares results:

1. **Baseline run**: PATH with a flat (uninformative) prior on host galaxy
   apparent magnitude, and a fixed prior P_U=0.1 on the host being below
   the detection threshold.

2. **zdm-informed run**: PATH using a physically motivated prior on host
   apparent magnitude derived from the Marnoch+2023 host galaxy luminosity
   model combined with the zdm p(z|DM_EG) probability distribution. The
   probability P_U that the true host is undetected is also estimated from
   the model rather than set by hand.

The output is a weighted histogram of posterior host galaxy apparent
magnitudes (P_Ox) across all FRBs, saved to ``posterior_pOx.png``.

Note: This script does NOT optimise any zdm or host galaxy model parameters.
It uses the CRAFT_ICS_1300 survey grid with default zdm parameter values.

Requirements
------------
- ``astropath`` package (PATH implementation)
- ``frb`` package (FRB utilities and optical data)
- PATH-compatible optical data for each FRB in ``opt.frblist``

References
----------
- Marnoch et al. 2023, MNRAS 525, 994 (host galaxy luminosity model)
- Macquart et al. 2020 (Macquart relation / p(DM|z))
"""

#standard Python imports
import numpy as np
from matplotlib import pyplot as plt

# imports from the "FRB" series
from zdm import optical as opt
from zdm import optical_numerics as on
from zdm import loading
from zdm import cosmology as cos
from zdm import parameters
from zdm import loading
from zdm import frb_lists as lists

import astropath.priors as pathpriors


def calc_path_priors():
    """
    Run PATH on all CRAFT ICS FRBs with and without zdm-derived priors.

    Initialises a zdm grid for the CRAFT_ICS_1300 survey and the Marnoch+2023
    host galaxy luminosity model. For each FRB in ``frblist.icslist``:

    - Matches the FRB to the CRAFT_ICS_1300 survey to retrieve its
      extragalactic dispersion measure (DM_EG).
    - Runs PATH with a flat apparent-magnitude prior and fixed P_U=0.1
      (``usemodel=False``), giving baseline posteriors P_Ox1.
    - Uses the zdm model to compute a physically motivated prior on apparent
      host magnitude, p(m_r | DM_EG), via ``wrapper.init_path_raw_prior_Oi``,
      and estimates P_U from the fraction of the magnitude prior that falls
      below the survey detection limit via ``wrapper.estimate_unseen_prior``.
    - Runs PATH again with the zdm-derived prior (``usemodel=True``) to give
      updated posteriors P_Ox2.

    After processing all FRBs, produces a weighted histogram of the posterior
    host apparent magnitudes (P_Ox2) across the whole sample and saves it to
    ``posterior_pOx.png``.

    Notes
    -----
    FRBs not found in the CRAFT_ICS_1300 survey (e.g. because they were
    detected by a different instrument configuration) are skipped with a
    warning.

    The zdm model parameters are held fixed at their default values; no
    parameter optimisation is performed here. See
    ``optimise_host_priors.py`` for the equivalent script with optimisation.
    """
    
    frblist = lists.icslist
    
    NFRB = len(frblist)
    
    # here is where I should initialise a zDM grid
    state = parameters.State()
    cos.set_cosmology(state)
    cos.init_dist_measures()
    model = opt.marnoch_model()
    name='CRAFT_ICS_1300'
    ss,gs = loading.surveys_and_grids(survey_names=[name])
    g = gs[0]
    s = ss[0]
    # must be done once for any fixed zvals
    wrapper = opt.model_wrapper(model,g.zvals)
    
    # do this only for a particular FRB
    # it gives a prior on apparent magnitude and pz
    #AppMagPriors,pz = model.get_posterior(g,DMlist)
    
    allmags = None
    allPOx = None
    
    for frb in frblist:
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
        
        DMEG = s.DMEGs[imatch]
        
        #
        
        # original calculation
        P_O1,P_Ox1,P_Ux1,mags1,ptbl = on.run_path(frb,usemodel=False,P_U=0.1)
        
        # initialises wrapper to give p(mr|DMEG) for p(z|DM) grid predictions
        wrapper.init_path_raw_prior_Oi(DMEG,g)
        PU = wrapper.estimate_unseen_prior()
        
        P_O2,P_Ox2,P_Ux2,mags2,ptbl = on.run_path(frb,usemodel=True,P_U = PU)
        
        if False:
            # compares outcomes
            print("FRB ",frb)
            print(" m_r               P_O: old               new               P_Ox: old               new")
            for i,P_O in enumerate(P_O1):
                print(i,mags1[i],P_O1[i],P_O2[i],P_Ox1[i],P_Ox2[i])
            print("\n")
        
        # keep cumulative histogram of posterior magnitude distributions
        #allmags.append(mags2)
        #allPOx.append(P_Ox2)
        mags2 = np.array(mags2)
        
        if allmags is None:
            allmags = mags2
            allPOx = P_Ox2
        else:
            allmags = np.append(allmags,mags2)
            allPOx = np.append(allPOx,P_Ox2)
    
    Nbins = int(wrapper.Appmax - wrapper.Appmin)+1
    bins = np.linspace(wrapper.Appmin,wrapper.Appmax,Nbins)
    plt.figure()
    plt.hist(allmags,weights = allPOx, bins = bins,label="Posterior")
    plt.legend()
    plt.tight_layout()
    plt.savefig("posterior_pOx.png")
    plt.close()
    


 

if __name__ == "__main__":
    
    calc_path_priors()

    
    
