.. _tutorials:

=========
Tutorials
=========

This section contains tutorials demonstrating common use cases for ``zdm``.

Available Notebooks
===================

The ``docs/nb/`` directory contains Jupyter notebooks with worked examples:

- **CHIME_pzDM.ipynb**: Working with CHIME survey data and p(z|DM) calculations
- **Exploring_H0.ipynb**: Exploring the effect of varying H0
- **Exploring_Emax.ipynb**: Exploring maximum energy parameter effects
- **Exploring_alpha.ipynb**: Exploring spectral index variations
- **Grid_sz.ipynb**: Understanding grid size and resolution
- **Max_Like.ipynb**: Maximum likelihood estimation
- **Median_DMcosmic.ipynb**: Computing median cosmic DM
- **Omegab_H0.ipynb**: Relationship between Omega_b and H0
- **Speedup_IGamma.ipynb**: Performance optimization with gamma functions

Tutorial: Basic Likelihood Calculation
======================================

This tutorial walks through computing likelihoods for a simple case.

Step 1: Setup
-------------

.. code-block:: python

   import numpy as np
   from zdm import parameters, survey, cosmology as cos
   from zdm import misc_functions as mf
   from zdm import grid as zdm_grid
   from zdm import iteration as it

   # Create state with default parameters
   state = parameters.State()

Step 2: Initialize Cosmology
----------------------------

.. code-block:: python

   # Set cosmological parameters
   cos.set_cosmology(state.params)
   cos.init_dist_measures()

Step 3: Define Grid Dimensions
------------------------------

.. code-block:: python

   # DM range (pc/cm^3)
   dmvals = np.linspace(0, 3000, 500)

   # Redshift range
   zvals = np.linspace(0.01, 3.0, 400)

Step 4: Load Survey
-------------------

.. code-block:: python

   # Load ASKAP ICS survey
   s = survey.Survey(
       state,
       survey_name='CRAFT_ICS',
       filename='CRAFT_ICS.ecsv',
       dmvals=dmvals,
       zvals=zvals
   )

Step 5: Build z-DM Grid
-----------------------

.. code-block:: python

   # Get base p(DM|z) grid
   zDMgrid, zvals, dmvals, smear = mf.get_zdm_grid(
       state, new=True, plot=False
   )

   # Build survey-specific grid
   g = zdm_grid.Grid(
       survey=s,
       state=state,
       zDMgrid=zDMgrid,
       zvals=zvals,
       dmvals=dmvals,
       smear_mask=smear
   )

Step 6: Compute Likelihood
--------------------------

.. code-block:: python

   # Calculate log-likelihood
   ll = it.get_log_likelihood(g, s)
   print(f"Log-likelihood: {ll:.2f}")

Tutorial: Parameter Exploration
===============================

Explore how likelihood varies with a parameter.

.. code-block:: python

   import numpy as np
   import matplotlib.pyplot as plt
   from zdm import loading, iteration as it

   # Load CHIME data
   dmvals, zvals, grids, surveys = loading.load_CHIME(Nbin=6)

   # Range of H0 values to test
   H0_values = np.linspace(60, 80, 21)
   likelihoods = []

   for H0 in H0_values:
       # Update H0 in state
       state = grids[0].state
       state.update_param('H0', H0)

       # Rebuild grids and compute likelihood
       # (simplified - full implementation needs grid rebuild)
       total_ll = sum(it.get_log_likelihood(g, s)
                      for g, s in zip(grids, surveys))
       likelihoods.append(total_ll)

   # Plot results
   plt.figure()
   plt.plot(H0_values, likelihoods)
   plt.xlabel('H0 (km/s/Mpc)')
   plt.ylabel('Log-likelihood')
   plt.title('Likelihood vs H0')
   plt.show()

Tutorial: MCMC Parameter Estimation
===================================

Run MCMC to constrain parameters.

.. code-block:: python

   from zdm import MCMC
   from zdm import loading
   import emcee

   # Load surveys
   dmvals, zvals, grids, surveys = loading.load_CHIME(Nbin=6)
   state = grids[0].state

   # Define parameters to vary with bounds
   params = {
       'H0': {'min': 50, 'max': 100},
       'gamma': {'min': -2.5, 'max': 0},
       'lEmax': {'min': 40, 'max': 43},
   }

   # Initial walker positions
   nwalkers = 32
   ndim = len(params)
   p0 = np.random.uniform(
       low=[v['min'] for v in params.values()],
       high=[v['max'] for v in params.values()],
       size=(nwalkers, ndim)
   )

   # Setup sampler
   surveys_sep = [surveys, []]  # Non-repeaters, repeaters
   sampler = emcee.EnsembleSampler(
       nwalkers, ndim, MCMC.calc_log_posterior,
       args=[state, params, surveys_sep]
   )

   # Run MCMC
   nsteps = 1000
   sampler.run_mcmc(p0, nsteps, progress=True)

   # Get results
   samples = sampler.get_chain(discard=200, flat=True)

Tutorial: Working with Repeaters
================================

Analyze repeating FRB populations.

.. code-block:: python

   from zdm import repeat_grid

   # Load repeater survey data
   # (requires appropriate survey file with repeater info)

   # Create repeater grid
   rg = repeat_grid.repeat_Grid(
       survey=s,
       state=state,
       zDMgrid=zDMgrid,
       zvals=zvals,
       dmvals=dmvals,
       smear_mask=smear
   )

   # Compute likelihood including repeater terms
   ll = it.get_log_likelihood(rg, s, pNreps=True)

Further Resources
=================

- See the :ref:`api` for complete function documentation
- See the :ref:`parameters` for all available parameters
- Check the ``papers/`` directory for publication-specific analysis scripts
