.. _quickstart:

==========
Quickstart
==========

This guide walks you through the basic workflow of using ``zdm`` to analyze
FRB populations and compute likelihoods.

Basic Concepts
==============

The ``zdm`` package is built around several key concepts:

1. **State**: A configuration object holding all model parameters
2. **Survey**: Represents an FRB survey with instrument properties and detected FRBs
3. **Grid**: A 2D probability grid over redshift (z) and dispersion measure (DM)
4. **Likelihood**: Statistical comparison between model predictions and observations

Setting Up a State
==================

The :class:`~zdm.parameters.State` object holds all model parameters:

.. code-block:: python

   from zdm import parameters
   from zdm import loading

   # Create a state with default parameters
   state = loading.set_state()

   # Or create a blank state and customize
   state = parameters.State()

   # Modify parameters
   vparams = {
       'cosmo': {'H0': 70.0},
       'energy': {'gamma': -1.1, 'lEmax': 41.5},
       'host': {'lmean': 2.2, 'lsigma': 0.5}
   }
   state.update_param_dict(vparams)

Initializing Cosmology
======================

Before computing grids, initialize the cosmological distance measures:

.. code-block:: python

   from zdm import cosmology as cos

   # Initialize with state parameters
   cos.set_cosmology(state.params)
   cos.init_dist_measures()

Loading a Survey
================

Surveys are loaded from data files in the ``zdm/data/Surveys/`` directory:

.. code-block:: python

   import numpy as np
   from zdm import survey

   # Define DM and z grids
   dmvals = np.linspace(0, 3000, 1000)
   zvals = np.linspace(0, 3, 500)

   # Load an ASKAP survey
   s = survey.Survey(
       state,
       survey_name='CRAFT_ICS',
       filename='CRAFT_ICS.ecsv',
       dmvals=dmvals,
       zvals=zvals
   )

Building a Grid
===============

The :class:`~zdm.grid.Grid` computes detection probabilities across z-DM space:

.. code-block:: python

   from zdm import misc_functions as mf

   # Get the z-DM probability grid (p(DM|z))
   zDMgrid, zvals, dmvals, smear = mf.get_zdm_grid(
       state,
       new=True,
       plot=False
   )

   # Create the survey grid
   from zdm import grid as zdm_grid

   g = zdm_grid.Grid(
       survey=s,
       state=state,
       zDMgrid=zDMgrid,
       zvals=zvals,
       dmvals=dmvals,
       smear_mask=smear
   )

Computing Likelihoods
=====================

Compute the log-likelihood of the model given observed FRBs:

.. code-block:: python

   from zdm import iteration as it

   # Compute log-likelihood
   ll = it.get_log_likelihood(g, s)
   print(f"Log-likelihood: {ll:.2f}")

Complete Example
================

Here's a complete example loading CHIME data and computing likelihoods:

.. code-block:: python

   from zdm import loading
   from zdm import cosmology as cos
   from zdm import iteration as it

   # Load CHIME survey and grids
   dmvals, zvals, grids, surveys = loading.load_CHIME(
       Nbin=6,
       make_plots=False
   )

   # Compute total likelihood across all declination bins
   total_ll = 0
   for g, s in zip(grids, surveys):
       ll = it.get_log_likelihood(g, s)
       total_ll += ll
       print(f"Survey {s.name}: ll = {ll:.2f}")

   print(f"Total log-likelihood: {total_ll:.2f}")

Working with Parameters
=======================

Individual parameters can be updated and the grid recalculated:

.. code-block:: python

   # Update a single parameter
   state.update_param('H0', 73.0)

   # Update multiple parameters
   vparams = {
       'energy': {'gamma': -1.2},
       'host': {'lmean': 2.0}
   }
   state.update_param_dict(vparams)

   # Rebuild grid with new parameters
   # (Note: some parameters require full grid rebuild)

Visualization
=============

The ``figures`` module provides plotting utilities:

.. code-block:: python

   from zdm import figures

   # Plot the z-DM grid
   figures.plot_grid(g, s, show=True)

Next Steps
==========

- See :ref:`parameters` for a complete list of model parameters
- See :ref:`architecture` for details on the code structure
- See :ref:`tutorials` for more detailed examples
