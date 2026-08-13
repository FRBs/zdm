.. _api:

=============
API Reference
=============

This section provides detailed API documentation for all public modules
and classes in the ``zdm`` package.

Core Modules
============

These modules form the backbone of the zdm package, providing parameter
management, survey handling, and z-DM grid computation.

parameters
----------

Central configuration system for FRB z-DM analysis. All model parameters
are organized into dataclasses grouped by category (cosmology, FRB demographics,
host galaxy, IGM, etc.). The ``State`` class aggregates all parameter groups.

.. automodapi:: zdm.parameters
   :no-inheritance-diagram:

survey
------

FRB Survey class for modeling telescope observations. Encapsulates instrument
characteristics, beam patterns, detection efficiencies, and detected FRB data.
Survey files are loaded from ECSV format with header metadata and FRB tables.

.. automodapi:: zdm.survey
   :no-inheritance-diagram:

grid
----

Core z-DM grid class for FRB population modeling. Computes 2D probability
distributions of FRB detection rates as a function of redshift and DM by
combining cosmological volumes, p(DM|z), telescope efficiency, and luminosity
functions.

.. automodapi:: zdm.grid
   :no-inheritance-diagram:

repeat_grid
-----------

Grid extension for repeating FRB population modeling. Models expected numbers
of single and repeated detections given a distribution of repeater rates
following dN/dR ~ R^Rgamma.

.. automodapi:: zdm.repeat_grid
   :no-inheritance-diagram:

Computation Modules
===================

Physics and statistics modules for computing cosmological quantities,
DM probability distributions, and energy functions.

cosmology
---------

Lambda CDM cosmology calculations including distance measures (comoving,
angular diameter, luminosity), volume elements, and source evolution functions.
Uses interpolation tables for fast array operations.

.. automodapi:: zdm.cosmology
   :no-inheritance-diagram:

pcosmic
-------

Probability distribution of cosmic dispersion measure given redshift, p(DM|z).
Implements the Macquart relation from Macquart et al. (2020) with the feedback
parameter F controlling variance. Also provides host galaxy DM convolution kernels.

.. automodapi:: zdm.pcosmic
   :no-inheritance-diagram:

energetics
----------

FRB luminosity/energy function implementations including power-law and
gamma-function distributions. Uses spline interpolation of the upper incomplete
gamma function for computational efficiency during grid calculations.

.. automodapi:: zdm.energetics
   :no-inheritance-diagram:

iteration
---------

Likelihood calculation routines for z-DM grids. Computes log-likelihoods of
FRB survey data given model predictions, including components for p(DM,z),
Poisson number counts, SNR distributions, and width/scattering.

.. automodapi:: zdm.iteration
   :no-inheritance-diagram:

Utility Modules
===============

Helper functions for data loading, plotting, and common operations.

loading
-------

High-level functions for loading surveys and initializing analysis state.
Provides convenience functions like ``set_state()`` for creating properly
configured State objects with default or best-fit parameters.

.. automodapi:: zdm.loading
   :no-inheritance-diagram:

misc_functions
--------------

Miscellaneous utility functions including grid initialization, parameter
updates, probability calculations, and other common operations used
throughout the package.

.. automodapi:: zdm.misc_functions
   :no-inheritance-diagram:

figures
-------

Plotting functions for visualizing z-DM grids and FRB data. Provides
publication-quality plots of probability distributions and analysis results.

.. automodapi:: zdm.figures
   :no-inheritance-diagram:

beams
-----

Telescope beam pattern modeling utilities. Provides functions for generating
and loading beam patterns (Gaussian, Airy, measured) that affect solid angle
and sensitivity variations across the field of view.

.. automodapi:: zdm.beams
   :no-inheritance-diagram:

Analysis Modules
================

Parameter estimation and analysis tools.

MCMC
----

MCMC parameter estimation using the emcee package. Provides functions for
running samplers, computing log-posteriors, and exploring parameter space
with support for multiple surveys and uniform priors.

.. automodapi:: zdm.MCMC
   :no-inheritance-diagram:

MCMC_analysis
-------------

Analysis utilities for MCMC chains including convergence diagnostics,
corner plots, and posterior summaries.

.. automodapi:: zdm.MCMC_analysis
   :no-inheritance-diagram:

analyze_cube
------------

Tools for analyzing pre-computed parameter cubes.

.. automodapi:: zdm.analyze_cube
   :no-inheritance-diagram:

vvmax
-----

V/Vmax statistical tests for FRB population analysis.

.. automodapi:: zdm.vvmax
   :no-inheritance-diagram:

Data Classes
============

Base classes and data structures.

data_class
----------

Base dataclass utilities for parameter management. Provides common functionality
for serialization, dictionary access, and parameter metadata handling used by
all parameter dataclasses.

.. automodapi:: zdm.data_class
   :no-inheritance-diagram:

survey_data
-----------

Data structures for survey metadata and FRB observations.

.. automodapi:: zdm.survey_data
   :no-inheritance-diagram:

Specialized Modules
===================

Additional modules for specific use cases.

galactic_dm_models
------------------

Models for Galactic DM contributions including different halo models.

.. automodapi:: zdm.galactic_dm_models
   :no-inheritance-diagram:

optical
-------

Optical counterpart and host galaxy association utilities.

.. automodapi:: zdm.optical
   :no-inheritance-diagram:

io
--

Input/output utilities for file handling including JSON I/O and
grid data persistence.

.. automodapi:: zdm.io
   :no-inheritance-diagram:
