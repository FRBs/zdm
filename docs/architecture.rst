.. _architecture:

============
Architecture
============

This document describes the high-level architecture of the ``zdm`` package
and how the various components interact.

Overview
========

The ``zdm`` package follows a layered architecture:

1. **Configuration Layer**: Parameter management via dataclasses
2. **Data Layer**: Survey definitions and FRB observations
3. **Computation Layer**: Cosmology, p(DM|z), and grid calculations
4. **Analysis Layer**: Likelihood computation and MCMC inference

.. code-block:: text

   ┌─────────────────────────────────────────────────────────┐
   │                    Analysis Layer                       │
   │              (MCMC, iteration, analyze_cube)            │
   └─────────────────────────────────────────────────────────┘
                              │
   ┌─────────────────────────────────────────────────────────┐
   │                  Computation Layer                      │
   │         (Grid, cosmology, pcosmic, energetics)          │
   └─────────────────────────────────────────────────────────┘
                              │
   ┌─────────────────────────────────────────────────────────┐
   │                     Data Layer                          │
   │              (Survey, survey_data, beams)               │
   └─────────────────────────────────────────────────────────┘
                              │
   ┌─────────────────────────────────────────────────────────┐
   │                Configuration Layer                      │
   │            (State, parameters, data_class)              │
   └─────────────────────────────────────────────────────────┘

Core Classes
============

State
-----

:class:`~zdm.parameters.State` is the central configuration object containing
all model parameters. It aggregates several parameter dataclasses:

- ``cosmo``: Cosmological parameters (H0, Omega_m, etc.)
- ``FRBdemo``: FRB population demographics
- ``energy``: Luminosity function parameters
- ``host``: Host galaxy DM distribution
- ``MW``: Milky Way DM contributions
- ``IGM``: Intergalactic medium parameters
- ``width``: Intrinsic width distribution
- ``scat``: Scattering parameters
- ``rep``: Repeater population parameters
- ``analysis``: Analysis control flags

.. code-block:: python

   state = parameters.State()
   # All parameter groups are accessible as attributes
   state.cosmo.H0  # Hubble constant
   state.energy.gamma  # Luminosity function slope

Survey
------

:class:`~zdm.survey.Survey` represents an FRB survey with:

- Instrument properties (beam pattern, frequency, threshold)
- Survey metadata (observation time, field of view)
- Detected FRBs with measured DMs and positions
- Detection efficiency as function of DM

Surveys are initialized from ECSV files in ``zdm/data/Surveys/``.

Grid
----

:class:`~zdm.grid.Grid` is the computational core, building a 2D probability
distribution over redshift and DM:

1. Takes a Survey and State as input
2. Computes detection thresholds and efficiencies
3. Applies beam pattern weighting
4. Calculates expected detection rates per z-DM cell

The grid represents P(detection | z, DM, survey) weighted by the
intrinsic FRB rate density.

Data Flow
=========

Typical Workflow
----------------

.. code-block:: text

   1. Initialize State with parameters
            │
            ▼
   2. Load Survey from data file
            │
            ▼
   3. Initialize cosmology distances
            │
            ▼
   4. Compute p(DM|z) grid (pcosmic)
            │
            ▼
   5. Build Grid for Survey
            │
            ▼
   6. Compute likelihood via iteration
            │
            ▼
   7. MCMC or optimization over parameters

Grid Construction
-----------------

The Grid construction involves several steps:

1. **parse_grid**: Store z, DM arrays and base p(DM|z) grid
2. **calc_dV**: Compute comoving volume elements per z bin
3. **smear_dm**: Apply DM smearing from host and halo contributions
4. **calc_thresholds**: Compute energy thresholds for detection
5. **calc_pdv**: Combine volume and threshold calculations
6. **set_evolution**: Apply source evolution model
7. **calc_rates**: Compute final detection rates

Likelihood Calculation
----------------------

The :func:`~zdm.iteration.get_log_likelihood` function computes:

.. math::

   \ln \mathcal{L} = \sum_i \ln p({\rm DM}_i, z_i | \theta) - N_{\rm exp}(\theta) + \ln N_{\rm obs}!

Where:

- Sum is over observed FRBs
- :math:`N_{\rm exp}` is expected number of detections
- :math:`\theta` represents model parameters

Key Modules
===========

cosmology
---------

Implements Lambda-CDM cosmology:

- Distance measures (comoving, luminosity, angular diameter)
- Volume elements
- Source evolution functions

Uses spline interpolation for efficiency.

pcosmic
-------

Computes p(DM_cosmic | z), the probability distribution of cosmic DM
given redshift. Implements the Macquart et al. formalism with the
fluctuation parameter F.

energetics
----------

Implements the FRB luminosity function using incomplete gamma functions.
Uses spline interpolation of pre-computed values for speed.

Supports multiple luminosity function forms:

- Power-law
- Gamma function
- Spline + gamma hybrid

iteration
---------

Contains likelihood calculation routines:

- ``get_log_likelihood``: Main likelihood function
- ``calc_likelihoods_1D``: For DM-only fits
- ``calc_likelihoods_2D``: For DM+z fits

Supports various likelihood components (normalization, p(SNR), etc.)

MCMC
----

Parameter estimation using emcee:

- ``calc_log_posterior``: Posterior evaluation
- Supports uniform priors with configurable bounds
- Handles multiple surveys simultaneously

File Organization
=================

.. code-block:: text

   zdm/
   ├── __init__.py
   ├── parameters.py      # State and parameter dataclasses
   ├── data_class.py      # Base dataclass utilities
   ├── survey.py          # Survey class
   ├── grid.py            # Grid class
   ├── repeat_grid.py     # Repeater grid extension
   ├── cosmology.py       # Cosmological calculations
   ├── pcosmic.py         # p(DM|z) calculations
   ├── energetics.py      # Luminosity functions
   ├── iteration.py       # Likelihood calculations
   ├── MCMC.py            # MCMC parameter estimation
   ├── loading.py         # High-level loading functions
   ├── misc_functions.py  # Utility functions
   ├── figures.py         # Plotting routines
   ├── beams.py           # Beam pattern handling
   ├── data/
   │   ├── Surveys/       # Survey definition files
   │   ├── BeamData/      # Beam response files
   │   └── Cube/          # Precomputed grids
   └── tests/             # Test suite

Extension Points
================

Adding New Surveys
------------------

1. Create an ECSV file in ``zdm/data/Surveys/`` with FRB data
2. Add beam pattern files to ``zdm/data/BeamData/`` if needed
3. Load via :class:`~zdm.survey.Survey` class

Custom Luminosity Functions
---------------------------

1. Add function to ``energetics.py``
2. Register in ``luminosity_function`` parameter options
3. Update grid initialization to use new function

New Source Evolution Models
---------------------------

1. Add evolution function to ``cosmology.py``
2. Register in ``source_evolution`` parameter options
3. Grid will automatically use via ``set_evolution()``
