.. _optical:

================================
Optical Host Galaxy Association
================================

This section describes the three modules that connect zdm's redshift-DM
predictions to the `PATH <https://github.com/FRBs/astropath>`_ (Probabilistic
Association of Transients to their Hosts) algorithm. Together they provide
physically motivated priors on FRB host galaxy apparent magnitudes, derived
from the zdm posterior p(z | DM\ :sub:`EG`).

- :mod:`zdm.optical_params` — parameter dataclasses configuring each model
- :mod:`zdm.optical` — host magnitude models and the PATH interface wrapper
- :mod:`zdm.optical_numerics` — numerical evaluation, optimisation, and
  statistics for fitting the models to CRAFT ICS optical data

Overview
========

Standard PATH assigns host galaxy candidates a prior based only on galaxy
surface density and angular size. The zdm optical modules replace this with a
prior informed by p(z | DM\ :sub:`EG`): given an FRB's extragalactic DM, zdm
predicts a redshift distribution, which is convolved with a host galaxy
luminosity model to produce p(m\ :sub:`r` | DM\ :sub:`EG`).

The modules are built around a two-layer design:

.. code-block:: text

   ┌──────────────────────────────────────────────────────────────┐
   │                  model_wrapper  (optical.py)                 │
   │   Convolves p(m_r|z) with zdm p(z|DM_EG) → p(m_r|DM_EG)   │
   │   Estimates P_U (undetected host prior)                      │
   │   Plugs into PATH via pathpriors.USR_raw_prior_Oi           │
   └──────────────────────────────────────────────────────────────┘
                              │ wraps
   ┌──────────────────────────────────────────────────────────────┐
   │               Host magnitude models  (optical.py)            │
   │                                                              │
   │   simple_host_model  — parametric p(M_r) histogram          │
   │   loudas_model       — mass/SFR-weighted tables (Loudas)    │
   │   marnoch_model      — Gaussian fit to known hosts          │
   │                        (Marnoch et al. 2023)                │
   └──────────────────────────────────────────────────────────────┘
                              │ configured by
   ┌──────────────────────────────────────────────────────────────┐
   │              Parameter dataclasses  (optical_params.py)      │
   │                                                              │
   │   OpticalState  ←  SimpleParams, LoudasParams,              │
   │                     Apparent, Identification                 │
   └──────────────────────────────────────────────────────────────┘

Host Magnitude Models
=====================

Three models are available, all implementing the same interface
``get_pmr_gz(mrbins, z)`` which returns p(m\ :sub:`r` | z) for a set of
apparent magnitude bin edges at a given redshift.

simple_host_model
-----------------

A parametric model describing the intrinsic absolute magnitude distribution
p(M\ :sub:`r`) as N amplitudes (default 10) at uniformly spaced points
between ``Absmin`` and ``Absmax``. The amplitudes are normalised to sum to
unity and interpolated onto a fine internal grid via one of four schemes
controlled by ``AbsModelID``:

.. list-table::
   :header-rows: 1
   :widths: 15 85

   * - ``AbsModelID``
     - Description
   * - 0
     - Step-function histogram — each parameter value applies uniformly to
       its bin
   * - 1
     - Linear interpolation between parameter points *(default)*
   * - 2
     - Cubic spline interpolation (negative values clamped to zero)
   * - 3
     - Cubic spline in log-space (parameters are log\ :sub:`10` weights)

Conversion from M\ :sub:`r` to m\ :sub:`r` is controlled by ``AppModelID``:

.. list-table::
   :header-rows: 1
   :widths: 15 85

   * - ``AppModelID``
     - Description
   * - 0
     - Pure distance modulus, no k-correction *(default)*
   * - 1
     - Distance modulus plus power-law k-correction
       ``2.5 × k × log10(1 + z)``

loudas_model
------------

Uses precomputed p(m\ :sub:`r` | z) tables from Nick Loudas, constructed by
weighting galaxy luminosities by either stellar mass or star-formation rate.
The single free parameter ``fSFR`` interpolates between the two:

.. math::

   p(m_r | z) = (1 - f_{\rm SFR})\,p_{\rm mass}(m_r | z)
                + f_{\rm SFR}\,p_{\rm SFR}(m_r | z)

Interpolation between tabulated redshift bins is performed in log-z space
with a luminosity-distance shift applied to each tabulated distribution before
combining, ensuring correct apparent magnitude evolution at low redshift.

marnoch_model
-------------

A zero-parameter model based on Marnoch et al. 2023 (MNRAS 525, 994). Fits
a Gaussian to the r-band magnitude distribution of known CRAFT ICS FRB host
galaxies, with mean and standard deviation described as cubic splines of
redshift. No free parameters; the model is fixed by the observed host sample.

The ``model_wrapper`` Class
===========================

:class:`~zdm.optical.model_wrapper` is a survey-independent wrapper around
any host model. Its key responsibilities are:

1. **Precomputation**: at initialisation it calls ``model.get_pmr_gz`` for
   every redshift value in the zdm grid to build a cached
   p(m\ :sub:`r` | z) array.
2. **DM integration**: ``init_path_raw_prior_Oi(DM, grid)`` extracts
   p(z | DM\ :sub:`EG`) from the grid and convolves it with the cached
   array to produce p(m\ :sub:`r` | DM\ :sub:`EG`).
3. **P_U estimation**: ``estimate_unseen_prior()`` integrates the magnitude
   prior against the detection probability curve
   (logistic function centred on ``pU_mean`` with width ``pU_width``) to
   obtain the prior probability that the true host is below the detection
   limit.
4. **PATH interface**: after ``init_path_raw_prior_Oi`` is called,
   ``pathpriors.USR_raw_prior_Oi`` is automatically pointed at
   ``path_raw_prior_Oi``, so PATH uses the zdm-derived prior transparently.

Typical Workflow
================

The following example shows how to obtain zdm-informed PATH posteriors for a
single CRAFT ICS FRB.

.. code-block:: python

   from zdm import optical as opt
   from zdm import optical_numerics as on
   from zdm import loading, cosmology as cos, parameters

   # 1. Initialise zdm grid
   state = parameters.State()
   cos.set_cosmology(state)
   cos.init_dist_measures()
   ss, gs = loading.surveys_and_grids(survey_names=['CRAFT_ICS_1300'])
   g, s = gs[0], ss[0]

   # 2. Choose a host magnitude model
   model = opt.marnoch_model()               # or simple_host_model / loudas_model

   # 3. Wrap it for the survey's redshift grid
   wrapper = opt.model_wrapper(model, g.zvals)

   # 4. For a specific FRB, look up its DM_EG
   frb = 'FRB20190608B'
   imatch = opt.matchFRB(frb, s)
   DMEG = s.DMEGs[imatch]

   # 5. Compute p(m_r | DM_EG) and estimate P_U
   wrapper.init_path_raw_prior_Oi(DMEG, g)   # also sets pathpriors.USR_raw_prior_Oi
   PU = wrapper.estimate_unseen_prior()

   # 6. Run PATH with the zdm prior
   P_O, P_Ox, P_Ux, mags, ptbl = on.run_path(frb, usemodel=True, P_U=PU)

To process the full CRAFT ICS sample and compare models, use
:func:`~zdm.optical_numerics.calc_path_priors` directly. To fit model
parameters, pass :func:`~zdm.optical_numerics.function` as the objective to
``scipy.optimize.minimize`` — see ``zdm/scripts/Path/optimise_host_priors.py``
for a complete example.

Parameter Reference
===================

All host galaxy model parameters are held in dataclasses collected by
:class:`~zdm.optical_params.OpticalState`. The four constituent dataclasses
and their parameters are described below.

SimpleParams
------------

Controls the :class:`~zdm.optical.simple_host_model`.

.. list-table::
   :header-rows: 1
   :widths: 20 12 12 56

   * - Parameter
     - Default
     - Units
     - Description
   * - ``Absmin``
     - −25
     - M\ :sub:`r`
     - Minimum absolute magnitude of the host distribution
   * - ``Absmax``
     - −15
     - M\ :sub:`r`
     - Maximum absolute magnitude of the host distribution
   * - ``NAbsBins``
     - 1000
     - —
     - Number of internal absolute magnitude bins (fine grid for
       computing p(m\ :sub:`r` | z))
   * - ``NModelBins``
     - 10
     - —
     - Number of free parameter bins describing p(M\ :sub:`r`)
   * - ``AbsPriorMeth``
     - 0
     - —
     - Initial prior on absolute magnitudes: 0 = uniform
   * - ``AbsModelID``
     - 1
     - —
     - Interpolation scheme for p(M\ :sub:`r`): 0 = histogram,
       1 = linear, 2 = spline, 3 = log-spline
   * - ``AppModelID``
     - 0
     - —
     - Absolute-to-apparent conversion: 0 = distance modulus only,
       1 = with power-law k-correction
   * - ``k``
     - 0.0
     - —
     - k-correction power-law index (only used when ``AppModelID=1``)

LoudasParams
------------

Controls the :class:`~zdm.optical.loudas_model`.

.. list-table::
   :header-rows: 1
   :widths: 20 12 12 56

   * - Parameter
     - Default
     - Units
     - Description
   * - ``fSFR``
     - 0.5
     - —
     - Fraction of FRB hosts tracing star-formation rate (0 = pure
       mass-weighted, 1 = pure SFR-weighted)
   * - ``NzBins``
     - 10
     - —
     - Number of redshift bins for histogram calculations
   * - ``zmin``
     - 0.0
     - —
     - Minimum redshift for p(m\ :sub:`r`) calculation
   * - ``zmax``
     - 0.0
     - —
     - Maximum redshift for p(m\ :sub:`r`) calculation
   * - ``NMrBins``
     - 0
     - —
     - Number of absolute magnitude bins
   * - ``Mrmin``
     - 0.0
     - M\ :sub:`r`
     - Minimum absolute magnitude
   * - ``Mrmax``
     - 0.0
     - M\ :sub:`r`
     - Maximum absolute magnitude

Apparent
--------

Controls the apparent magnitude grid used by :class:`~zdm.optical.model_wrapper`.

.. list-table::
   :header-rows: 1
   :widths: 20 12 12 56

   * - Parameter
     - Default
     - Units
     - Description
   * - ``Appmin``
     - 10
     - m\ :sub:`r`
     - Minimum apparent magnitude of the internal grid
   * - ``Appmax``
     - 35
     - m\ :sub:`r`
     - Maximum apparent magnitude of the internal grid
   * - ``NAppBins``
     - 250
     - —
     - Number of apparent magnitude bins

Identification
--------------

Controls the survey detection completeness model used to compute P_U.
The detection probability is modelled as a logistic function:
p(detected | m\ :sub:`r`) = 1 − p(U | m\ :sub:`r`) where

.. math::

   p(U | m_r) = \frac{1}{1 + \exp\!\left(\frac{\mu - m_r}{w}\right)}

with μ = ``pU_mean`` and w = ``pU_width``.

.. list-table::
   :header-rows: 1
   :widths: 20 12 12 56

   * - Parameter
     - Default
     - Units
     - Description
   * - ``pU_mean``
     - 26.385
     - m\ :sub:`r`
     - Magnitude at which 50 % of host galaxies are undetected
       (the survey's half-completeness limit). Default value is
       calibrated to VLT/FORS2 R-band observations.
   * - ``pU_width``
     - 0.279
     - m\ :sub:`r`
     - Characteristic width of the completeness rolloff. Smaller
       values give a sharper transition between detected and
       undetected regimes.

Optimisation and Statistics
============================

:mod:`zdm.optical_numerics` provides two goodness-of-fit statistics for
comparing the model-predicted apparent magnitude prior to observed PATH
posteriors across a sample of FRBs:

**Maximum-likelihood statistic** (:func:`~zdm.optical_numerics.calculate_likelihood_statistic`)

For each FRB, evaluates

.. math::

   \ln \mathcal{L}_i = \log_{10}\!\left(\sum_j \frac{P(O_j|x)}{s_i} + P_{U,i}^{\rm prior}\right)

where the sum runs over candidate host galaxies and *s*\ :sub:`i` is a
rescale factor that undoes PATH's internal renormalisation. The total
statistic is Σ ln *ℒ*\ :sub:`i` over all FRBs. This is the recommended
objective for parameter fitting.

**KS-like statistic** (:func:`~zdm.optical_numerics.calculate_ks_statistic`)

Builds normalised cumulative distributions of the model prior and the
observed posteriors (weighted by P(O|x)) over apparent magnitude, then
returns the maximum absolute difference — analogous to the KS test statistic.
Smaller values indicate a better fit.

Both statistics accept a ``POxcut`` argument to restrict the sample to FRBs
with a confidently identified host (max P(O|x) > threshold), simulating a
traditional host-identification approach.

Scripts
=======

Ready-to-run scripts using these modules are in ``zdm/scripts/Path/``:

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Script
     - Purpose
   * - ``estimate_path_priors.py``
     - Demonstrate zdm-informed PATH priors on all CRAFT ICS FRBs;
       compare flat vs. model priors; save posterior magnitude histogram
   * - ``optimise_host_priors.py``
     - Fit host model parameters to the CRAFT ICS sample using
       ``scipy.optimize.minimize``
   * - ``plot_host_models.py``
     - Visualise all three host models and compare their PATH posteriors
       across the CRAFT ICS sample

API Reference
=============

optical_params
--------------

Parameter dataclasses for configuring host galaxy models.

.. automodapi:: zdm.optical_params
   :no-inheritance-diagram:

optical
-------

Host magnitude model classes and the ``model_wrapper`` PATH interface.

.. automodapi:: zdm.optical
   :no-inheritance-diagram:

optical_numerics
----------------

Numerical evaluation, optimisation, and statistics for fitting host models.

.. automodapi:: zdm.optical_numerics
   :no-inheritance-diagram:

References
==========

- Marnoch et al. 2023, MNRAS 525, 994 —
  FRB host galaxy r-band magnitude model
  (https://doi.org/10.1093/mnras/stad2353)
- Macquart et al. 2020, Nature 581, 391 —
  Macquart relation / p(DM | z)
- Aggarwal et al. 2021, ApJ 911, 95 —
  PATH algorithm for probabilistic host association
