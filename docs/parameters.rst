.. _parameters:

==========
Parameters
==========

The ``zdm`` package uses a hierarchical parameter system organized into
dataclasses. All parameters are accessible through the :class:`~zdm.parameters.State`
object.

State Object
============

The :class:`~zdm.parameters.State` object is the central configuration container:

.. code-block:: python

   from zdm import parameters

   state = parameters.State()

   # Access parameter groups
   print(state.cosmo.H0)      # Cosmology parameters
   print(state.energy.gamma)  # Energy parameters
   print(state.host.lmean)    # Host galaxy parameters

Updating Parameters
-------------------

Single parameter update:

.. code-block:: python

   state.update_param('H0', 70.0)

Multiple parameter update via dictionary:

.. code-block:: python

   vparams = {
       'cosmo': {'H0': 70.0},
       'energy': {'gamma': -1.1}
   }
   state.update_param_dict(vparams)

Cosmology Parameters (cosmo)
============================

Parameters for Lambda-CDM cosmology. Default values from Planck18.

.. list-table::
   :widths: 20 15 15 50
   :header-rows: 1

   * - Parameter
     - Default
     - Unit
     - Description
   * - ``H0``
     - 67.66
     - km/s/Mpc
     - Hubble constant
   * - ``Omega_k``
     - 0.0
     - --
     - Curvature density parameter
   * - ``Omega_lambda``
     - 0.6889
     - --
     - Dark energy density parameter
   * - ``Omega_m``
     - 0.3111
     - --
     - Matter density parameter
   * - ``Omega_b``
     - 0.0490
     - --
     - Baryon density parameter
   * - ``Omega_b_h2``
     - 0.0224
     - --
     - Baryon density times h^2
   * - ``fix_Omega_b_h2``
     - True
     - --
     - Keep Omega_b_h2 fixed when varying H0

FRB Demographics Parameters (FRBdemo)
=====================================

Parameters controlling FRB source population evolution.

.. list-table::
   :widths: 20 15 15 50
   :header-rows: 1

   * - Parameter
     - Default
     - Unit
     - Description
   * - ``source_evolution``
     - 0
     - --
     - Evolution function: 0=SFR^n, 1=(1+z)^(2.7n)
   * - ``alpha_method``
     - 1
     - --
     - Scaling method: 0=k-correction, 1=rate interpretation
   * - ``sfr_n``
     - 1.77
     - --
     - Scaling of FRB rate with star-formation rate
   * - ``lC``
     - 3.3249
     - --
     - log10 rate constant (Gpc^-3 day^-1 at z=0)

Energy Parameters (energy)
==========================

Parameters for the FRB luminosity/energy function.

.. list-table::
   :widths: 20 15 15 50
   :header-rows: 1

   * - Parameter
     - Default
     - Unit
     - Description
   * - ``lEmin``
     - 30.0
     - log10(erg)
     - Minimum FRB energy
   * - ``lEmax``
     - 41.84
     - log10(erg)
     - Maximum FRB energy
   * - ``alpha``
     - 1.54
     - --
     - Spectral index (rate ~ nu^alpha)
   * - ``gamma``
     - -1.16
     - --
     - Luminosity function slope
   * - ``luminosity_function``
     - 2
     - --
     - LF type: 0=power-law, 1=gamma, 2=spline+gamma, 3=gamma+linear+log10

Host Galaxy Parameters (host)
=============================

Parameters for the host galaxy DM contribution.

.. list-table::
   :widths: 20 15 15 50
   :header-rows: 1

   * - Parameter
     - Default
     - Unit
     - Description
   * - ``lmean``
     - 2.16
     - log10(pc/cm^3)
     - Log-mean of host DM distribution
   * - ``lsigma``
     - 0.51
     - log10(pc/cm^3)
     - Log-sigma of host DM distribution

Milky Way Parameters (MW)
=========================

Parameters for Galactic DM contributions.

.. list-table::
   :widths: 20 15 15 50
   :header-rows: 1

   * - Parameter
     - Default
     - Unit
     - Description
   * - ``ISM``
     - 35.0
     - pc/cm^3
     - Assumed DM for Galactic ISM
   * - ``DMhalo``
     - 50.0
     - pc/cm^3
     - DM contribution from Galactic halo
   * - ``halo_method``
     - 0
     - --
     - Halo model: 0=uniform, 1=Yamasaki & Totani, 2=Das+
   * - ``sigmaDMG``
     - 0.0
     - --
     - Fractional uncertainty in Galactic ISM DM
   * - ``sigmaHalo``
     - 15.0
     - pc/cm^3
     - Uncertainty in halo DM
   * - ``logu``
     - False
     - --
     - Use log-normal for DMG distributions

IGM Parameters (IGM)
====================

Parameters for the intergalactic medium DM distribution.

.. list-table::
   :widths: 20 15 15 50
   :header-rows: 1

   * - Parameter
     - Default
     - Unit
     - Description
   * - ``logF``
     - -0.49
     - --
     - log10(F), cosmic web fluctuation parameter

Width Parameters (width)
========================

Parameters for intrinsic FRB width distribution.

.. list-table::
   :widths: 20 15 15 50
   :header-rows: 1

   * - Parameter
     - Default
     - Unit
     - Description
   * - ``Wlogmean``
     - -0.29
     - log10(ms)
     - Log-mean of intrinsic width distribution
   * - ``Wlogsigma``
     - 0.65
     - log10(ms)
     - Log-sigma of intrinsic width distribution
   * - ``Wmethod``
     - 2
     - --
     - Width method: 0=ignore, 1=intrinsic, 2=+scattering, 3=+z-dep, 4=specific
   * - ``WidthFunction``
     - 2
     - --
     - Width function: 0=log-constant, 1=log-normal, 2=half-lognormal
   * - ``WNbins``
     - 12
     - --
     - Number of width bins
   * - ``Wthresh``
     - 0.5
     - --
     - Starting fraction for width histogramming

Scattering Parameters (scat)
============================

Parameters for FRB scattering distribution.

.. list-table::
   :widths: 20 15 15 50
   :header-rows: 1

   * - Parameter
     - Default
     - Unit
     - Description
   * - ``Slogmean``
     - -1.3
     - log10(ms)
     - Mean of log-scattering distribution at 600 MHz
   * - ``Slogsigma``
     - 0.2
     - log10(ms)
     - Sigma of log-scattering distribution
   * - ``Sfnorm``
     - 1000
     - MHz
     - Reference frequency for scattering
   * - ``Sfpower``
     - -4.0
     - --
     - Frequency scaling power-law index
   * - ``ScatFunction``
     - 2
     - --
     - Scattering function: 0=log-constant, 1=lognormal, 2=half-lognormal
   * - ``Sbackproject``
     - False
     - --
     - Calculate p(tau|w,DM,z) arrays

Repeater Parameters (rep)
=========================

Parameters for repeating FRB population models.

.. list-table::
   :widths: 20 15 15 50
   :header-rows: 1

   * - Parameter
     - Default
     - Unit
     - Description
   * - ``lRmin``
     - -3.0
     - log10(day^-1)
     - Minimum repeater rate
   * - ``lRmax``
     - 1.0
     - log10(day^-1)
     - Maximum repeater rate
   * - ``Rgamma``
     - -2.375
     - --
     - Differential index of repeater density
   * - ``RC``
     - 0.01
     - --
     - Constant repeater density
   * - ``RE0``
     - 1e39
     - erg
     - Energy at which rates are defined

Analysis Parameters (analysis)
==============================

Parameters controlling analysis behavior.

.. list-table::
   :widths: 20 15 15 50
   :header-rows: 1

   * - Parameter
     - Default
     - Unit
     - Description
   * - ``NewGrids``
     - True
     - --
     - Generate new z-DM grids
   * - ``sprefix``
     - "Std"
     - --
     - Calculation detail: "Full" or "Std"
   * - ``min_lat``
     - -1.0
     - degrees
     - Minimum absolute Galactic latitude
   * - ``DMG_cut``
     - None
     - pc/cm^3
     - Maximum DMG to include
