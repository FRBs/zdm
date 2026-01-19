.. zdm documentation master file

===
zdm
===

**zdm** is a Python package for Fast Radio Burst (FRB) redshift-dispersion measure
(z-DM) calculations. It provides tools for statistical modeling of FRB populations,
computing likelihoods, and constraining cosmological and FRB population parameters.

The package implements the theoretical framework for relating observed FRB dispersion
measures to their redshifts, accounting for contributions from the Milky Way, host
galaxies, and the intergalactic medium.

.. image:: https://github.com/FRBs/zdm/workflows/CI%20Tests/badge.svg
   :target: https://github.com/FRBs/zdm/actions

Key Features
============

- **Cosmological Modeling**: Lambda-CDM cosmology with configurable parameters
- **z-DM Grids**: Efficient computation of probability distributions over redshift and dispersion measure
- **Survey Support**: Built-in support for CHIME, ASKAP, Parkes and other FRB surveys
- **Likelihood Analysis**: Compute likelihoods for FRB populations given survey data
- **MCMC Parameter Estimation**: Integration with emcee for Bayesian parameter inference
- **Repeater Modeling**: Support for repeating FRB population models

Getting Started
===============

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   installation
   quickstart
   tutorials

.. toctree::
   :maxdepth: 2
   :caption: Reference

   architecture
   parameters
   api

.. toctree::
   :maxdepth: 1
   :caption: Development

   contributing

Citation
========

If you use ``zdm`` in your research, please cite:

.. code-block:: bibtex

   @article{james2022,
      author = {James, Clancy W. and others},
      title = {A measurement of Hubble's Constant using Fast Radio Bursts},
      journal = {MNRAS},
      year = {2022}
   }

License
=======

``zdm`` is licensed under a 3-clause BSD style license. See the LICENSE file
in the repository for details.

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
