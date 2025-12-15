.. _installation:

============
Installation
============

Requirements
============

``zdm`` requires Python 3.10 or later and depends on several scientific Python packages:

- numpy >= 1.18
- scipy >= 1.4
- astropy >= 5.2.1
- matplotlib >= 3.3
- pandas >= 1.3
- emcee >= 3.1.4
- h5py >= 3.10.0
- mpmath >= 1.2.0

Additionally, ``zdm`` requires two external FRB-related packages:

- `ne2001 <https://github.com/FRBs/ne2001>`_: Galactic electron density model
- `frb <https://github.com/FRBs/FRB>`_: FRB utilities library

Installing from Source
======================

Clone the repository and install in development mode:

.. code-block:: bash

   git clone https://github.com/FRBs/zdm.git
   cd zdm
   pip install -e .

For development with testing tools:

.. code-block:: bash

   pip install -e .[dev]

Installing Dependencies
=======================

The external FRB packages must be installed from GitHub:

.. code-block:: bash

   pip install git+https://github.com/FRBs/ne2001.git#egg=ne2001
   pip install git+https://github.com/FRBs/FRB.git#egg=frb

Full Installation Script
========================

For a complete installation from scratch:

.. code-block:: bash

   # Clone and enter the repository
   git clone https://github.com/FRBs/zdm.git
   cd zdm

   # Install external dependencies
   pip install git+https://github.com/FRBs/ne2001.git#egg=ne2001
   pip install git+https://github.com/FRBs/FRB.git#egg=frb

   # Install zdm in development mode
   pip install -e .[dev]

Verifying Installation
======================

To verify your installation, run the test suite:

.. code-block:: bash

   pytest

Or run a quick import test:

.. code-block:: python

   import zdm
   from zdm import parameters, survey, grid
   print("zdm installed successfully!")

Using tox
=========

For isolated testing environments, use tox:

.. code-block:: bash

   pip install tox
   tox -e test-alldeps

Available tox environments:

- ``test``: Basic test suite
- ``test-alldeps``: Tests with all optional dependencies
- ``test-astropydev``: Tests with development version of astropy
- ``codestyle``: Check code style with pycodestyle

Troubleshooting
===============

ne2001 Installation Issues
--------------------------

If you encounter issues installing ``ne2001``, ensure you have a C compiler available:

.. code-block:: bash

   # On Ubuntu/Debian
   sudo apt-get install build-essential

   # On macOS with Homebrew
   xcode-select --install

Missing frb Package
-------------------

The ``frb`` package is required for DM calculations. If import errors occur, reinstall:

.. code-block:: bash

   pip uninstall frb
   pip install git+https://github.com/FRBs/FRB.git#egg=frb
