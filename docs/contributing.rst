.. _contributing:

============
Contributing
============

We welcome contributions to ``zdm``! This document provides guidelines
for contributing to the project.

Getting Started
===============

1. Fork the repository on GitHub
2. Clone your fork locally:

   .. code-block:: bash

      git clone https://github.com/YOUR_USERNAME/zdm.git
      cd zdm

3. Install in development mode:

   .. code-block:: bash

      pip install -e .[dev]

4. Create a branch for your changes:

   .. code-block:: bash

      git checkout -b my-feature-branch

Running Tests
=============

Before submitting changes, ensure all tests pass:

.. code-block:: bash

   # Run full test suite
   pytest

   # Run with coverage
   pytest --cov=zdm

   # Run specific test file
   pytest zdm/tests/test_energetics.py

Code Style
==========

We follow PEP 8 guidelines. Check your code with:

.. code-block:: bash

   tox -e codestyle

Submitting Changes
==================

1. Commit your changes with descriptive messages
2. Push to your fork
3. Open a Pull Request against the main repository
4. Ensure CI tests pass

Reporting Issues
================

Please report issues on the GitHub issue tracker:
https://github.com/FRBs/zdm/issues

Include:

- A clear description of the problem
- Steps to reproduce
- Expected vs actual behavior
- Python and package versions
