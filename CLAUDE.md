# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

The `zdm` package is a Python library for Fast Radio Burst (FRB) redshift-dispersion measure (z-DM) calculations. It implements statistical models for analyzing FRB populations, computing likelihoods, and constraining cosmological and FRB population parameters.

## Build and Installation

```bash
# Install in development mode
pip install -e .[dev]

# Install dependencies for external FRB packages
pip install git+https://github.com/FRBs/ne2001.git#egg=ne2001
pip install git+https://github.com/FRBs/FRB.git#egg=frb
```

## Running Tests

```bash
# Run all tests with pytest
pytest

# Run tests with tox
tox -e test-alldeps

# Run a single test file
pytest zdm/tests/test_energetics.py

# Run a specific test
pytest zdm/tests/test_energetics.py::test_init_gamma
```

## Architecture

### Core Classes

- **`parameters.State`** ([parameters.py](zdm/parameters.py)): Central configuration object containing all model parameters organized in dataclasses (`CosmoParams`, `FRBDemoParams`, `EnergyParams`, `HostParams`, `MWParams`, etc.). Use `state.update_param_dict(vparams)` to update parameters.

- **`survey.Survey`** ([survey.py](zdm/survey.py)): Represents an FRB survey with instrument properties, beam patterns, and detected FRB data. Handles efficiency calculations, DM processing, and FRB metadata.

- **`grid.Grid`** ([grid.py](zdm/grid.py)): Core computational class that builds 2D z-DM grids representing probability distributions of FRB detection rates. Takes a Survey and State, computes thresholds, detection efficiencies, and expected rates across the z-DM plane.

- **`repeat_grid.repeat_Grid`** ([repeat_grid.py](zdm/repeat_grid.py)): Extension of Grid for handling repeating FRBs with additional rate parameters.

### Key Modules

- **`cosmology`** ([cosmology.py](zdm/cosmology.py)): Lambda CDM cosmology calculations including distance measures and volume elements.

- **`pcosmic`** ([pcosmic.py](zdm/pcosmic.py)): Computes p(DM|z) - the probability distribution of cosmic DM given redshift, implementing the Macquart relation.

- **`energetics`** ([energetics.py](zdm/energetics.py)): FRB luminosity function implementations using incomplete gamma functions with spline interpolation for efficiency.

- **`iteration`** ([iteration.py](zdm/iteration.py)): Likelihood calculation routines. Key function `get_log_likelihood(grid, survey)` computes total log-likelihood for a grid given survey data.

- **`loading`** ([loading.py](zdm/loading.py)): High-level functions to load surveys and initialize states. `set_state()` creates default parameter configurations, `load_CHIME()` loads CHIME survey data.

- **`MCMC`** ([MCMC.py](zdm/MCMC.py)): MCMC parameter estimation using emcee. `calc_log_posterior()` is the main posterior evaluation function.

### Data Flow

1. Initialize a `State` with desired parameters via `loading.set_state()` or `parameters.State()`
2. Load survey data via `Survey` class pointing to survey files in `zdm/data/Surveys/`
3. Build a `Grid` from the survey and state
4. Compute likelihoods using `iteration.get_log_likelihood(grid, survey)`
5. For parameter estimation, use MCMC functions to explore parameter space

### Data Files

Survey data and beam patterns are stored in `zdm/data/`:
- `Surveys/`: Survey definition files (ECSV format) for CHIME, ASKAP, Parkes, etc.
- `BeamData/`: Beam response histograms for different telescopes
- `Cube/`: Precomputed parameter grids

## Console Scripts

```bash
zdm_build_cube   # Build parameter cubes (zdm/scripts/build_cube.py)
zdm_pzdm         # Compute p(z|DM) distributions (zdm/scripts/pzdm.py)
```

## Dependencies

Key external dependencies:
- `frb`: FRB utilities from the FRBs organization
- `ne2001`: Galactic electron density model
- `emcee`: MCMC sampling
- `astropy`: Cosmology and astronomical utilities
