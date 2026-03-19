# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
from unittest.mock import MagicMock

# Add the project root to the path for autodoc
sys.path.insert(0, os.path.abspath('..'))

# Inject mocks for GitHub-only packages directly into sys.modules.
# This must be done before Sphinx loads any extensions, because
# sphinx-automodapi's automodsumm handler fires at builder-inited —
# earlier than autodoc_mock_imports takes effect — and will fail with
# "No module named X" if the package is not installed.
#
# Packages mocked here:
#   astropath: pip install git+https://github.com/FRBs/astropath.git
#   frb:       pip install git+https://github.com/FRBs/FRB.git
#   ne2001:    pip install git+https://github.com/FRBs/ne2001.git
_MOCK_MODULES = [
    'astropath',
    'astropath.priors',
    'astropath.path',
    'frb',
    'frb.frb',
    'frb.associate',
    'ne2001',
]
for _mod in _MOCK_MODULES:
    sys.modules[_mod] = MagicMock()

# -- Project information -----------------------------------------------------
project = 'zdm'
copyright = '2024, Clancy James and contributors'
author = 'Clancy James'

# The full version, including alpha/beta/rc tags
release = '0.1'
version = '0.1'

# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'sphinx_rtd_theme',
    'sphinx_automodapi.automodapi',
    'sphinx_automodapi.smart_resolver',
]

# Autosummary settings
autosummary_generate = True
numpydoc_show_class_members = False

# Napoleon settings for Google/NumPy style docstrings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_type_aliases = None

# Intersphinx mapping
intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/', None),
    'astropy': ('https://docs.astropy.org/en/stable/', None),
    'matplotlib': ('https://matplotlib.org/stable/', None),
}

# Templates path
templates_path = ['_templates']

# List of patterns to exclude
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', 'nb/.ipynb_checkpoints']

# The suffix of source filenames
source_suffix = '.rst'

# The master toctree document
master_doc = 'index'

# -- Options for HTML output -------------------------------------------------
html_theme = 'sphinx_rtd_theme'

html_theme_options = {
    'canonical_url': '',
    'analytics_id': '',
    'logo_only': False,
    'display_version': True,
    'prev_next_buttons_location': 'bottom',
    'style_external_links': False,
    'style_nav_header_background': '#2980B9',
    # Toc options
    'collapse_navigation': True,
    'sticky_navigation': True,
    'navigation_depth': 4,
    'includehidden': True,
    'titles_only': False
}

html_static_path = ['_static']

# -- Options for LaTeX output ------------------------------------------------
latex_elements = {
    'papersize': 'letterpaper',
    'pointsize': '10pt',
}

# Autodoc settings
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True,
    'exclude-members': '__weakref__'
}

# Mock imports for modules that may not be available during doc build
# These packages can only be installed from GitHub and are not available
# on ReadTheDocs or in the standard docs build environment:
#   ne2001:    pip install git+https://github.com/FRBs/ne2001.git
#   frb:       pip install git+https://github.com/FRBs/FRB.git
#   astropath: pip install git+https://github.com/FRBs/astropath.git
autodoc_mock_imports = [
    'ne2001',
    'frb',
    'astropath',
]
