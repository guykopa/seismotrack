"""Sphinx configuration for seismotrack documentation."""

import os
import sys

# Make the seismotrack package importable from docs/
sys.path.insert(0, os.path.abspath(".."))

# -- Project information -------------------------------------------------------
project = "seismotrack"
author = "Florian Kopa"
release = "0.1.0"

# -- General configuration -----------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",       # generates API docs from docstrings
    "sphinx.ext.napoleon",      # supports Google-style docstrings
    "sphinx.ext.viewcode",      # adds [source] links to API pages
    "sphinx.ext.intersphinx",   # cross-links to numpy/scipy docs
]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
}

autodoc_member_order = "bysource"
autodoc_typehints = "description"

templates_path = ["_templates"]
exclude_patterns = ["_build"]

# -- HTML output ---------------------------------------------------------------
html_theme = "sphinx_rtd_theme"
html_static_path = []
