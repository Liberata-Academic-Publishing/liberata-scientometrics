# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information


project = 'Liberata Scientometrics'
copyright = '2025, Liberata'
author = 'Hanlin Wang, Arjun Saha Choudhury, Derek Wang, Anshuman Sabath, Aarsh Roongta, Clayton Knittel'

import os
import sys
sys.path.insert(0, os.path.abspath('../../src'))   # ensure package importable

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',    # support NumPy/Google style docstrings
    'sphinx.ext.viewcode',
    'sphinx.ext.autosummary',
]

autosummary_generate = True
autodoc_default_options = {
    'members': True,
    'undoc-members': False,
    'show-inheritance': True,
}

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'pydata_sphinx_theme'
html_static_path = ['_static']
html_css_files = ['custom.css']
html_baseurl = 'https://www.liberata.info/'

html_theme_options = {
    "github_url": "https://github.com/Liberata-Academic-Publishing/liberata-metrics",
    "show_toc_level": 2,
    "navigation_with_keys": True,
}
