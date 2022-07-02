# Configuration file for the Sphinx documentation builder.


# -- Path setup

import os
import sys

sys.path.insert(0,os.path.abspath('../../'))
import tomas


# -- Project information

project = 'TOMAS'
copyright = '2022, Qiuyu Lian'
author = 'Qiuyu Lian'

release = '0.1'
version = '0.1.0'


# -- General configuration ------------------------------------------------

nitpicky = True  # If true, Sphinx will warn about all references where the target cannot be found. 
needs_sphinx = '2.0'  

# default settings
templates_path = ["_templates"]
#html_static_path = ["_static"]
source_suffix = ".rst"
master_doc = "index"
default_role = "literal"
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
pygments_style = "sphinx"

extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    #'sphinx_autodoc_annotation',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
]

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
}
intersphinx_disabled_domains = ['std']



# -- Options for HTML output

html_theme = 'sphinx_rtd_theme'



# Generate the API documentation when building

autosummary_generate = True
autodoc_member_order = 'bysource'


