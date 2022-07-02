# Configuration file for the Sphinx documentation builder.


# -- Path setup

import os
import sys
sys.path.insert(0,os.path.abspath('../../'))
import tomas



#from pathlib import Path


#from sphinx.application import Sphinx

#HERE = Path(__file__).parent
#sys.path[:0] = [str(HERE.parent)]






# -- General configuration ------------------------------------------------

needs_sphinx = "5.0.2"  # autosummary bugfix



# -- Project information

project = 'TOMAS'
copyright = '2022, Qiuyu Lian'
author = 'Qiuyu Lian'

release = '0.1'
version = '0.1.0'

# -- General configuration

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

templates_path = ['_templates']

# -- Options for HTML output

html_theme = 'sphinx_rtd_theme'

# -- Options for EPUB output
epub_show_urls = 'footnote'
