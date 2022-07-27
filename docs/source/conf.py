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
needs_sphinx = '5.0'  

# default settings
# templates_path = ["_templates"]
# html_static_path = ["_static"]
source_suffix = ".rst"
master_doc = "index"
default_role = "literal"
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "**.ipynb_checkpoints"]
pygments_style = "sphinx"

extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    #'sphinx_autodoc_annotation',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
    'sphinx_autodoc_typehints',  # needs to be after napoleon
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
    'nbsphinx',  # Select nbsphinx and, if needed, other Sphinx extensions:
    #'sphinxcontrib.bibtex',  # for bibliographic references
    #'sphinxcontrib.rsvgconverter',  # for SVG->PDF conversion in LaTeX output
    #'sphinx_gallery.load_style',  # load CSS for gallery (needs SG >= 0.6)
    #'sphinx_last_updated_by_git',  # get "last updated" from Git
    #'sphinx_codeautolink',  # automatic links from code to documentation
    #'sphinx.ext.intersphinx',  # links to other Sphinx projects (e.g. NumPy)
]


intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
}
intersphinx_disabled_domains = ['std']



# -- Options for HTML output

html_theme = 'sphinx_rtd_theme'
#html_theme_options = dict(navigation_depth=4)
html_context = dict(
    display_github=True,  # Integrate GitHub
    github_user="QiuyuLian",  # Username
    github_repo="TOMAS",  # Repo name
    github_version="master",  # Version
    conf_py_path="/docs/source/",  # Path in the checkout to the docs root
)
issues_github_path = "{github_user}/{github_repo}".format_map(html_context)
html_show_sphinx = False
html_static_path = ['_static']
html_logo = '_static/img/logo_v2.png'


# Generate the API documentation when building

autosummary_generate = True
autodoc_member_order = 'bysource'
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_use_rtype = True  # having a separate entry generally helps readability
napoleon_use_param = True
napoleon_custom_sections = [('Params', 'Parameters')]

typehints_defaults = 'braces'
html_sourcelink_suffix = ''  # Don't add .txt suffix to source files:














