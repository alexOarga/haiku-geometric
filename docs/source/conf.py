# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
#sys.path.insert(0, os.path.abspath('.'))
import doctest
import inspect
import os
import sys
import typing

#sys.path.insert(0, os.path.abspath('../..'))
import haiku_geometric

# -- Project information -----------------------------------------------------

project = 'haiku-geometric'
copyright = '2022, Alex Oarga'
author = 'Alex Oarga'


# -- General configuration ---------------------------------------------------
master_doc = 'index'

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.doctest',
    'sphinx.ext.inheritance_diagram',
    'sphinx.ext.linkcode',
    'sphinx.ext.napoleon',
    'sphinx_autodoc_typehints',
    'sphinx_book_theme',
    'nbsphinx',
    'IPython.sphinxext.ipython_console_highlighting',
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_book_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

html_theme_options = {
    "repository_url": "https://github.com/alexOarga/haiku-geometric",
    "use_repository_button": True,
}

html_title = "Haiku Geometric"

# ----------------------------------------------------------------------------
import haiku as hk
import jax
import jax.numpy as jnp

# ----------------------------------------------------------------------------

def setup(app):
    def rst_jinja_render(app, _, source):
        rst_context = {'haiku_geometric': haiku_geometric}
        source[0] = app.builder.templates.render_string(source[0], rst_context)
    app.connect('source-read', rst_jinja_render)

mathjax3_config = {'chtml': {'displayAlign': 'left'}}

def linkcode_resolve(domain, info):
  """Resolve a GitHub URL corresponding to Python object."""
  if domain != 'py':
    return None

  try:
    mod = sys.modules[info['module']]
  except ImportError:
    return None

  obj = mod
  try:
    for attr in info['fullname'].split('.'):
      obj = getattr(obj, attr)
  except AttributeError:
    return None
  else:
    obj = inspect.unwrap(obj)

  try:
    filename = inspect.getsourcefile(obj)
  except TypeError:
    return None

  try:
    source, lineno = inspect.getsourcelines(obj)
  except OSError:
    return None
    
  o = os.path.relpath(filename, start=os.path.dirname(hk.__file__))
  s = 'https://github.com/alexOarga/haiku-geometric/blob/main/haiku_geometric/%s#L%d#L%d' % (
      o, lineno, lineno + len(source) - 1)
  return s
