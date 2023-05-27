.. haiku-geometric documentation master file.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Haiku Geometric Documentation
=============================

Graph Neural Networks in JAX
----------------------------

Haiku Geometric is a collection of graph neural network (GNN) implementations in `JAX <https://jax.readthedocs.io/en/latest/notebooks/quickstart.html>`_.
It tries to provide **object-oriented** and **easy-to-use** modules for GNNs.

Haiku Geometric is built on top of `Haiku <https://github.com/deepmind/dm-haiku>`_ and `Jraph <https://github.com/deepmind/jraph>`_.
It is deeply inspired by `PyTorch Geometric <https://github.com/pyg-team/pytorch_geometric>`_.
In most cases, Haiku Geometric tries to replicate the API of PyTorch Geometric to allow code sharing between the two.

Haiku Geometric is still under development and I would advise against using it in production.


.. toctree::
   :maxdepth: 2
   :caption: Contents:

Installation
------------
Haiku Geomtric can be installed via Pypi::

    pip install haiku-geometric

Alternatively you can install it from source::

   pip install git+https://github.com/alexOarga/haiku-geometric

   
.. toctree::
   :caption: Contents
   :maxdepth: 1

   notebooks/1_quickstart
   notebooks/creating_dataset
   notebooks/batch_support
   examples

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: API Reference

   modules/nn
   modules/datasets
   modules/models
   modules/posenc
   modules/transforms
   modules/utils



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
