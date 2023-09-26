# Haiku Geometric

[**Overview**](#overview)
| [**Installation**](#installation)
| [**Quickstart**](#quickstart)
| [**Examples**](#examples)
| [**Documentation**](https://haiku-geometric.readthedocs.io/en/latest/)
| [**License**](#license)

[![Documentation Status](https://readthedocs.org/projects/haiku-geometric/badge/?version=latest)](https://haiku-geometric.readthedocs.io/en/latest/?badge=latest)
[![Python application](https://github.com/alexOarga/haiku-geometric/actions/workflows/python-app.yml/badge.svg)](https://github.com/alexOarga/haiku-geometric/actions/workflows/python-app.yml)
![pypi](https://img.shields.io/pypi/v/haiku-geometric)

## Overview

Haiku Geometric is a collection of graph neural networks (GNNs) implemented using [JAX](https://jax.readthedocs.io/en/latest/notebooks/quickstart.html). It tries to provide **object-oriented** and **easy-to-use** modules for GNNs.

Haiku Geometric is built on top of [Haiku](https://github.com/deepmind/dm-haiku) and [Jraph](https://github.com/deepmind/jraph).
It is deeply inspired by [PyTorch Geometric](https://github.com/pyg-team/pytorch_geometric). 
In most cases, Haiku Geometric tries to replicate the API of PyTorch Geometric to allow code sharing between the two.

Haiku Geometric is still under development and I would advise against using it in production.

## Installation

Haiku Geometric can be installed from source:

```bash
pip install git+https://github.com/alexOarga/haiku-geometric.git
```

Alternatively, you can install Haiku Geometric using pip:
```bash
pip install haiku-geometric
```

## Quickstart

For instance, we can create a simple graph convolutional network (GCN) of 2 layers 
as follows:
```python
import jax
import haiku as hk
from haiku_geometric.nn import GCNConv

class GCN(hk.Module):
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(hidden_channels)
        self.conv2 = GCNConv(hidden_channels)
        self.linear = hk.Linear(out_channels)

    def __call__(self, nodes,senders, receivers):
        x = self.conv1(nodes, senders, receivers)
        x = jax.nn.relu(x)
        x = self.conv2(x, senders, receivers)
        x = self.linear(nodes)
        return x

def forward(nodes, senders, receivers):
    gcn = GCN(16, 7)
    return gcn(nodes, senders, receivers)
```

The GNN that we have defined is a Haiku Module. 
To convert our module in a function that can be used with JAX, we transform
it using `hk.transform` as described in the 
[Haiku documentation](https://dm-haiku.readthedocs.io/en/latest/).

```python
model = hk.transform(forward)
model = hk.without_apply_rng(model)
rng = jax.random.PRNGKey(42)
params = model.init(rng, nodes=nodes, senders=senders, receivers=receivers)
```

We can now run a forward pass on the model:
```python
output = model.apply(params=params, nodes=nodes, senders=senders, receivers=receivers)
```

## Documentation

The documentation for Haiku Geometric can be found [here](https://haiku-geometric.readthedocs.io/en/latest/).

## Examples

Haiku Geometric comes with a few examples that showcase the usage of the library.
The following examples are available:

|                                                     | Link                                                                                                                                                                                                                                                    |
|-----------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Quickstart Example                                  | [![Open in Colab](https://img.shields.io/static/v1.svg?logo=google-colab&label=Quickstart&message=Open%20In%20Colab&color=blue)](https://colab.research.google.com/github/alexOarga/haiku-geometric/blob/main/docs/source/notebooks/1_quickstart.ipynb) |
| Graph Convolution Networks with Karate Club dataset | [![Open in Colab](https://img.shields.io/static/v1.svg?logo=google-colab&label=GCNConv&message=Open%20In%20Colab&color=blue)](https://colab.research.google.com/github/alexOarga/haiku-geometric/blob/main/examples/GCNConv_karate_club.ipynb)          |
| Graph Attention Networks with CORA dataset          | [![Open in Colab](https://img.shields.io/static/v1.svg?logo=google-colab&label=GATConv&message=Open%20In%20Colab&color=blue)](https://colab.research.google.com/github/alexOarga/haiku-geometric/blob/main/examples/GATConv_CORA.ipynb)                 |
| TopKPooling and GraphConv with PROTEINS dataset     | [![Open in Colab](https://img.shields.io/static/v1.svg?logo=google-colab&label=TopKPooling&message=Open%20In%20Colab&color=blue)](https://colab.research.google.com/github/alexOarga/haiku-geometric/blob/main/examples/TopKPooling_GraphConv_PROTEINS.ipynb)                 |


## Implemented GNNs modules

Currently, Haiku Geometric includes the following GNN modules:

| Model                                                                                                                     | Description                                                                                                                                    |
|---------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------|
| [GCNConv](https://haiku-geometric.readthedocs.io/en/latest/modules/nn.html#haiku_geometric.nn.conv.GCNConv)               | Graph convolution layer from the [Semi-Supervised Classification with Graph Convolutional Networks](https://arxiv.org/abs/1609.02907) paper.   |
| [GATConv](https://haiku-geometric.readthedocs.io/en/latest/modules/nn.html#haiku_geometric.nn.conv.GATConv)               | Graph attention layer from the [Graph Attention Networks](https://arxiv.org/abs/1710.10903) paper.                                             |
| [SAGEConv](https://haiku-geometric.readthedocs.io/en/latest/modules/nn.html#haiku_geometric.nn.conv.SAGEConv)             | Graph convolution layer from the [Inductive Representation Learning on Large Graphs](https://arxiv.org/abs/1706.02216) paper.                  |
| [GINConv](https://haiku-geometric.readthedocs.io/en/latest/modules/nn.html#haiku_geometric.nn.conv.GINConv)               | Graph isomorphism network layer from the [How Powerful are Graph Neural Networks?](https://arxiv.org/abs/1810.00826) paper.                    |
| [GINEConv](https://haiku-geometric.readthedocs.io/en/latest/modules/nn.html#haiku_geometric.nn.conv.GINEConv)             | Graph isomorphism network layer from the [Strategies for Pre-training Graph Neural Networks](https://arxiv.org/abs/1905.12265) paper.          |
| [GraphConv](https://haiku-geometric.readthedocs.io/en/latest/modules/nn.html#haiku_geometric.nn.conv.GraphConv)           | Graph convolution layer from the [Weisfeiler and Leman Go Neural: Higher-order Graph Neural Networks](https://arxiv.org/abs/1810.02244) paper. |
| [GeneralConv](https://haiku-geometric.readthedocs.io/en/latest/modules/nn.html#haiku_geometric.nn.conv.GeneralConv)       | A general GNN layer adapted from the [Design Space for Graph Neural Networks](https://arxiv.org/abs/2011.08843) paper.                         |
| [GatedGraphConv](https://haiku-geometric.readthedocs.io/en/latest/modules/nn.html#haiku_geometric.nn.conv.GatedGraphConv) | Graph convolution layer from the [Gated Graph Sequence Neural Networks](https://arxiv.org/abs/1511.05493) paper.                               |
| [EdgeConv](https://haiku-geometric.readthedocs.io/en/latest/modules/nn.html#haiku_geometric.nn.conv.EdgeConv)             | Edge convolution layer from the [Dynamic Graph CNN for Learning on Point Clouds](https://arxiv.org/abs/1801.07829) paper.                      |
| [PNAConv](https://haiku-geometric.readthedocs.io/en/latest/modules/nn.html#haiku_geometric.nn.conv.PNAConv)               | Propagation Network layer from the [Principal Neighbourhood Aggregation for Graph Nets](https://arxiv.org/abs/2004.05718) paper.               |
| [MetaLayer](https://haiku-geometric.readthedocs.io/en/latest/modules/nn.html#haiku_geometric.nn.conv.MetaLayer)           | Meta layer from the [Relational Inductive Biases, Deep Learning, and Graph Networks](https://arxiv.org/abs/1806.01261) paper.                  |
| [GPSLayer](https://haiku-geometric.readthedocs.io/en/latest/modules/nn.html#haiku_geometric.nn.conv.GPSLayer)             | Graph layer from the [Recipe for a General, Powerful, Scalable Graph Transformer](https://arxiv.org/abs/2205.12454) paper.                     |

## Implemented positional encodings

The following positional encodings are currently available:

| Model                                                                                                                     | Description                                                                                                                                     |
|---------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------|
| [LaplacianEncoder](https://haiku-geometric.readthedocs.io/en/latest/modules/posenc.html#haiku_geometric.posenc.LaplacianEncoder)    | Laplacian positional encoding from the [Rethinking Graph Transformers with Spectral Attention](https://arxiv.org/pdf/2106.03893) paper.         |
| [MagLaplacianEncoder](https://haiku-geometric.readthedocs.io/en/latest/modules/posenc.html#haiku_geometric.posenc.MagLaplacianEncoder)  | Magnetic Laplacian positional encoding from the [Transformers Meet Directed Graphs](https://arxiv.org/pdf/2302.00049) paper. |

## Issues

If you encounter any issue, please [open an issue](https://github.com/alexOarga/haiku-geometric/issues/new).

## Running tests

Haiku Geometric can be tested using `pytest` by running the following command:

```bash
python -m pytest test/
```

## License

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
