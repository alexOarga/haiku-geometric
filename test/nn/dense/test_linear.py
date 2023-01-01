import pytest
import haiku as hk
import jax

from haiku_geometric.nn import Linear
from haiku_geometric.datasets.toy_dataset import ToyGraphDataset


def test_dense_linear():
    args = {
        'out_channels': 8,
    }

    def forward(x, **args):
        module = Linear(**args)
        x = module(x)
        return x

    # Test with edge features
    graph = ToyGraphDataset().data[0]
    nodes, edges, receivers, senders = graph.nodes, graph.edges, graph.receivers, graph.senders
    network = hk.without_apply_rng(hk.transform(forward))
    params_n = network.init(jax.random.PRNGKey(42), nodes, **args)
    out = network.apply(params_n, nodes, **args)
    assert out.shape == (4, 8)