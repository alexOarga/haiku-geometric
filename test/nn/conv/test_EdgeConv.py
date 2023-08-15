import pytest
import haiku as hk
import jax

from haiku_geometric.nn import EdgeConv
from haiku_geometric.datasets.toy_dataset import ToyGraphDataset

def test_edge_conv():

    def forward(nodes, receivers, senders):
        nn = hk.nets.MLP((8, 8))
        module = EdgeConv(nn)
        return module(nodes, receivers, senders)

    # Test with edge features
    graph = ToyGraphDataset().data[0]
    nodes, edges, receivers, senders = graph.nodes, graph.edges, graph.receivers, graph.senders
    network = hk.without_apply_rng(hk.transform(forward))
    params_n = network.init(jax.random.PRNGKey(42), nodes, senders, receivers)
    out = network.apply(params_n, nodes, senders, receivers)
    assert out.shape == (4, 8)