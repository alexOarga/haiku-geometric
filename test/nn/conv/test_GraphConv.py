import pytest
import haiku as hk
import jax

from haiku_geometric.nn import GraphConv
from haiku_geometric.datasets.toy_dataset import ToyGraphDataset

@pytest.mark.parametrize('bias', [False, True])
def test_graph_conv(bias):
    args = {
        'out_channels': 8,
        'bias': bias,
        'aggr': 'add',
    }

    def forward(nodes, receivers, senders, **args):
        module = GraphConv(**args)
        return module(nodes, receivers, senders)

    # Test with edge features
    graph = ToyGraphDataset().data[0]
    nodes, edges, receivers, senders = graph.nodes, graph.edges, graph.receivers, graph.senders
    network = hk.without_apply_rng(hk.transform(forward))
    params_n = network.init(jax.random.PRNGKey(42), nodes, senders, receivers, **args)
    out = network.apply(params_n, nodes, senders, receivers, **args)
    assert out.shape == (4, 8)

    # Test without edge features
    graph = ToyGraphDataset().data[0]
    graph = graph._replace(edges=None)
    graph = graph._replace(n_edge=None)
    nodes, edges, receivers, senders = graph.nodes, graph.edges, graph.receivers, graph.senders
    network = hk.without_apply_rng(hk.transform(forward))
    params_n = network.init(jax.random.PRNGKey(42), nodes, senders, receivers, **args)
    out = network.apply(params_n, nodes, senders, receivers, **args)
    assert out.shape == (4, 8)