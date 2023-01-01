import pytest
import haiku as hk
import jax

from haiku_geometric.nn import GeneralConv
from haiku_geometric.datasets.toy_dataset import ToyGraphDataset

@pytest.mark.parametrize('aggr', ['add', 'mean', 'max', 'min'])
def test_general_conv(aggr):
    args = {
        'out_channels': 8,
        'in_edge_channels': 1,
        'aggr': aggr,
    }

    def forward(nodes, receivers, senders, edges, **args):
        module = GeneralConv(**args)
        return module(nodes, receivers, senders, edges)

    # Test with edge features
    graph = ToyGraphDataset().data[0]
    nodes, edges, receivers, senders = graph.nodes, graph.edges, graph.receivers, graph.senders
    network = hk.without_apply_rng(hk.transform(forward))
    params_n = network.init(jax.random.PRNGKey(42), nodes, receivers, senders, edges, **args)
    out = network.apply(params_n, nodes, receivers, senders, edges, **args)
    assert out.shape == (4, 8)