import pytest
import haiku as hk
import jax

from haiku_geometric.nn import GatedGraphConv
from haiku_geometric.datasets.toy_dataset import ToyGraphDataset


@pytest.mark.parametrize('num_layers', [1, 4])
def test_gated_graph_conv(num_layers):
    args = {
        'out_channels': 8,
        'num_layers': num_layers,
    }

    def forward(nodes, receivers, senders, edges, **args):
        module = GatedGraphConv(**args)
        return module(nodes, receivers, senders, edges)

    # Test with edge features
    graph = ToyGraphDataset().data[0]
    nodes, edges, receivers, senders = graph.nodes, graph.edges, graph.receivers, graph.senders
    network = hk.without_apply_rng(hk.transform(forward))
    params_n = network.init(jax.random.PRNGKey(42), nodes, senders, receivers, edges, **args)
    out = network.apply(params_n, nodes, senders, receivers, edges, **args)
    assert out.shape == (4, 8)

    # Test without edge features
    graph = ToyGraphDataset().data[0]
    nodes, edges, receivers, senders = graph.nodes, graph.edges, graph.receivers, graph.senders
    graph = graph._replace(edges=None)
    graph = graph._replace(n_edge=None)
    network = hk.without_apply_rng(hk.transform(forward))
    params_n = network.init(jax.random.PRNGKey(42), nodes, senders, receivers, edges, **args)
    out = network.apply(params_n, nodes, senders, receivers, edges, **args)
    assert out.shape == (4, 8)