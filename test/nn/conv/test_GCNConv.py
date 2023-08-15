import pytest
import haiku as hk
import jax

from haiku_geometric.nn import GCNConv
from haiku_geometric.datasets.toy_dataset import ToyGraphDataset

@pytest.mark.parametrize('improved', [False, True])
@pytest.mark.parametrize('add_self_loops, normalize', [
    (False, True),
    (False, False),
    (True, True),
    (True, False),
])
@pytest.mark.parametrize('bias', [False, True])
def test_gconv(improved, add_self_loops, normalize, bias):
    args = {
        'out_channels': 8,
        'improved': improved,
        'add_self_loops': add_self_loops,
        'normalize': normalize,
        'bias': bias,
    }

    def forward(x, senders, receivers, **args):
        module = GCNConv(**args)
        return module(x, senders, receivers)

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