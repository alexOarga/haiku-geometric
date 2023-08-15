import pytest
import haiku as hk
import jax

from haiku_geometric.nn import GINConv
from haiku_geometric.datasets.toy_dataset import ToyGraphDataset

@pytest.mark.parametrize('train_eps', [False, True])
def test_gin_conv(train_eps):
    args = {
        'train_eps': 8,
    }

    def forward(nodes, receivers, senders, **args):
        nn = hk.nets.MLP((8, 8))
        module = GINConv(nn, **args)
        return module(nodes, senders, receivers)

    # Test with edge features
    graph = ToyGraphDataset().data[0]
    nodes, edges, receivers, senders = graph.nodes, graph.edges, graph.receivers, graph.senders
    network = hk.without_apply_rng(hk.transform(forward))
    params_n = network.init(jax.random.PRNGKey(42), nodes, senders, receivers, **args)
    out = network.apply(params_n, nodes, senders, receivers, **args)
    assert out.shape == (4, 8)