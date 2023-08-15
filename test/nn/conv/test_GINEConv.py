import pytest
import haiku as hk
import jax

from haiku_geometric.nn import GINEConv
from haiku_geometric.datasets.toy_dataset import ToyGraphDataset

@pytest.mark.parametrize('train_eps', [False, True])
@pytest.mark.parametrize('edge_dim', [1, None])
def test_gine_conv(train_eps, edge_dim):
    args = {
        'train_eps': 8,
    }

    def forward(nodes, receivers, senders, edges, **args):
        nn = hk.nets.MLP((8, 8))
        module = GINEConv(nn, **args)
        return module(nodes, senders, receivers, edges)

    # Test with edge features
    graph = ToyGraphDataset().data[0]
    nodes, edges, receivers, senders = graph.nodes, graph.edges, graph.receivers, graph.senders
    network = hk.without_apply_rng(hk.transform(forward))
    params_n = network.init(jax.random.PRNGKey(42), nodes, senders, receivers, edges, **args)
    out = network.apply(params_n, nodes, senders, receivers, edges, **args)
    assert out.shape == (4, 8)