import pytest
import haiku as hk
import jax

from haiku_geometric.nn import SAGEConv
from haiku_geometric.datasets.toy_dataset import ToyGraphDataset


@pytest.mark.parametrize('project', [False, True])
@pytest.mark.parametrize('normalize', [False, True])
@pytest.mark.parametrize('root_weight', [False, True])
@pytest.mark.parametrize('bias', [False, True])
def test_sage_conv(project, normalize, root_weight, bias):
    args = {
        'out_channels': 8,
        'project': project,
        'normalize': normalize,
        'root_weight': root_weight,
        'bias': bias,
        'aggr': 'max'
    }

    def forward(nodes, receivers, senders, **args):
        module = SAGEConv(**args)
        return module(nodes, senders, receivers)

    graph = ToyGraphDataset().data[0]
    nodes, edges, receivers, senders = graph.nodes, graph.edges, graph.receivers, graph.senders
    network = hk.without_apply_rng(hk.transform(forward))
    params = network.init(jax.random.PRNGKey(42), nodes, senders, receivers, **args)
    out = network.apply(params, nodes, senders, receivers, **args)
    assert out.shape == (4, 8)