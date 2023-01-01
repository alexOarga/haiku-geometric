import pytest
import haiku as hk
import jax

from haiku_geometric.nn import GeneralConv
from haiku_geometric.datasets.toy_dataset import ToyGraphDataset

@pytest.mark.parametrize('skip_linear', [False, True])
@pytest.mark.parametrize('directed_msg', [False, True])
@pytest.mark.parametrize('attention, attention_type, heads', [
    (False, 'additive', 1),
    (False, 'additive', 2),
    (True, 'additive', 1),
    (True, 'additive', 2),
    (True, 'dot_product', 1),
    (True, 'dot_product', 2)
])
@pytest.mark.parametrize('l2_normalize', [False, True])
@pytest.mark.parametrize('bias', [False, True])
def test_general_conv(skip_linear, directed_msg, heads, attention, attention_type, l2_normalize, bias):
    args = {
        'out_channels': 8,
        'in_edge_channels': 1,
        'skip_linear': skip_linear,
        'directed_msg': directed_msg,
        'heads': heads,
        'attention': attention,
        'attention_type': attention_type,
        'l2_normalize': l2_normalize,
        'bias': bias
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