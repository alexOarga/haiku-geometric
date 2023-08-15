import pytest
import haiku as hk
import jax
import jax.numpy as jnp

from haiku_geometric.nn import PNAConv
from haiku_geometric.datasets.toy_dataset import ToyGraphDataset
from haiku_geometric.utils import degree



@pytest.mark.parametrize('aggrs', ['sum', ['sum', 'mean', 'max', 'min']])
@pytest.mark.parametrize('scaler', ['identity', ['identity', 'amplification', 'attenuation', 'linear', 'inverse_linear']])
def test_pna_conv(aggrs, scaler):

    train_dataset = [ToyGraphDataset().data[0], ToyGraphDataset().data[0], ToyGraphDataset().data[0]]

    # Compute the maximum in-degree in the training data.
    max_degree = -1
    for data in train_dataset:
        total_num_nodes = data.nodes.shape[0]
        d = degree(data.receivers, total_num_nodes)
        max_degree = max(max_degree, int(jnp.max(d)))

    # Compute the in-degree histogram tensor
    deg = jnp.zeros(max_degree + 1)
    for data in train_dataset:
        total_num_nodes = data.nodes.shape[0]
        d = degree(data.receivers, total_num_nodes)
        deg += jnp.bincount(d, minlength=deg.size)

    kwargs = {
        'out_channels': 8,
        'aggregators': aggrs,
        'scalers': scaler,
        'deg': deg,
        'edge_dim': 1,
    }

    def forward(nodes, senders, receivers, edges, **kwargs):
        module = PNAConv(**kwargs)
        return module(nodes, senders, receivers, edges)

    graph = train_dataset[0]
    nodes, edges, receivers, senders = graph.nodes, graph.edges, graph.receivers, graph.senders
    network = hk.without_apply_rng(hk.transform(forward))
    params = network.init(jax.random.PRNGKey(42), 
            nodes, senders, receivers, edges, **kwargs)
    out = network.apply(params, nodes, senders, receivers, edges, **kwargs)
    assert out.shape == (4, 8)

    # Test without edge features
    edges = None
    kwargs['edge_dim'] = None
    network = hk.without_apply_rng(hk.transform(forward))
    params = network.init(jax.random.PRNGKey(42), 
            nodes, senders, receivers, edges, **kwargs)
    out = network.apply(params, nodes, senders, receivers, edges, **kwargs)
    assert out.shape == (4, 8)