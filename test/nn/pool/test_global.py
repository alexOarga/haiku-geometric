import pytest
import haiku as hk
import jax

from haiku_geometric.nn import global_add_pool, global_mean_pool, global_max_pool
from haiku_geometric.nn import GCNConv
from haiku_geometric.datasets.toy_dataset import ToyGraphDataset


def test_add_pool():
    args = {
        'out_channels': 8,
    }

    def forward(x, senders, receivers, **args):
        module = GCNConv(**args)
        x = module(x, senders, receivers)
        x = global_add_pool(x)
        return x

    # Test with edge features
    graph = ToyGraphDataset().data[0]
    nodes, edges, receivers, senders = graph.nodes, graph.edges, graph.receivers, graph.senders
    network = hk.without_apply_rng(hk.transform(forward))
    params_n = network.init(jax.random.PRNGKey(42), nodes, receivers, senders, **args)
    out = network.apply(params_n, nodes, receivers, senders, **args)
    assert out.shape == (1, 8)


def test_mean_pool():
    args = {
        'out_channels': 8,
    }

    def forward(x, senders, receivers, **args):
        module = GCNConv(**args)
        x = module(x, senders, receivers)
        x = global_mean_pool(x)
        return x

    # Test with edge features
    graph = ToyGraphDataset().data[0]
    nodes, edges, receivers, senders = graph.nodes, graph.edges, graph.receivers, graph.senders
    network = hk.without_apply_rng(hk.transform(forward))
    params_n = network.init(jax.random.PRNGKey(42), nodes, receivers, senders, **args)
    out = network.apply(params_n, nodes, receivers, senders, **args)
    assert out.shape == (1, 8)


def test_max_pool():
    args = {
        'out_channels': 8,
    }

    def forward(x, senders, receivers, **args):
        module = GCNConv(**args)
        x = module(x, senders, receivers)
        x = global_max_pool(x)
        return x

    # Test with edge features
    graph = ToyGraphDataset().data[0]
    nodes, edges, receivers, senders = graph.nodes, graph.edges, graph.receivers, graph.senders
    network = hk.without_apply_rng(hk.transform(forward))
    params_n = network.init(jax.random.PRNGKey(42), nodes, receivers, senders, **args)
    out = network.apply(params_n, nodes, receivers, senders, **args)
    assert out.shape == (1, 8)