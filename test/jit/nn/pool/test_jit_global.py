import pytest
import haiku as hk
import jax

from functools import partial
from haiku_geometric.nn import global_add_pool, global_mean_pool, global_max_pool
from haiku_geometric.nn import GCNConv
from haiku_geometric.datasets.toy_dataset import ToyGraphDataset


def test_add_pool():
    args = {
        'out_channels': 8,
    }


    def forward(x, senders, receivers, args):
        module = GCNConv(*args)
        x = module(x, senders, receivers)
        x = global_add_pool(x)
        return x

    @partial(jax.jit, static_argnums=(3,))
    def call_model(x, senders, receivers, args):
        out = network.apply(params_n, x, senders, receivers, args)
        return out

    # Test with edge features
    graph = ToyGraphDataset().data[0]
    nodes, edges, receivers, senders = graph.nodes, graph.edges, graph.receivers, graph.senders
    network = hk.without_apply_rng(hk.transform(forward))
    params_n = network.init(jax.random.PRNGKey(42), nodes, receivers, senders, tuple(args.values()))
    out = call_model(nodes, receivers, senders, tuple(args.values()))
    assert out.shape == (1, 8)


def test_mean_pool():
    args = {
        'out_channels': 8,
    }

    def forward(x, senders, receivers, args):
        module = GCNConv(*args)
        x = module(x, senders, receivers)
        x = global_mean_pool(x)
        return x

    @partial(jax.jit, static_argnums=(3,))
    def call_model(x, senders, receivers, args):
        out = network.apply(params_n, x, senders, receivers, args)
        return out

    # Test with edge features
    graph = ToyGraphDataset().data[0]
    nodes, edges, receivers, senders = graph.nodes, graph.edges, graph.receivers, graph.senders
    network = hk.without_apply_rng(hk.transform(forward))
    params_n = network.init(jax.random.PRNGKey(42), nodes, receivers, senders, tuple(args.values()))
    out = call_model(nodes, receivers, senders, tuple(args.values()))
    assert out.shape == (1, 8)


def test_max_pool():
    args = {
        'out_channels': 8,
    }

    def forward(x, senders, receivers, args):
        module = GCNConv(*args)
        x = module(x, senders, receivers)
        x = global_max_pool(x)
        return x

    @partial(jax.jit, static_argnums=(3,))
    def call_model(x, senders, receivers, args):
        out = network.apply(params_n, x, senders, receivers, args)
        return out

    # Test with edge features
    graph = ToyGraphDataset().data[0]
    nodes, edges, receivers, senders = graph.nodes, graph.edges, graph.receivers, graph.senders
    network = hk.without_apply_rng(hk.transform(forward))
    params_n = network.init(jax.random.PRNGKey(42), nodes, receivers, senders, tuple(args.values()))
    out = call_model(nodes, receivers, senders, tuple(args.values()))
    assert out.shape == (1, 8)