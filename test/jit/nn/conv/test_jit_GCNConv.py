import pytest
import haiku as hk
import jax
import jax.numpy as jnp
import optax

from functools import partial
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

    @partial(jax.jit, static_argnums=(2,))
    def prediction_loss(params, graph, args):
        logits = network.apply(params=params, graph=graph, args=args)
        labels = jnp.ones(logits.shape)  # dummy labels
        loss = jnp.sum(optax.softmax_cross_entropy(logits, labels))
        return loss

    @partial(jax.jit, static_argnums=(3,))
    def update(params, opt_state, graph, args):
        g = jax.grad(prediction_loss)(params, graph, args)
        updates, opt_state = opt_update(g, opt_state, params=params)
        return optax.apply_updates(params, updates), opt_state

    def forward(graph, args):
        nodes, edges, receivers, senders = graph.nodes, graph.edges, graph.receivers, graph.senders
        module = GCNConv(*args)
        return module(nodes, senders, receivers, edges)

    # Test with edge features
    graph = ToyGraphDataset().data[0]
    network = hk.without_apply_rng(hk.transform(forward))
    params_n = network.init(jax.random.PRNGKey(42), graph, tuple(args.values()))

    opt_init, opt_update = optax.adamw(0.01)
    opt_state = opt_init(params_n)
    params_n, opt_state = update(params_n, opt_state, graph, tuple(args.values()))

    out = network.apply(params_n, graph, tuple(args.values()))
    assert out.shape == (4, 8)

    # Test without edge features
    graph = ToyGraphDataset().data[0]
    graph = graph._replace(edges=None)
    graph = graph._replace(n_edge=None)
    network = hk.without_apply_rng(hk.transform(forward))
    params_n = network.init(jax.random.PRNGKey(42), graph, tuple(args.values()))

    opt_init, opt_update = optax.adamw(0.01)
    opt_state = opt_init(params_n)
    params_n, opt_state = update(params_n, opt_state, graph, tuple(args.values()))

    out = network.apply(params_n, graph, tuple(args.values()))
    assert out.shape == (4, 8)