import pytest
import haiku as hk
import jax
import jax.numpy as jnp
import optax

from functools import partial
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
        'aggr': "add",
        'skip_linear': skip_linear,
        'directed_msg': directed_msg,
        'heads': heads,
        'attention': attention,
        'attention_type': attention_type,
        'l2_normalize': l2_normalize,
        'bias': bias
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
        module = GeneralConv(*args)
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