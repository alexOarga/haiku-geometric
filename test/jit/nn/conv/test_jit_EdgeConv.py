import pytest
import haiku as hk
import jax
import jax.numpy as jnp
import optax

from haiku_geometric.nn import EdgeConv
from haiku_geometric.datasets.toy_dataset import ToyGraphDataset

def test_edge_conv():

    def forward(graph):
        nodes, edges, receivers, senders = graph.nodes, graph.edges, graph.receivers, graph.senders
        nn = hk.nets.MLP((8, 8))
        module = EdgeConv(nn)
        return module(nodes, senders, receivers)

    @jax.jit
    def prediction_loss(params, graph):
        logits = model.apply(params=params, graph=graph)
        labels = jnp.ones(logits.shape)  # dummy labels
        loss = jnp.sum(optax.softmax_cross_entropy(logits, labels))
        return loss

    @jax.jit
    def update(params, opt_state, graph):
        g = jax.grad(prediction_loss)(params, graph)
        updates, opt_state = opt_update(g, opt_state, params=params)
        return optax.apply_updates(params, updates), opt_state

    # Test with edge features
    graph = ToyGraphDataset().data[0]

    model = hk.without_apply_rng(hk.transform(forward))
    params_n = model.init(jax.random.PRNGKey(42), graph)

    opt_init, opt_update = optax.adamw(0.01)
    opt_state = opt_init(params_n)
    params_n, opt_state = update(params_n, opt_state, graph)

    out = model.apply(params_n, graph)
    assert out.shape == (4, 8)