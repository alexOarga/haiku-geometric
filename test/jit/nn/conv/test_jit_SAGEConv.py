import pytest
import haiku as hk
import jax
import jax.numpy as jnp
import optax

from functools import partial
from haiku_geometric.nn import SAGEConv
from haiku_geometric.datasets.toy_dataset import ToyGraphDataset


@pytest.mark.parametrize('project', [False, True])
@pytest.mark.parametrize('normalize', [False, True])
@pytest.mark.parametrize('root_weight', [False, True])
@pytest.mark.parametrize('bias', [False, True])
def test_sage_conv(project, normalize, root_weight, bias):
    args = {
        'out_channels': 8,
        'aggr': 'max',
        'normalize': normalize,
        'root_weight': root_weight,
        'project': project,
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
        nodes, receivers, senders = graph.nodes, graph.receivers, graph.senders
        module = SAGEConv(*args)
        return module(nodes, senders, receivers)

    graph = ToyGraphDataset().data[0]
    network = hk.without_apply_rng(hk.transform(forward))
    params = network.init(jax.random.PRNGKey(42), graph, tuple(args.values()))

    opt_init, opt_update = optax.adamw(0.01)
    opt_state = opt_init(params)
    params, opt_state = update(params, opt_state, graph, tuple(args.values()))

    out = network.apply(params, graph, tuple(args.values()))
    assert out.shape == (4, 8)