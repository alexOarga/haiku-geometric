import pytest
import haiku as hk
import jax
import jax.numpy as jnp
import optax

from functools import partial
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

    # Define params
    out_channels = 8
    aggregators = aggrs
    scalers = scaler
    deg = deg
    edge_dim = 1

    def generate_params(out_channels, deg, edge_dim):
        return {
            'out_channels': out_channels,
            'aggregators': aggregators,
            'scalers': scalers,
            'deg': deg,
            'edge_dim': edge_dim,
        }

    @partial(jax.jit, static_argnums=(2,))
    def prediction_loss(params, graph, out_channels,  deg, edge_dim):
        logits = network.apply(params, graph, out_channels, deg, edge_dim)
        labels = jnp.ones(logits.shape)  # dummy labels
        loss = jnp.sum(optax.softmax_cross_entropy(logits, labels))
        return loss

    @partial(jax.jit, static_argnums=(3,))
    def update(params, opt_state, graph, out_channels, deg, edge_dim):
        g = jax.grad(prediction_loss)(params, graph, out_channels, deg, edge_dim)
        updates, opt_state = opt_update(g, opt_state, params=params)
        return optax.apply_updates(params, updates), opt_state

    def forward(graph, out_channels, deg, edge_dim):
        nodes, receivers, senders, edges = graph.nodes, graph.receivers, graph.senders, graph.edges
        params = generate_params(out_channels, deg, edge_dim)
        module = PNAConv(**params)
        return module(nodes, senders, receivers, edges)


    graph = train_dataset[0]
    network = hk.without_apply_rng(hk.transform(forward))
    params = network.init(jax.random.PRNGKey(42), graph, out_channels, deg, edge_dim)

    opt_init, opt_update = optax.adamw(0.01)
    opt_state = opt_init(params)
    params, opt_state = update(params, opt_state, graph, out_channels, deg, edge_dim)

    out = network.apply(params, graph, out_channels, deg, edge_dim)
    assert out.shape == (4, 8)

    '''
    # Test without edge features
    args['edge_dim'] = None
    network = hk.without_apply_rng(hk.transform(forward))
    params = network.init(jax.random.PRNGKey(42), graph, tuple(args.values()))

    opt_init, opt_update = optax.adamw(0.01)
    opt_state = opt_init(params)
    params, opt_state = update(params, opt_state, graph, tuple(args.values()))

    out = network.apply(params, graph, tuple(args.values()))
    assert out.shape == (4, 8)
    '''