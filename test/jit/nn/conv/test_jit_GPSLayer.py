import pytest
import haiku as hk
import jax
import jax.numpy as jnp
import optax

from functools import partial
from haiku_geometric.nn import GPSLayer
from haiku_geometric.datasets.toy_dataset import ToyGraphDataset
from haiku_geometric.utils import degree


@pytest.mark.parametrize('local_gnn_type', ['GAT', 'PNA'])
@pytest.mark.parametrize('global_model_type', ['Transformer'])
@pytest.mark.parametrize('layer_norm, batch_norm', [
    (True, False), (False, True)    
])
def test_pna_conv(local_gnn_type, global_model_type,
                layer_norm, batch_norm):

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


    dim_h = 8
    num_heads = 1

    def generate_params(dim_h, num_heads, deg):
        return {
            'dim_h': dim_h,
            'local_gnn_type': local_gnn_type,
            'global_model_type': global_model_type,
            'act': jax.nn.relu,
            'num_heads': num_heads,
            'pna_degrees': deg,
            'equivstable_pe': False,
            'dropout': 0.0,
            'attn_dropout': 0.0,
            'layer_norm': layer_norm,
            'batch_norm': batch_norm
        }


    def forward(graph, dim_h, num_heads, deg):
        nodes, receivers, senders, edges = graph.nodes, graph.receivers, graph.senders, graph.edges
        parms = generate_params(dim_h, num_heads, deg)
        module = GPSLayer(**parms)
        return module(True, nodes, senders, receivers, edges)

    @partial(jax.jit, static_argnums=(4,5))
    def call_model(params_n, opt_state, rgn, graph, dim_h, num_heads, deg):
        return network.apply(params_n, opt_state, rgn, graph, dim_h, num_heads, deg)

    graph = train_dataset[0]
    network = hk.transform_with_state(forward)
    rng = jax.random.PRNGKey(42)
    params_n, opt_state = network.init(rng, graph, dim_h, num_heads, deg)
    out, _ = call_model(params_n, opt_state, rng, graph, dim_h, num_heads, deg)
    assert out.shape == (4, 8)

