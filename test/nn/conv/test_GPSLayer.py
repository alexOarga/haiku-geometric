import pytest
import haiku as hk
import jax
import jax.numpy as jnp

from haiku_geometric.nn import GPSLayer
from haiku_geometric.datasets.toy_dataset import ToyGraphDataset
from haiku_geometric.utils import degree


@pytest.mark.parametrize('local_gnn_type', ['GAT', 'PNA'])
@pytest.mark.parametrize('global_model_type', ['Transformer', 'Performer'])
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

    kwargs = {
        'dim_h': 8, 
        'local_gnn_type': local_gnn_type,
        'global_model_type': global_model_type,
        'act': jax.nn.relu, 
        'num_heads': 1,
        'pna_degrees': deg, 
        'equivstable_pe': False, 
        'dropout': 0.0, 
        'attn_dropout': 0.0, 
        'layer_norm': layer_norm,
        'batch_norm': batch_norm
    }

    def forward(nodes, receivers, senders, edges, **kwargs):
        module = GPSLayer(**kwargs)
        return module(True, nodes, receivers, senders, edges)

    graph = train_dataset[0]
    nodes, edges, receivers, senders = graph.nodes, graph.edges, graph.receivers, graph.senders
    network = hk.transform_with_state(forward)
    params, state = network.init(jax.random.PRNGKey(42), 
            nodes, receivers, senders, edges, **kwargs)
    out, state = network.apply(params, state, jax.random.PRNGKey(42), nodes, receivers, senders, edges, **kwargs)
    assert out.shape == (4, 8)

