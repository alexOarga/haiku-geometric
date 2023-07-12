
import pytest
import haiku as hk
import jax
import jax.numpy as jnp
import jax.tree_util as tree

from haiku_geometric import batch
from haiku_geometric.nn.aggr.utils import aggregation

from haiku_geometric.nn import MetaLayer
from haiku_geometric.datasets.toy_dataset import ToyGraphDataset

def test_edge_conv():
    
    class EdgeModel(hk.Module):
        def __init__(self, dim):
            super().__init__()
            self.mlp = hk.Sequential([hk.Linear(dim), jax.nn.relu, hk.Linear(dim)])

        def __call__(self, senders_features, receivers_features, edges_features, globals, batch):
            h = jnp.concatenate([senders_features, receivers_features, edges_features], axis=-1)
            return self.mlp(h)

    class NodeModel(hk.Module):
        def __init__(self, dim):
            super().__init__()
            self.aggr = aggregation('mean')
            self.mlp = hk.Sequential([hk.Linear(dim), jax.nn.relu, hk.Linear(dim)])

        def __call__(self, nodes, senders, receivers, edge_attr, globals, batch):
            h = jnp.concatenate([nodes[senders], edge_attr], axis=1)
            messages = self.mlp(h)
            total_num_nodes = tree.tree_leaves(nodes)[0].shape[0]
            return self.aggr(messages, receivers, total_num_nodes)

    class GlobalModel(hk.Module):
        def __init__(self, dim):
            super().__init__()
            self.mlp = hk.Sequential([hk.Linear(dim), jax.nn.relu, hk.Linear(dim)])

        def __call__(self, nodes, senders, receivers, edge_attr, globals, batch):
            num_batches = jnp.max(batch) + 1
            globals = globals.reshape((num_batches, -1))
            return self.mlp(globals)

    def forward(nodes, receivers, senders, edges, globals, batch_idx):
        edge_model = EdgeModel(16)
        node_model = NodeModel(16)
        global_model = GlobalModel(16)
        module = MetaLayer(edge_model, node_model, global_model)
        return module(nodes, receivers, senders, edges, globals, batch_idx)

    # Test with edge features
    graph1 = ToyGraphDataset().data[0]
    graph2 = ToyGraphDataset().data[0]
    graph, batch_idx = batch([graph1, graph2])
    nodes, edges, receivers, senders = graph.nodes, graph.edges, graph.receivers, graph.senders
    globals = jnp.ones((2, 4))  # just a dummy global feature
    network = hk.without_apply_rng(hk.transform(forward))
    params_n = network.init(jax.random.PRNGKey(42), nodes, receivers, senders, edges, globals, batch_idx)
    out_nodes, out_edges, out_globals = network.apply(params_n, nodes, receivers, senders, edges, globals, batch_idx)
    assert out_nodes.shape == (8, 16)
    assert out_edges.shape == (10, 16)
    assert out_globals.shape == (2, 16)
