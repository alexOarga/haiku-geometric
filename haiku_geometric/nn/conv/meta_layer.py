import haiku as hk
import jax.tree_util as tree
import jraph
import jax.numpy as jnp
from typing import Optional, Tuple
from haiku_geometric.nn.aggr.utils import aggregation
from haiku_geometric.nn.conv.utils import validate_input
from typing import Callable


class MetaLayer(hk.Module):
    r"""A meta layer for building any kind of graph network, inspired by the
    `"Relational Inductive Biases, Deep Learning, and Graph Networks"
    <https://arxiv.org/abs/1806.01261>`_ paper.

    General graph network that takes as input the nodes features :obj:`nodes`,
    the edges features :obj:`edge_attr`, the senders nodes indices :obj:`senders`,
    the receivers nodes indices :obj:`receivers` and the global features of the graph
    :obj:`globals`.

    It returns the updated nodes features :obj:`nodes`, the updated edges features :obj:`edge_attr`
    and the updated global features :obj:`globals`.

    Nodes features are updated after calling the node model :obj:`node_model`, edges features
    are updated after calling the edge model :obj:`edge_model` and global features are updated
    after calling the global model :obj:`global_model`.

    Args:
        edge_model (hk.Module, optional): A neural network that updates its edge features based on
            its source and target nodes features. It receives as input:

                - senders features of shape :obj:`[E, F_N]` where :obj:`E` is the number of edges
                  and :obj:`F_N` the number of input node features.
                - receivers features of shape :obj:`[E, F_N]` where :obj:`E` is the number of edges
                  and :obj:`F_N` the number of input node features.
                - edges features of shape :obj:`[E, F_E]` where :obj:`E` is the number of edges
                  and :obj:`F_E` the number of input edge features.
                - globals features of shape :obj:`[F_G]` for non-batched graphs or shape
                  :obj:`[G * F_G]` for batched graphs, where :obj:`G` is the number of graphs
                  and :obj:`F_G` the shape of the global features.
                - batch indices of shape :obj:`[N]`, where :obj:`N` is the number of nodes. This
                  array indicates to which graph each node belongs to.

        node_model (hk.Module, optional): A neural network that updates the nodes features based on
            the current node features, edge features and global features. It receives as input:

                - nodes features of shape :obj:`[N, F_N]` where :obj:`N` is the number of nodes
                  and :obj:`F_N` the number of input node features.
                - senders features of shape :obj:`[E, F_N]` where :obj:`E` is the number of edges
                  and :obj:`F_N` the number of input node features.
                - receivers features of shape :obj:`[E, F_N]` where :obj:`E` is the number of edges
                  and :obj:`F_N` the number of input node features.
                - edges features of shape :obj:`[E, F_E]` where :obj:`E` is the number of edges
                  and :obj:`F_E` the number of input edge features.
                - globals features of shape :obj:`[F_G]` for non-batched graphs or shape
                  :obj:`[G * F_G]` for batched graphs, where :obj:`G` is the number of graphs
                  and :obj:`F_G` the shape of the global features.
                - batch indices of shape :obj:`[N]`, where :obj:`N` is the number of nodes. This
                  array indicates to which graph each node belongs to.

        global_model (hk.Module, optional): A neural network that updates a graph global features
            based on the current nodes features, edges features and global features. It receives as
            input:

                - nodes features of shape :obj:`[N, F_N]` where :obj:`N` is the number of nodes
                  and :obj:`F_N` the number of input node features.
                - senders features of shape :obj:`[E, F_N]` where :obj:`E` is the number of edges
                  and :obj:`F_N` the number of input node features.
                - receivers features of shape :obj:`[E, F_N]` where :obj:`E` is the number of edges
                  and :obj:`F_N` the number of input node features.
                - edges features of shape :obj:`[E, F_E]` where :obj:`E` is the number of edges
                  and :obj:`F_E` the number of input edge features.
                - globals features of shape :obj:`[F_G]` for non-batched graphs or shape
                  :obj:`[G * F_G]` for batched graphs, where :obj:`G` is the number of graphs
                  and :obj:`F_G` the shape of the global features.
                - batch indices of shape :obj:`[N]`, where :obj:`N` is the number of nodes. This
                  array indicates to which graph each node belongs to.


    Returns:
        :obj:`Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]`:
            - The updated nodes features if :obj:`node_model` is not :obj:`None`.
            - The updated edges features if :obj:`edge_model` is not :obj:`None`.
            - The updated globals features if :obj:`global_model` is not :obj:`None`.


    **Examples**::

        import haiku as hk
        from haiku_geometric.nn.aggr.utils import aggregation

        class EdgeModel(hk.Module):
            def __init__(self):
                super().__init__()
                self.mlp = hk.Sequential([hk.Linear(...), jax.nn.relu, hk.Linear(...)])

            def __call__(self, senders_features, receivers_features, edges_features, globals, batch, num_nodes=None):
                h = jnp.concatenate([senders_features, receivers_features, edges_features], axis=-1)
                return self.mlp(h)

        class NodeModel(hk.Module):
            def __init__(self):
                super().__init__()
                self.aggr = aggregation('mean')
                self.mlp = hk.Sequential([hk.Linear(...), jax.nn.relu, hk.Linear(...)])

            def __call__(self, nodes, senders, receivers, edge_attr, globals, batch, num_nodes=None):
                h = jnp.concatenate([nodes[senders], edge_attr], axis=1)
                messages = self.mlp(h)
                total_num_nodes = tree.tree_leaves(nodes)[0].shape[0]
                return self.aggr(messages, receivers, total_num_nodes)

        class GlobalModel(hk.Module):
            def __init__(self):
                super().__init__()
                self.mlp = hk.Sequential([hk.Linear(...), jax.nn.relu, hk.Linear(...)])

            def __call__(self, nodes, senders, receivers, edge_attr, globals, batch, num_nodes=None):
                return self.mlp(globals)
    """

    def __init__(
        self,
        edge_model: Optional[hk.Module] = None,
        node_model: Optional[hk.Module] = None,
        global_model: Optional[hk.Module] = None,
    ):
        super().__init__()
        self.edge_model = edge_model
        self.node_model = node_model
        self.global_model = global_model


    def __call__(self,
                 nodes: jnp.ndarray,
                 senders: jnp.ndarray,
                 receivers: jnp.ndarray,
                 edge_attr: Optional[jnp.ndarray] = None,
                 globals: Optional[jnp.ndarray] = None,
                 batch: Optional[jnp.ndarray] = None,
                 num_nodes: Optional[int] = None,
        ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """"""
        if self.edge_model is not None:
            edge_attr = self.edge_model(nodes[senders], nodes[receivers], edge_attr, globals, batch, num_nodes=num_nodes)
        if self.node_model is not None:
            nodes = self.node_model(nodes, senders, receivers, edge_attr, globals, batch, num_nodes=num_nodes)
        if self.global_model is not None:
            globals = self.global_model(nodes, senders, receivers, edge_attr, globals, batch, num_nodes=num_nodes)
        return nodes, edge_attr, globals


