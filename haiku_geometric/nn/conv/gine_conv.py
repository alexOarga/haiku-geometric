import haiku as hk
import jax.tree_util as tree
import jraph
import jax.numpy as jnp
from typing import Optional, Union
from haiku_geometric.nn.aggr.utils import aggregation
from haiku_geometric.nn.conv.utils import validate_input
from typing import Callable, Optional


class GINEConv(hk.Module):
    r"""
    Graph Isomorphism operator introduced to include edge features from `"Strategies for
    Pre-training Graph Neural Networks" <https://arxiv.org/abs/1905.12265>`_ paper

    The node features are computed as follows:
    
    .. math::
        \mathbf{{h}}_{u}^{k}= \phi\left( (1 + \epsilon) \mathbf{{h}}_{u}^{k-1} + \sum_{v \in \mathcal{N}(u)}  ReLU(\mathbf{{h}}_{v}^{k-1} + \mathbf{e}_{u,v})  \right)

    where :math:`\phi` is a neural network (e.g. a MLP) and
    :math:`\mathbf{e}_{j,i}` are the edge features.

    Args:
        nn (hk.Module): A neural network :math:`\phi` that produces
            output features of shape :obj:`out_channels` defined by the user.
        eps (float, optional): :math:`\epsilon` value.
            (default: :obj:`0.`)
        train_eps (bool, optional): If :obj:`True`, :math:`\epsilon` will be a trainable parameter.
            (default: :obj:`False`)
        edge_dim (int, optional): If :obj:`None`, edge and node features shapes
            are expected to match. Otherwise, the edge features are linearly transformed
            to match the node features shape.
    """

    def __init__(
            self,
            nn: Callable,
            eps: float = 0.,
            train_eps: bool = False,
            edge_dim: Optional[int] = None,
    ):
        """"""
        super().__init__()
        self.aggr = aggregation('add')
        self.nn = nn
        self.train_eps = train_eps
        self.edge_dim = edge_dim
        if train_eps:
            self.eps = hk.get_parameter("eps", shape=[1, 1], init=hk.initializers.RandomNormal())
        else:
            self.eps = eps

    def __call__(self,
                 nodes: jnp.ndarray,
                 senders: jnp.ndarray,
                 receivers: jnp.ndarray,
                 edges: jnp.ndarray,
                 num_nodes: Optional[int] = None,
                 ) -> jnp.ndarray:
        """"""

        in_channels = nodes.shape[1]
        if self.edge_dim is not None:
            linear = hk.Linear(in_channels, with_bias=False)
            edges = linear(edges)

        messages = nodes[senders] + edges

        if num_nodes is None:
            num_nodes = tree.tree_leaves(nodes)[0].shape[0]

        h = self.aggr(messages, receivers, num_nodes)

        h = h + ((1 + self.eps) * nodes)
        out = self.nn(h)

        return out