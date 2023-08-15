import haiku as hk
import jraph
import jax
import jax.numpy as jnp
import jax.tree_util as tree

from jraph._src.utils import segment_sum, segment_mean, segment_max
from typing import Optional, Union

from haiku_geometric.nn.conv.utils import validate_input
from haiku_geometric.nn.aggr.utils import aggregation


class GCNConv(hk.Module):
    r"""The graph convolutional operator from the `"Semi-supervised
    Classification with Graph Convolutional Networks"
    <https://arxiv.org/abs/1609.02907>`_ paper

    .. math::
        H^{(l+1)} = \tilde{D}^{-1/2} \tilde{A} \tilde{D}^{-1/2} H^{(l)} W

    where :math:`\tilde{A} = A + I_N` is the adjacency matrix with added self loops :math:`I_N`.
    and :math:`\tilde{D}_ii = \sum_j \tilde{A}_ij`.

    The node-wise formulation is given by:

    .. math::
        \mathbf{h}_u = W^{\top} \sum_{v \in
        \mathcal{N}(u) \cup \{ u \}} \frac{e_{v,u}}{\sqrt{\hat{d}_u
        \hat{d}_v}} \mathbf{h}_v

    where :math:`e_{v,u}` is the edge weight and :math:`\hat{d}_u = 1 + \sum_{v \in \mathcal{N}(u)} e_{v,u}`

    Args:
        out_channels (int): Size of the output features of each node.
        improved (bool, optional): If :obj:`True`, then
            :math:`\mathbf{\hat{A}}` is computed as :math:`\mathbf{A} + 2\mathbf{I}`.
            (default: :obj:`False`)
        cached (bool, optional): If :obj:`True`, the value :math:`\tilde{D}^{-1/2} \tilde{A} \tilde{D}^{-1/2}` on first execution, and will use the
            is cached and used in further executions.
            This parameter should only be set to :obj:`True` in transductive
            learning scenarios.
            (default: :obj:`False`)
        add_self_loops (bool, optional): If :obj:`True`, will add
            a self-loop for each node of the graph.
            (default: :obj:`True`)
        normalize (bool, optional): Whether to compute and apply
            the symmetric normalization.
            (default: :obj:`True`)
        bias (bool, optional): If :obj:`True`, the layer will add
            an additive bias to the output.
            (default: :obj:`True`)
    """

    def __init__(
            self,
            out_channels: int,
            improved: bool = False,
            cached: bool = False,
            add_self_loops: bool = True,
            normalize: bool = True,
            bias: bool = True,
            aggr: Optional[str] = "add",
    ):
        """"""
        super().__init__()
        # Initialize parameters
        self.out_channels = out_channels
        self.improved = improved
        self.cached = cached
        self.add_self_loops = add_self_loops
        self.normalize = normalize

        self.linear = hk.Linear(out_channels, with_bias=False)
        if bias:
            self.bias = hk.get_parameter("bias", shape=[out_channels], init=hk.initializers.TruncatedNormal())
        else:
            self.bias = None
        self.aggr = aggregation(aggr)

    def __call__(self,
                 nodes: jnp.ndarray,
                 senders: jnp.ndarray,
                 receivers: jnp.ndarray,
                 edges: Optional[jnp.ndarray] = None,
                 num_nodes: Optional[int] = None,
                 ) -> jnp.ndarray:
        """"""
        # forward pass

        nodes = self.linear(nodes)

        if edges is not None and edges.shape[-1] > 1:
            raise ValueError("GCNConv does not allow edges with features of dim " + \
                             "greater than 1.")
        if edges is None:
            num_edges = receivers.shape[0]
            edges = jnp.ones((num_edges, 1))

        if num_nodes is None:
            num_nodes = tree.tree_leaves(nodes)[0].shape[0]

        if self.add_self_loops:
            conv_receivers = jnp.concatenate((receivers, jnp.arange(num_nodes)),
                                             axis=0)
            conv_senders = jnp.concatenate((senders, jnp.arange(num_nodes)),
                                           axis=0)
            fill_value = 2. if self.improved else 1.
            fill_self_edges = jnp.ones((num_nodes, 1)) * fill_value
            edges = jnp.concatenate((edges, fill_self_edges),
                                    axis=0)
        else:
            conv_senders = senders
            conv_receivers = receivers

        if self.normalize:

            if self.cached and hasattr(self, 'norm_vals'):
                norm_vals = self.norm_vals
            else:
                d = edges + jnp.ones_like(conv_senders).reshape(-1, 1)
                d = d.reshape(-1)
                count_edges = lambda x: segment_sum(
                    d, x, num_nodes)
                sender_degree = count_edges(conv_senders)
                receiver_degree = count_edges(conv_receivers)

                di = jax.lax.rsqrt(jnp.maximum(sender_degree, 1.0))[:, None]
                dj = jax.lax.rsqrt(jnp.maximum(receiver_degree, 1.0))[:, None]
                norm_vals = di * dj
                if self.cached:
                    self.norm_vals = norm_vals

            nodes = nodes * norm_vals

        messages = nodes[conv_senders].transpose()
        messages = jnp.multiply(messages, jnp.squeeze(edges)).transpose()
        nodes = self.aggr(messages, conv_receivers,
                          num_nodes)

        return nodes