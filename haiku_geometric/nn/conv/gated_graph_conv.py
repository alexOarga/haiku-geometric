import warnings
import haiku as hk
import jraph
import jax.numpy as jnp
import jax.tree_util as tree

from typing import Optional, Union
from haiku_geometric.nn.aggr.utils import aggregation


class GatedGraphConv(hk.Module):
    r"""The gated graph convolution operator from the `"Gated Graph Sequence
    Neural Networks" <https://arxiv.org/abs/1511.05493>`_ paper

    where the output features are computed as follows:

    .. math::
        \mathbf{{h}}_{u}^{(0)} = \mathbf{{h}}_{u}^{(0)} \Vert \mathbf{0}

    for layer :math:`k: 1,...,L`:

    .. math::
        \mathbf{{m}}_{u}^{(k)} &= \text{AGGREGATE}(\{e_{u, v} \cdot \mathbf{{W}} \cdot \mathbf{{h}}_{v}^{(k - 1)}, \forall v \in \mathcal{N}(u)\}) \\
        \mathbf{{h}}_{u}^{(k)} &= GRU(\mathbf{{m}}_{u}^{(k)}, \mathbf{{h}}_{u}^{(k - 1)})
        

    with :math:`AGGREGATE` being the aggregation operator (i.e. :obj:`"mean"`, :obj:`"max"`, or :obj:`"add"`).

    Args:
        out_channels (int): Size of the output features of each node.
        num_layers (int): Number of layers :math:`L`.
        aggr (string, optional): Aggregation operator
            (:obj:`"add"`, :obj:`"mean"`, :obj:`"max"`).
            (default: :obj:`"add"`)

    """
    def __init__(
            self,
            out_channels: int,
            num_layers: int,
            aggr: str = 'add',
            # bias: bool = True  # haiku GRU always includes bias. TODO: make bias optional.
    ):
        """"""
        super().__init__()
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.aggr = aggregation(aggr)
        self.weights = hk.get_parameter("weights", shape=[num_layers, out_channels, out_channels],
                                        init=hk.initializers.RandomNormal())
        self.rnn = hk.GRU(out_channels)

    def __call__(self,
                 nodes: jnp.ndarray,
                 senders: jnp.ndarray,
                 receivers: jnp.ndarray,
                 edges: Optional[jnp.ndarray] = None,
                 num_nodes: Optional[int] = None,
                 ) -> jnp.ndarray:
        """"""

        in_channels = nodes.shape[-1]
        if num_nodes is None:
            num_nodes = tree.tree_leaves(nodes)[0].shape[0]

        if in_channels > self.out_channels:
            raise RuntimeError("Input features size of GatedGraphConv cannot be larger than "
                               "the output feature size")
        if edges is not None and edges.shape[-1] > 1:
            warnings.warn("Edge features of size larger than 1 are not taken into account in GatedGraphConv")

        if in_channels < self.out_channels:
            zeros = jnp.zeros((nodes.shape[0], self.out_channels - in_channels))
            x = jnp.concatenate((nodes, zeros), axis=1)
        else:
            x = nodes

        for i in range(self.num_layers):
            m = jnp.matmul(x, self.weights[i])
            messages = m[senders]
            if edges is not None and edges.shape[-1] == 1:
                messages = messages * edges
            m = self.aggr(messages, receivers, num_segments=num_nodes)

            x = self.rnn(m, x)
            x = x[0]

        return x