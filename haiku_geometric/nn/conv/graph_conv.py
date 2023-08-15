import haiku as hk
import jax.tree_util as tree
import jraph
import jax.numpy as jnp
from typing import Optional, Union
from haiku_geometric.nn.aggr.utils import aggregation
from haiku_geometric.nn.conv.utils import validate_input


class GraphConv(hk.Module):
    r"""The graph neural network operator from the `"Weisfeiler and Leman Go
    Neural: Higher-order Graph Neural Networks"
    <https://arxiv.org/abs/1810.02244>`_ paper

    The node features are computed as follows:

    .. math::
        \mathbf{{h}}_{u}^{k}=\mathbf{W}_1 \cdot \mathbf{{h}}_{u}^{k-1} + \mathbf{W}_2 \cdot \text{AGGREGATE}(\{e_{u, v} \cdot \mathbf{{h}}_{v}^{k-1}, \forall v \in \mathcal{N}(u)\})

    with :math:`AGGREGATE` being the aggregation operator (i.e. :obj:`"mean"`, :obj:`"max"`, or :obj:`"add"`)

    Args:
        out_channels (int): Size of the output features of a node.
        aggr (string or Aggregation, optional): The aggregation operator.
            Available values are: :obj:`"mean"`, :obj:`"max"`, or :obj:`"add"`.
            (default: :obj:`"add"`)
        bias (bool, optional): If :obj:`True`, the layer will add
            an additive bias to the output.
            (default: :obj:`True`)

    """

    def __init__(
            self,
            out_channels: int,
            aggr: str = 'add',
            bias: bool = True,
    ):
        """"""
        super().__init__()
        self.out_channels = out_channels
        self.aggr = aggregation(aggr)
        self.linear = hk.Linear(out_channels, with_bias=bias)
        self.linear_root = hk.Linear(out_channels, with_bias=False)

    def __call__(self,
                 nodes: jnp.ndarray = None,
                 senders: jnp.ndarray = None,
                 receivers: jnp.ndarray = None,
                 edges: Optional[jnp.ndarray] = None,
                 num_nodes: int = None
                 ) -> Union[jnp.ndarray, jraph.GraphsTuple]:
        """"""

        messages = nodes[senders]
        if edges is not None:
            messages = messages * edges

        if num_nodes is None:
            num_nodes = tree.tree_leaves(nodes)[0].shape[0]
        out = self.aggr(messages, receivers, num_nodes)

        out = self.linear(out)
        aux = self.linear_root(nodes)
        out = out + aux

        return out