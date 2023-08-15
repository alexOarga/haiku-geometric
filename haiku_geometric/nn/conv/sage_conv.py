import haiku as hk
import jax
import jax.numpy as jnp
import jax.tree_util as tree
import jraph
from typing import Optional, Union
from haiku_geometric.nn.aggr.utils import aggregation
from haiku_geometric.nn.conv.utils import validate_input
from typing import Optional


class SAGEConv(hk.Module):
    r"""
    The GraphSAGE operator from the `"Inductive Representation Learning on
    Large Graphs" <https://arxiv.org/abs/1706.02216>`_ paper

    The node features are computed as follows:
    
    .. math::
        \mathbf{{h}}_{u}^{k}=\mathbf{W}_1 \cdot \mathbf{{h}}_{u}^{k-1} + \mathbf{W}_2 \cdot \text{CONCAT}(\mathbf{{h}}_{u}^{k-1}, \mathbf{{h}}_{\mathcal{N}(u)}^{k})

    having:
    
    .. math::
        \mathbf{{h}}_{\mathcal{N}(u)}^{k} = \text{AGGREGATE}(\{\mathbf{{h}}_{v}^{k-1}, \forall v \in \mathcal{N}(u)\})

    and :math:`AGGREGATE` being an aggregation operator (i.e. :obj:`"mean"`, :obj:`"max"`, or :obj:`"sum"`)

    If :obj:`project = True`, then :math:`\mathbf{{h}}_{u}^{k-1}` is first projected via:

    .. math::
        \mathbf{{h}}_{v}^{k-1}=\text{ReLU}(\mathbf{W}_3 \mathbf{{h}}_{v}^{k-1} + \mathbf{b})

    Args:
        out_channels (int): Size of the output features of a node.
        aggr (string or Aggregation, optional): The aggregation operator.
            Available values are: :obj:`"mean"`, :obj:`"max"`, or :obj:`"sum"`.
            (default: :obj:`"mean"`)
        normalize (bool, optional): If :obj:`True`, output features
            are :math:`\ell_2`-normalized.
            (default: :obj:`False`)
        root_weight (bool, optional): If :obj:`False` the linear transformed features
            :math:`\mathbf{W}_1 \cdot \mathbf{{h}}_{u}^{k-1}` are not added to the output features.
            (default: :obj:`True`)
        project (bool, optional): If :obj:`True`, neighbour features are projected before aggregation as
            explained above.
            (default: :obj:`False`)
        bias (bool, optional): If :obj:`True`, the layer will add
            an additive bias to the output.
            (default: :obj:`True`)
    """

    def __init__(
            self,
            out_channels: int,
            aggr: Optional[str] = "mean",
            normalize: bool = False,
            root_weight: bool = True,
            project: bool = False,
            bias: bool = True
    ):
        """"""
        super().__init__()
        self.out_channels = out_channels
        self.normalize = normalize
        self.root_weight = root_weight
        self.project = project

        self.linear_left = hk.Linear(out_channels, with_bias=bias)
        if self.root_weight:
            self.linear_right = hk.Linear(out_channels, with_bias=False)
        self.aggr = aggregation(aggr)

    def __call__(self,
                 nodes: jnp.ndarray,
                 senders: jnp.ndarray,
                 receivers: jnp.ndarray,
                 num_nodes: Optional[int] = None,
                 ) -> jnp.ndarray:
        """"""

        if num_nodes is None:
            num_nodes = tree.tree_leaves(nodes)[0].shape[0]
        in_channels = nodes.shape[1]

        if self.project:
            h = jax.nn.relu(hk.Linear(in_channels, with_bias=True)(nodes))
        else:
            h = nodes

        h = self.aggr(h[senders], receivers, num_nodes)
        h = jnp.concatenate((nodes, h), axis=1)
        out = self.linear_left(h)

        if self.root_weight:
            out = out + self.linear_right(nodes)

        if self.normalize:
            out /= jnp.linalg.norm(out, ord=2, axis=-1, keepdims=True)

        return out
