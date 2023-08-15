import haiku as hk
import jax.tree_util as tree
import jraph
import jax.numpy as jnp
from typing import Optional, Union
from haiku_geometric.nn.aggr.utils import aggregation
from haiku_geometric.nn.conv.utils import validate_input
from typing import Callable


class EdgeConv(hk.Module):
    r"""The edge convolutional operator from the `"Dynamic Graph CNN for
    Learning on Point Clouds" <https://arxiv.org/abs/1801.07829>`_ paper.
    
    The node features are computed as follows:
    
    .. math::
        \mathbf{h}^{k + 1}_i = \sum_{j \in \mathcal{N}(i)} h_{\mathbf{\Theta}}(\mathbf{h}_i \, \Vert \, \mathbf{h}_j - \mathbf{h}_i)
    
    where :math:`h_{\mathbf{\Theta}}` denotes a neural network, e.g. a MLP.
    
    Args:
        nn (hk.Module): A neural network :math:`h_{\mathbf{\Theta}}` that
            maps node features :obj:`h` of shape
            :obj:`[-1, 2 * in_channels]` to shape :obj:`[-1, out_channels]`.
        aggr (string, optional): The aggregation operator
            (:obj:`"add"`, :obj:`"mean"`, :obj:`"max"`).
            (default: :obj:`"max"`)
    """
    def __init__(self, nn: Callable, aggr: str = 'max'):
        """"""
        super().__init__()
        self.aggr = aggregation(aggr)
        self.nn = nn
        
    def __call__(self,
                 nodes: jnp.ndarray,
                 senders: jnp.ndarray,
                 receivers: jnp.ndarray,
                 edges: Optional[jnp.ndarray] = None,
                 num_nodes: int = None
                 ) -> jnp.ndarray:
        """"""

        h_senders = nodes[senders]
        h_receivers = nodes[receivers]
        h = jnp.concatenate((h_senders, h_receivers - h_senders), axis=-1)
        messages = self.nn(h)
        if num_nodes is None:
            num_nodes = tree.tree_leaves(nodes)[0].shape[0]
        out = self.aggr(messages, receivers, num_nodes)
        return out