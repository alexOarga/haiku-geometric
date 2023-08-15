import haiku as hk
import jax.tree_util as tree
import jraph
import jax.numpy as jnp
from typing import Optional, Union
from haiku_geometric.nn.aggr.utils import aggregation
from haiku_geometric.nn.conv.utils import validate_input
from typing import Callable


class GINConv(hk.Module):
    r"""The graph isomorphism operator from the `"How Powerful are
    Graph Neural Networks?" <https://arxiv.org/abs/1810.00826>`_ paper

    The node features are computed as follows:
    
    .. math::
        \mathbf{{h}}_{u}^{k}= \phi\left( (1 + \epsilon) \mathbf{{h}}_{u}^{k-1} + \sum_{v \in \mathcal{N}(u)}  \mathbf{{h}}_{v}^{k-1}\right)

    where :math:`\phi` is a neural network (e.g. a MLP).

    Args:
        nn (hk.Module): A neural network :math:`\phi` that produces
            output features of shape :obj:`out_channels` defined by the user.
        eps (float, optional): :math:`\epsilon` value.
            (default: :obj:`0.`)
        train_eps (bool, optional): If :obj:`True`, :math:`\epsilon`
            will be a trainable parameter.
            (default: :obj:`False`)
    """

    def __init__(
            self, nn: Callable,
            eps: float = 0.,
            train_eps: bool = False,
    ):
        """"""
        super().__init__()
        self.aggr = aggregation('add')
        self.nn = nn
        self.train_eps = train_eps
        if train_eps:
            self.eps = hk.get_parameter("eps", shape=[1, 1], init=hk.initializers.RandomNormal())
        else:
            self.eps = eps

    def __call__(self,
                 nodes: jnp.ndarray,
                 senders: jnp.ndarray,
                 receivers: jnp.ndarray,
                 edges: Optional[jnp.ndarray] = None,
                 num_nodes: int = None
                 ) -> jnp.ndarray:
        """"""

        if num_nodes is None:
            num_nodes = tree.tree_leaves(nodes)[0].shape[0]
        h = tree.tree_map(lambda x: self.aggr(x[senders], receivers,
                                              num_nodes), nodes)
        h = h + ((1 + self.eps) * nodes)
        out = self.nn(h)

        return out
