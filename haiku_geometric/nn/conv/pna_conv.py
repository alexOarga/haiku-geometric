import haiku as hk
import jraph
import jax
import jax.numpy as jnp
import jax.tree_util as tree

from jraph._src.utils import segment_sum, segment_mean, segment_max
from typing import Optional, Union, List, Callable, Dict, Any
from haiku_geometric.nn.aggr import DegreeScalerAggregation
from haiku_geometric.utils import degree

class PNAConv(hk.Module):
    r"""
    The Principal Neighbourhood Aggregation graph convolution operator
    from the `"Principal Neighbourhood Aggregation for Graph Nets"
    <https://arxiv.org/abs/2004.05718>`_ paper.
    
    Out features are computed as follows:
    
    .. math::
        \mathbf{h}_u^{(k+1)} = U \left(
        \mathbf{h}_u^{(k)}, \underset{v \in \mathcal{N}(u)}{\bigoplus}
        M \left( \mathbf{h}_u^{(k)}, \mathbf{h}_v^{(k)} \right)
        \right)
    
    with :math:`M` and :math:`U` being MLPs, and:

    .. math::
        \bigoplus = \underbrace{\begin{bmatrix}
            1 \\
            S(\mathbf{D}, \alpha=1) \\
            S(\mathbf{D}, \alpha=-1)
        \end{bmatrix} }_{\text{scalers}}
        \otimes \underbrace{\begin{bmatrix}
            \mu \\
            \sigma \\
            \max \\
            \min
        \end{bmatrix}}_{\text{aggregators}},
    """

    def __init__(
            self,
            out_channels: int,
            aggregators: List[str],
            scalers: List[str],
            deg: jnp.ndarray,
            edge_dim: Optional[int] = None,
            towers: int = 1,
            pre_layers: int = 1,
            post_layers: int = 1,
            divide_input: bool = False,
            act: Union[Callable, None] = "relu",
            act_kwargs: Optional[Dict[str, Any]] = None,
            train_norm: bool = False,
    ):
        """"""
        super().__init__()
        # Initialize parameters
        self.aggr = DegreeScalerAggregation(aggregators, scalers, deg, train_norm)
        
        assert out_channels % towers == 0
        self.out_channels = out_channels
        self.aggregators = aggregators
        self.scalers = scalers
        self.edge_dim = edge_dim
        self.towers = towers
        self.divide_input = divide_input
        self.pre_layers = pre_layers
        self.post_layers = post_layers

        self.linear = hk.Linear(out_channels)
        self.act = act

    def __call__(self,
                 nodes: jnp.ndarray,
                 senders: jnp.ndarray,
                 receivers: jnp.ndarray,
                 edges: Optional[jnp.ndarray] = None,
                 num_nodes: Optional[int] = None,
                 ) -> jnp.ndarray:
        """"""

        # Initialize modules
        in_channels = nodes.shape[-1]
        self.F_in = in_channels // self.towers if self.divide_input else in_channels
        self.F_out = self.out_channels // self.towers
        
        if (self.edge_dim is not None) and (edges is None):
            raise ValueError("Edge features must be provided if edge_dim is not None.")
        if self.edge_dim is not None:
            self.edge_encoder = hk.Linear(self.F_in)
        
        self.pre_nns = []
        self.post_nns = []
        for _ in range(self.towers):
            modules = [hk.Linear(self.F_in)]
            for _ in range(self.pre_layers - 1):
                modules += [self.act]
                modules += [hk.Linear(self.F_in)]
            self.pre_nns.append(hk.Sequential(modules))

            # No need to compute in channels
            #in_channels = (len(self.aggregators) * len(self.scalers) + 1) * self.F_in
            modules = [hk.Linear(self.F_out)]
            for _ in range(self.post_layers - 1):
                modules += [self.act]
                modules += [hk.Linear(self.F_out)]
            self.post_nns.append(hk.Sequential(modules))
        
        # Forward pass
        if num_nodes is None:
            num_nodes = tree.tree_leaves(nodes)[0].shape[0]

        if self.divide_input:
            nodes = nodes.reshape(-1, self.towers, self.F_in)
        else:
            nodes = nodes.reshape(-1, 1, self.F_in)
            nodes = jnp.tile(nodes, (1, self.towers, 1))
        
        # Build messages
        if edges is not None:
            edges = self.edge_encoder(edges)
            edges = edges.reshape(-1, 1, self.F_in)
            edges = jnp.tile(edges, (1, self.towers, 1))
            x_i = nodes[senders]
            x_j = nodes[receivers]
            h = jnp.concatenate([x_i, x_j, edges], axis=-1)
        else:
            x_i = nodes[senders]
            x_j = nodes[receivers]
            h = jnp.concatenate([x_i, x_j], axis=-1)
            
        hs = [nn(h[:, i]) for i, nn in enumerate(self.pre_nns)]
        msg = jnp.stack(hs, axis=1)
            
        # Propagate
        out = self.aggr(msg, receivers, num_nodes)
        
        out = jnp.concatenate((nodes, out), axis=-1)
        outs = [nn(out[:, i]) for i, nn in enumerate(self.post_nns)]
        out = jnp.concatenate(outs, axis=1)

        return self.linear(out)