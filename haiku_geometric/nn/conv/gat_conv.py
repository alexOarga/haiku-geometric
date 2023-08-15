# Adapted from Gordić's Annotated Graph Attention Networks:
# https://github.com/gordicaleksa/pytorch-GAT/blob/main/The%20Annotated%20GAT%20(Cora).ipynb
import haiku as hk
import jax
import jax.numpy as jnp
import jax.tree_util as tree
import jraph

from typing import Optional, Union
from haiku_geometric.nn.aggr.utils import aggregation
from haiku_geometric.nn.conv.utils import validate_input
from haiku_geometric.transforms import add_self_loops


class GATConv(hk.Module):
    r"""Graph attention layer from `"Graph Attention Networks"
    <https://arxiv.org/abs/1710.10903>`_ paper

    where each node's output feature is computed as follows:
    
    .. math::
        \vec{h}_{i}^{\prime}=\sigma\left(\sum_{j \in \mathcal{N}_{i}} \alpha_{i j} \mathbf{W} \vec{h}_{j}\right)

    where the attention coefficients are computed as:
    
    .. math::
        \alpha_{i j}=\frac{\exp \left(\operatorname{LeakyReLU}\left(\overrightarrow{\mathbf{a}}^{T}\left[\mathbf{W} \vec{h}_{i} \| \mathbf{W} \vec{h}_{j}\right]\right)\right)}{\sum_{k \in \mathcal{N}_{i}} \exp \left(\operatorname{LeakyReLU}\left(\overrightarrow{\mathbf{a}}^{T}\left[\mathbf{W} \vec{h}_{i} \| \mathbf{W} \vec{h}_{k}\right]\right)\right)}

    When multiple attention heads are used, the output nodes features are averaged:
    
    .. math::
        \vec{h}_{i}^{\prime}=\sigma\left(\frac{1}{K} \sum_{k=1}^{K} \sum_{j \in \mathcal{N}_{i}} \alpha_{i j}^{k} \mathbf{W}^{k} \vec{h}_{j}\right)

    If `concat=True` the output feature is the concatenation of the :math:`K` heads features:
    
    .. math::
        \vec{h}_{i}^{\prime}=\|_{k=1}^{K} \sigma\left(\sum_{j \in \mathcal{N}_{i}} \alpha_{i j}^{k} \mathbf{W}^{k} \vec{h}_{j}\right)

    Args:
        out_channels (int): Size of the output features produced by the layer for each node.
        heads (int, optional): Number of head attentions.
            (default: :obj:`1`)
        concat (bool, optional): If :obj:`False`, the multi-head features are averaged
            else concatenated.
            (default: :obj:`True`)
        negative_slope (float, optional): scalar specifying the negative slope of the LeakyReLU.
            (default: :obj:`0.2`)
        add_self_loops (bool, optional): If :obj:`True`, will add
            a self-loop for each node of the graph. (default: :obj:`True`)
        dropout (float, optional): Dropout applied to attention weights.
            This dropout simulates random sampling of the neigbours.
            (default: :obj:`0.0`)
        dropout_nodes (float, optional): Dropout applied initially to the input features.
            (default: :obj:`0.0`)
        bias (bool, optional): If :obj:`True`, the layer will add
            an additive bias to the output. (default: :obj:`True`)
        init (hk.initializers.Initializer): Weights initializer
            (default: :obj:`hk.initializers.VarianceScaling(1.0, "fan_avg", "truncated_normal")`)
    """

    def __init__(
            self,
            out_channels: int,
            heads: int = 1,
            concat: bool = True,
            negative_slope: float = 0.2,
            dropout: float = 0.0,
            dropout_nodes: float = 0.0,
            add_self_loops: bool = True,
            # edge_dim: Optional[int] = None, # TODO: include edges in GATConv
            # fill_value: Union[float, Tensor, str] = 'mean',
            bias: bool = True,
            init: hk.initializers.Initializer = None
    ):
        """"""
        super().__init__()
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.dropout_attention = dropout
        self.dropout_nodes = dropout_nodes
        self.negative_slope = negative_slope
        self.add_self_loops = add_self_loops

        # Initialize parameters
        C = self.out_channels
        H = self.heads

        if init is None:
          init = hk.initializers.VarianceScaling(1.0, "fan_avg", "truncated_normal")

        self.linear_proj = hk.Linear(C * H, with_bias=False, 
                                     w_init=init)

        self.scoring_fn_target = hk.get_parameter(
            "scoring_fn_target", 
            shape=[1, H, C], 
            init=init)
        self.scoring_fn_source = hk.get_parameter(
            "scoring_fn_source", 
            shape=[1, H, C], 
            init=init)

    def __call__(self,
                 in_nodes_features: jnp.ndarray,
                 senders: jnp.ndarray,
                 receivers: jnp.ndarray,
                 edges: Optional[jnp.ndarray] = None,
                 num_nodes: Optional[int] = None,
                 training: bool = False
                 ) -> jnp.ndarray:
        """"""

        C = self.out_channels
        H = self.heads


        try:
            if num_nodes is None:
                num_nodes = tree.tree_leaves(in_nodes_features)[0].shape[0]
        except IndexError:
            raise IndexError('GATConv requires node features')

        # reshape to : (N, H, C)
        nodes_features_proj = self.linear_proj(in_nodes_features).reshape(-1, H, C)

        if training:
            nodes_features_proj = hk.dropout(
                jax.random.PRNGKey(42), self.dropout_nodes, nodes_features_proj)

        if self.add_self_loops:
            # We add self edges to the senders and receivers so that each node
            # includes itself in aggregation.
            #   receivers (or senders) shape = (|edges|, 1)
            senders, receivers, edges = add_self_loops(senders, receivers, edges,
                                                fill_value=1.0, # TODO: use 'mean' as a fill value
                                                num_nodes=num_nodes)

        # shape: (N, H)
        scores_source = jnp.sum(nodes_features_proj * self.scoring_fn_source, axis=-1)
        scores_target = jnp.sum(nodes_features_proj * self.scoring_fn_target, axis=-1)

        # scores_source_lifted shape: (|edges|, H)
        # nodes_features_proj shape: (|edges|, H, C)
        scores_source_lifted, scores_target_lifted, nodes_features_proj_lifted \
            = self.lift(scores_source, scores_target, nodes_features_proj, senders, receivers)

        # shape: (|edges|, 1)
        scores_per_edge = jax.nn.leaky_relu(
            (scores_source_lifted + scores_target_lifted), 
            negative_slope=self.negative_slope)

        # shape: (|edges|, 1)
        attentions_per_edge = jraph.segment_softmax(scores_per_edge, receivers, num_segments=num_nodes)

        if training:
            attentions_per_edge = hk.dropout(
                jax.random.PRNGKey(42), self.dropout_attention, attentions_per_edge)

        # shape: (|edges|, H, C)
        nodes_features_proj_lifted_weighted = nodes_features_proj_lifted * jnp.expand_dims(attentions_per_edge, axis=-1)

        # shape: (N, H, C)
        out_nodes_features = jax.ops.segment_sum(nodes_features_proj_lifted_weighted, receivers, num_segments=num_nodes)

        if self.concat:
            out_nodes_features = jnp.reshape(out_nodes_features, (-1, H * C))
        else:
            out_nodes_features = jnp.mean(out_nodes_features, axis=1)

        return out_nodes_features

    def lift(self, scores_source, scores_target, nodes_features_matrix_proj, senders, receivers):
        src_nodes_index = senders
        trg_nodes_index = receivers

        scores_source = scores_source[src_nodes_index]
        scores_target = scores_target[trg_nodes_index]
        nodes_features_matrix_proj_lifted = nodes_features_matrix_proj[src_nodes_index]

        return scores_source, scores_target, nodes_features_matrix_proj_lifted