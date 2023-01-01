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
        bias (bool, optional): If :obj:`True`, the layer will add
            an additive bias to the output. (default: :obj:`True`)
    """

    def __init__(
            self,
            out_channels: int,
            heads: int = 1,
            concat: bool = True,
            negative_slope: float = 0.2,
            # dropout: float = 0.0,
            add_self_loops: bool = True,
            # edge_dim: Optional[int] = None, # TODO: include edges in GATConv
            # fill_value: Union[float, Tensor, str] = 'mean',
            bias: bool = True
    ):
        """"""
        super().__init__()
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.add_self_loops = add_self_loops

        # Initialize parameters
        C = self.out_channels
        H = self.heads
        self.linear = hk.Linear(C * H)
        # Notice that the shape here is (2 * channel size) because the alpha weight vector
        # is applied to the concatenation of two nodes features
        self.alpha = hk.get_parameter("alpha", shape=[1, H, 2 * C], init=hk.initializers.RandomNormal())
        if bias:
            self.bias = hk.Bias()
        else:
            self.bias = None

    def __call__(self,
                 nodes: jnp.ndarray = None,
                 senders: jnp.ndarray = None,
                 receivers: jnp.ndarray = None,
                 edges: Optional[jnp.ndarray] = None,
                 graph: Optional[jraph.GraphsTuple] = None
                 ) -> Union[jnp.ndarray, jraph.GraphsTuple]:
        """"""
        nodes, edges, receivers, senders = \
            validate_input(nodes, senders, receivers, edges, graph)

        C = self.out_channels
        H = self.heads

        try:
            sum_n_node = nodes.shape[0]
        except IndexError:
            raise IndexError('GATConv requires node features')

        # Nodes features are transformed with a Linear layer
        # Output size is (out_channels * heads) to simulate multiple heads weights
        #  nodes.shape = (N, C*H)
        nodes = self.linear(nodes)

        total_num_nodes = tree.tree_leaves(nodes)[0].shape[0]
        if self.add_self_loops:
            # We add self edges to the senders and receivers so that each node
            # includes itself in aggregation.
            #   receivers (or senders) shape = (|edges|, 1)
            receivers, senders = add_self_loops(receivers, senders,
                                                   total_num_nodes)

        # We compute the softmax logits using a function that takes the
        # embedded sender and receiver attributes.
        #  sent_attributes.shape = (|edges|, C*H)
        sent_attributes = nodes[senders]
        received_attributes = nodes[receivers]

        # x.shape = (|edges|, 2*C*H)
        x = jnp.concatenate(
            (sent_attributes, received_attributes), axis=1)

        # x.shape = (|edges|, 2, C*H)
        x = jnp.reshape(x, (-1, H, 2 * C))

        att_softmax_logits = x * self.alpha
        #  att_softmax_logits.shape = (|edges|, H)
        att_softmax_logits = jnp.sum(att_softmax_logits, axis=-1)

        att_softmax_logits = jax.nn.leaky_relu(
            att_softmax_logits, negative_slope=self.negative_slope)

        #: TODO: dropout only during training
        # nodes = haiku.dropout(
        #    rng, self.dropout, nodes)

        # Compute the attention softmax weights on the entire tree.
        #  att_weights.shape = (|edges|, H)
        att_weights = jraph.segment_softmax(
            att_softmax_logits, segment_ids=receivers, num_segments=sum_n_node)

        # sent_attributes.shape = (|edges|, H, C)
        sent_attributes = jnp.reshape(sent_attributes, (-1, H, C))

        # att_weights.shape = (|edges|, H, 1)
        att_weights = jnp.reshape(att_weights, (-1, H, 1))

        # Apply attention weights.
        # messages.shape = (|edges|, H, C)
        messages = att_weights * sent_attributes

        # Aggregate messages to nodes.
        nodes = jax.ops.segment_sum(messages, receivers, num_segments=sum_n_node)

        if self.concat:
            nodes = jnp.reshape(nodes, (-1, H * C))
        else:
            nodes = jnp.mean(nodes, axis=1)

        if self.bias is not None:
            nodes = self.bias(nodes)

        if graph is not None:
            graph = graph._replace(nodes=nodes)
            return graph
        else:
            return nodes