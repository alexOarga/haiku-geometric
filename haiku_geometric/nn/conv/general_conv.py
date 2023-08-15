import haiku as hk
import jax
import jax.numpy as jnp
import jax.tree_util as tree
import jraph

from haiku_geometric.nn.aggr.utils import aggregation
from haiku_geometric.nn.conv.utils import validate_input
from typing import Optional, Union


class GeneralConv(hk.Module):
    r"""A general GNN layer adapted from the `"Design Space for Graph Neural
    Networks" <https://arxiv.org/abs/2011.08843>`_ paper.

    where the output features are computed as follows:
    
    .. math::
        \mathbf{{h}}_{u}^{k}= \text{AGGREGATE}(\{\mathbf{m}_{u,v}, \forall v \in \mathcal{N}(u)\})

    with :math:`AGGREGATE` being the aggregation operator (i.e. :obj:`"mean"`, :obj:`"max"`, or :obj:`"add"`) and each message :math:`\mathbf{m}_{u,v}` is computed as:
    
    .. math::
        \mathbf{m}_{u,v} = \mathbf{W}_1 \cdot \mathbf{{h}}_{v}^{k-1}

    If :obj:`directed_msg=True`, the message is bidirectional:
    
    .. math::
        \mathbf{m}_{u,v} = \mathbf{W}_1 \cdot \mathbf{{h}}_{v}^{k-1} + \mathbf{W}_2 \cdot \mathbf{{h}}_{u}^{k-1}

    If :obj:`in_edge_channels` is not :obj:`None`, the edge features are also added to the message:
    
    .. math::
        \mathbf{m}_{u,v} = \mathbf{W}_1 \cdot \mathbf{{h}}_{v}^{k-1} + \mathbf{W}_3 \cdot \mathbf{{e}}_{u, v}

    If :obj:`attention=True`, attention is performed on the message computation:
    
    .. math::
        \mathbf{m}_{u,v} = \alpha_{u,v}(\mathbf{W}_1 \cdot \mathbf{{h}}_{v}^{k-1})

    where the attention coefficient :math:`\alpha_{u,v}` is computed as follows:
    
    .. math::
        \alpha_{i j}=\frac{\exp \left(\operatorname{LeakyReLU}\left(\overrightarrow{\mathbf{a}}^{T}\left[\mathbf{W} \vec{h}_{i} + \mathbf{W} \vec{h}_{j}\right]\right)\right)}{\sum_{k \in \mathcal{N}_{i}} \exp \left(\operatorname{LeakyReLU}\left(\overrightarrow{\mathbf{a}}^{T}\left[\mathbf{W} \vec{h}_{i} + \mathbf{W} \vec{h}_{k}\right]\right)\right)}

    If :obj:`skip_linear=True` a skip connection is added to the output:
    
    .. math::
        \mathbf{{h}}_{u}^{k}= \text{AGGREGATE}(\{\mathbf{m}_{u,v}, \forall v \in \mathcal{N}(u)\}) + \mathbf{W}_4 \cdot \mathbf{{h}}_{u}^{k-1} \\


    Args:
        out_channels (int): Size of the output features of a node.
        in_edge_channels (int, optional): Size of each edge features.
            (default: :obj:`None`)
        aggr (string or Aggregation, optional): The aggregation operator.
            Available values are: :obj:`"mean"`, :obj:`"max"`, or :obj:`"add"`.
            (default: :obj:`"add"`)
        skip_linear (bool, optional):
            (default: :obj:`False`)
        directed_msg (bool, optional):
            (default: :obj:`True`)
        heads (int, optional): Number of head attentions.
            If (:obj:`attention=True`) and (:obj:`heads > 1`) the multi-head features are mean aggregated.
            (default: :obj:`1`)
        attention (bool, optional): perform attention over the messages
            (default: :obj:`False`)
        attention_type (str, optional): Type of attention: :obj:`"additive"`,
            :obj:`"dot_product"`.
            (default: :obj:`"additive"`)
        l2_normalize (bool, optional): If :obj:`True`, output features
            are :math:`\ell_2`-normalized.
            (default: :obj:`False`)
        bias (bool, optional): If :obj:`True`, linear transformation also add bias.
            (default: :obj:`True`)
    """

    def __init__(
            self,
            out_channels: Optional[int],
            in_edge_channels: int = None,
            aggr: str = "add",
            skip_linear: bool = False,
            directed_msg: bool = True,
            heads: int = 1,
            attention: bool = False,
            attention_type: str = "additive",
            l2_normalize: bool = False,
            bias: bool = True,
    ):
        """"""
        super().__init__()
        self.out_channels = out_channels
        self.heads = heads
        self.in_edge_channels = in_edge_channels
        self.skip_linear = skip_linear
        self.directed_msg = directed_msg
        self.attention = attention
        self.attention_type = attention_type
        self.l2_normalize = l2_normalize

        # Initialize parameters
        self.aggr = aggregation(aggr)
        C = self.out_channels
        H = self.heads
        if self.directed_msg:
            self.linear_msg = hk.Linear(C * H, with_bias=bias)
        else:
            self.linear_msg = hk.Linear(C * H, with_bias=bias)
            self.linear_msg_i = hk.Linear(C * H, with_bias=bias)

        if self.skip_linear:
            self.linear_skip = hk.Linear(out_channels, with_bias=bias)
        else:
            self.linear_skip = None

        if self.in_edge_channels is not None:
            self.linear_edge = hk.Linear(C * H, with_bias=bias)

        if self.attention:
            if self.attention_type == 'additive':
                # Notice that the shape here is (2 * channel size) because the alpha weight vector
                # is applied to the concatenation of two nodes features
                self.alpha = hk.get_parameter("alpha", shape=[1, H, C], init=hk.initializers.TruncatedNormal())
            elif self.attention_type == 'dot_product':
                self.scaler = hk.get_parameter("scaler", shape=[1], init=hk.initializers.TruncatedNormal())
            else:
                raise ValueError("Unknown attention type: {self.attention_type}.")

    def __call__(self,
                 nodes: jnp.ndarray,
                 senders: jnp.ndarray,
                 receivers: jnp.ndarray,
                 edges: Optional[jnp.ndarray] = None,
                 num_nodes: Optional[int] = None,
                 ) -> jnp.ndarray:
        """"""

        C = self.out_channels
        H = self.heads

        try:
            if num_nodes is None:
                num_nodes = tree.tree_leaves(nodes)[0].shape[0]
        except IndexError:
            raise IndexError('GeneralConv requires node features')

        total_num_nodes = tree.tree_leaves(nodes)[0].shape[0]

        # We compute the softmax logits using a function that takes the
        # embedded sender and receiver attributes.
        #  sent_attributes.shape = (|edges|, in_channels)
        sent_attributes = nodes[senders]
        received_attributes = nodes[receivers]

        def generate_message(sent_attributes, received_attributes):
            # Nodes features are transformed with a Linear layer
            # Output size is (out_channels * heads) to simulate multiple heads weights
            #  nodes.shape = (|edges|, C*H)
            if self.directed_msg:
                sent_attributes = self.linear_msg(sent_attributes)
            else:
                sent_attributes = self.linear_msg(sent_attributes) \
                                  + self.linear_msg(received_attributes)
            if self.in_edge_channels is not None:
                sent_attributes += self.linear_edge(edges)
            return sent_attributes

        # x.shape = (|edges|, C*H)
        x = generate_message(sent_attributes, received_attributes)

        # Reshape to: x.shape = (|edges|, H, C)
        x = jnp.reshape(x, (-1, H, C))

        if self.attention:
            if self.attention_type == 'dot_product':
                x_j = generate_message(received_attributes, sent_attributes)
                x_j = jnp.reshape(x_j, (-1, H, C))
                att_softmax_logits = jnp.sum((x * x_j), axis=-1)
                print("att_softmax_logits", att_softmax_logits.shape)
                att_softmax_logits = att_softmax_logits / jnp.sqrt(self.scaler)
            elif self.attention_type == 'additive':
                att_softmax_logits = x * self.alpha
                #  att_softmax_logits.shape = (|edges|, H)
                att_softmax_logits = jnp.sum(att_softmax_logits, axis=-1)
            else:
                raise ValueError("Unknown attention type: {self.attention_type}.")

            att_softmax_logits = jax.nn.leaky_relu(att_softmax_logits)

            # Compute the attention softmax weights on the entire tree.
            #  att_weights.shape = (|edges|, H)
            att_weights = jraph.segment_softmax(
                att_softmax_logits, segment_ids=receivers, num_segments=num_nodes)

            # sent_attributes.shape = (|edges|, H, C)
            x = jnp.reshape(x, (-1, H, C))

            # att_weights.shape = (|edges|, H, 1)
            att_weights = jnp.reshape(att_weights, (-1, H, 1))

            # Apply attention weights.
            # messages.shape = (|edges|, H, C)
            messages = att_weights * x
        else:
            messages = x

        # Aggregate messages to nodes.
        new_nodes = self.aggr(messages, receivers, num_segments=num_nodes)

        # Aggregate heads
        new_nodes = jnp.mean(new_nodes, axis=1)

        if self.skip_linear:
            new_nodes += self.linear_skip(nodes)
        if self.l2_normalize:
            new_nodes /= jnp.linalg.norm(new_nodes, ord=2, axis=-1, keepdims=True)

        return new_nodes