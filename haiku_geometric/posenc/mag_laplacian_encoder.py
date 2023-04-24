import haiku as hk
import jax
import jax.numpy as jnp
from typing import Callable, Any, Optional
from haiku_geometric.nn import GCNConv
from haiku_geometric.models import MLP
from haiku_geometric.utils import to_undirected, eigv_laplacian

# Most of this code was obtained from the official repository from the paper: Transformers Meet Directed Graphs
# See: github.com/deepmind/digraph_transformer

class MagLaplacianEncoder(hk.Module):
    r"""
    MagLapNet Positional Encoder described in the `"Transformers Meet Directed Graphs" <https://arxiv.org/pdf/2302.00049>`_ paper.
    Positional encodings are computed using the Magnetic Laplacian matrix (Hermitian matrix).

    Usage::

        from haiku_geometric.utils import eigv_magnetic_laplace
        from haiku_geometric.posenc import MagLaplacianEncoder

        # Compute the eigenvectors of the Magnetic Laplacian matrix
        eigenvalues, eigenvectors = eigv_magnetic_laplace(
            senders=senders,
            receivers=receivers,
            k=...)

        # The function that you will transform with Haiku
        def your_forward_function(...):

            # Create the encoder model
            model = MagLaplacianEncoder(...)

            # Encode the eignevalues and eigenvectors
            h = model(senders, receivers, eigenvalues, eigenvectors, is_training)
      
    """
    def __init__(self,
                 d_model_elem: int = 32,
                 d_model_aggr: int = 256,
                 num_heads: int = 4,
                 n_layers: int = 1,
                 dropout: float = 0.2,
                 activation: Callable[[jnp.ndarray], jnp.ndarray] = jax.nn.relu,
                 return_real_output: bool = True,
                 consider_im_part: bool = True,
                 use_signnet: bool = True,
                 use_gnn: bool = False,
                 use_attention: bool = False,
                 concatenate_eigenvalues: bool = False,
                 norm: Optional[Any] = None):
        r"""
        Args:
            d_model_elem (int, optional): Embedding dimension for each element of the eigenvectors.
                (default: :obj:`32`).
            d_model_aggr (int, optional): Dimension of the aggregation of all the elements of the eigenvectors.
                (default: :obj:`256`).
            num_heads (int, optional): Number of attention heads 
                (default: :obj:`4`).
            n_layers (int, optional): Number of layers for the MLPs.
                (default: :obj:`1`).
            dropout (float, optional): Dropout rate.
                (default: :obj:`0.2`).
            activation (Callable[[jnp.ndarray], jnp.ndarray], optional): Activation function.
                (default: :obj:`jax.nn.relu`).
            return_real_output (bool, optional): Whether to return only the real part of the output.
                (default: :obj:`True`).
            consider_im_part (bool, optional): Whether to consider the imaginary part of the eigenvectors.
                (default: :obj:`True`).
            use_signnet (bool, optional): Whether to use the SignNet, this is, each eigenvector :math:`\gamma_i` is
                processed as :math:`f_{elem}(\gamma_i) + f_{elem}(-\gamma_i)` where :math:`f_{elem}` is an MLP or a GNN.
                (default: :obj:`True`).
            use_gnn (bool, optional): Whether to use a GNN aggregate embeddings of the eigenvectors instead of an MLP.
                (default: :obj:`False`).
            use_attention (bool, optional): Whether to apply a multi-head attention layer to the embeddings.
                (default: :obj:`False`).
            concatenate_eigenvalues (bool, optional): Wheter to initially concatenate eignevalues to the eigenvectors.
                (default: :obj:`False`).
            norm (Callable[[jnp.ndarray], jnp.ndarray], optional): Normalization layer
                (default: :obj:`None`).
        """
        super().__init__()
        self.concatenate_eigenvalues = concatenate_eigenvalues
        self.consider_im_part = consider_im_part
        self.use_signnet = use_signnet
        self.use_gnn = use_gnn
        self.use_attention = use_attention
        self.num_heads = num_heads
        self.dropout_p = dropout
        self.norm = norm

        if self.use_gnn:
            self.element_gnn = _GNN(
                int(2 * d_model_elem) if self.consider_im_part else d_model_elem,
                gnn_type='gcn',
                k_hop=n_layers,
                mlp_layers=n_layers,
                activation=activation,
                use_edge_attr=False,
                concat=True,
                residual=False,
                name='re_element')
        else:
            dim = int(2 * d_model_elem) if self.consider_im_part else d_model_elem
            self.element_mlp = MLP(
                [dim] * n_layers,
                with_layer_norm=False,
                activation=activation,
                activate_final=True)

        self.re_aggregate_mlp = MLP(
            [d_model_aggr] * n_layers,
            activation=activation,
            with_layer_norm=False,
            activate_final=True)

        self.im_aggregate_mlp = None
        if not return_real_output and self.consider_im_part:
            self.im_aggregate_mlp = MLP(
                [d_model_aggr] * n_layers,
                activation=activation,
                with_layer_norm=False,
                activate_final=True)

    def __call__(self, senders: jnp.ndarray, receivers: jnp.ndarray,
                 eigenvalues: jnp.ndarray, eigenvectors: jnp.ndarray,
                 is_training: bool, call_args=None):
        r"""
        Args:
            senders (jnp.ndarray): indices of the senders nodes.
            receivers (jnp.ndarray): indices of the receivers nodes.
            eigenvalues (torch.Tensor): Eigenvalues of the Laplacian matrix with shape :obj:`[K,]`.
            eigenvectors (torch.Tensor): Eigenvectors of the Laplacian matrix with shape :obj:`[N, K]`.
            is_training (bool): Whether the model is in training mode.
        Returns:
            torch.Tensor: Encoded features with shape :obj:`[N, d_model_aggr]`.
        """
        padding_mask = (eigenvalues > 0)[..., None, :]
        padding_mask = padding_mask.at[..., 0].set(True)
        attn_padding_mask = padding_mask[..., None] & padding_mask[..., None, :]

        trans_eig = jnp.real(eigenvectors)
        trans_eig = trans_eig[..., None]

        if self.consider_im_part and jnp.iscomplexobj(eigenvectors):
            trans_eig_im = jnp.imag(eigenvectors)[..., None]
            trans_eig = jnp.concatenate((trans_eig, trans_eig_im), axis=-1)

        if self.use_gnn:
            trans = self.element_gnn(
                trans_eig, senders, receivers, None, call_args)
            if self.use_signnet:
                trans_neg = self.element_gnn(
                    -trans_eig, senders, receivers, None, call_args)
            trans += trans_neg
        else:
            trans = self.element_mlp(trans_eig)
            if self.use_signnet:
                trans += self.element_mlp(-trans_eig)

        if self.concatenate_eigenvalues:
            eigenvalues_ = jnp.broadcast_to(eigenvalues[..., None, :],
                                            trans.shape[:-1])
            trans = jnp.concatenate((eigenvalues_[..., None], trans), axis=-1)

        if self.use_attention:
            if self.norm is not None:
                trans = self.norm()(trans)
            attn = MultiHeadAttention(
                self.num_heads,
                key_size=trans.shape[-1] // self.num_heads,
                value_size=trans.shape[-1] // self.num_heads,
                model_size=trans.shape[-1],
                #w_init_scale=1.0
                w_init=None,
                dropout_p=self.dropout_p,
                with_bias=False
            )
            mhattn = attn(
                trans,
                trans,
                trans,
                mask=attn_padding_mask,
                is_training=is_training)
            if is_training and self.dropout_p > 0:
                mhattn = hk.dropout(hk.next_rng_key(), self.dropout_p, mhattn)
            trans += mhattn

        padding_mask = padding_mask[..., None]
        trans = trans * padding_mask
        trans = trans.reshape(trans.shape[:-2] + (-1,))

        if self.dropout_p and is_training:
            trans = hk.dropout(hk.next_rng_key(), self.dropout_p, trans)

        output = self.re_aggregate_mlp(trans)
        if self.im_aggregate_mlp is None:
            return output

        output_im = self.im_aggregate_mlp(trans)
        output = output + 1j * output_im
        return output

class _GNN(hk.Module):
    def __init__(self,
                 d_model: int = 256,
                 activation: Callable[[jnp.ndarray], jnp.ndarray] = jax.nn.relu,
                 gnn_type='gcn',
                 use_edge_attr=True,
                 k_hop=2,
                 mlp_layers: int = 2,
                 tightening_factor: int = 1,
                 norm=None,
                 concat: bool = False,
                 residual: bool = True,
                 bidirectional: bool = True,
                 name: Optional[str] = None):
        super().__init__()
        self.d_model = d_model
        self.mlp_layers = mlp_layers
        self.tightening_factor = tightening_factor
        self.activation = activation
        self.gnn_type = gnn_type
        self.use_edge_attr = use_edge_attr
        self.k_hop = k_hop
        self.norm = norm
        self.concat = concat
        self.residual = residual
        self.bidirectional = bidirectional

    def _layer(self, idx: int):
        if self.gnn_type == 'gcn':
            layer = GCNConv(self.d_model // self.tightening_factor,
                            add_self_loops=False)
        elif self.gnn_type == 'gnn':
            # TODO
            raise NotImplementedError("GNN not implemented from MagLapEncoder")
        else:
            raise ValueError(f'Invalid GNN type: {self.gnn_type}')
        return layer

    def __call__(self,
                 nodes: jnp.ndarray,
                 senders: jnp.ndarray,
                 receivers: jnp.ndarray,
                 edges: jnp.ndarray,
                 call_args=None):
        if self.k_hop == 0:
            return nodes
        nodes_list = [nodes]
        if self.bidirectional:
            if edges is not None:
                senders, receivers, edges = to_undirected(senders, receivers, edges)
            else:
                senders, receivers = to_undirected(senders, receivers, edges)
        for idx in range(self.k_hop):
            new_nodes = self._layer(idx)(nodes, senders, receivers, edges)
            if self.residual and self.tightening_factor == 1:
                nodes += new_nodes
                # Note that we dont have new_edges here since GCNConv does not update edge features.
                # edges += new_edges
            else:
                nodes = new_nodes
            if self.concat:
                nodes_list.append(nodes)
        if self.concat:
            nodes = jnp.concatenate(nodes_list, axis=-1)
        if self.norm:
            nodes = self.norm(nodes, call_args)
        if self.concat or self.tightening_factor > 1:
            nodes = hk.Linear(self.d_model)(nodes)
        return nodes


class MultiHeadAttention(hk.Module):
  """Multi-headed attention (MHA) module.
  This module extends the haiku implementation by optional biases in the
  linear transofrmrations and dropout_p on the attention matrix.
  Rough sketch:
  - Compute keys (K), queries (Q), and values (V) as projections of inputs.
  - Attention weights are computed as W = softmax(QK^T / sqrt(key_size)).
  - Output is another projection of WV^T.
  For more detail, see the original Transformer paper:
    "Attention is all you need" https://arxiv.org/abs/1706.03762.
  Glossary of shapes:
  - T: Sequence length.
  - D: Vector (embedding) size.
  - H: Number of attention heads.
  """

  def __init__(
      self,
      num_heads: int,
      key_size: int,
      w_init: Optional[hk.initializers.Initializer] = None,
      value_size: Optional[int] = None,
      model_size: Optional[int] = None,
      dropout_p: float = 0.2,
      with_bias: bool = False,
      re_im_separate_projection: bool = False,
      name: Optional[str] = None,
  ):
    """Initialises the module.
    Args:
      num_heads: Number of independent attention heads (H).
      key_size: The size of keys (K) and queries used for attention.
      w_init: Initialiser for weights in the linear map.
      value_size: Optional size of the value projection (V). If None, defaults
        to the key size (K).
      model_size: Optional size of the output embedding (D'). If None, defaults
        to the key size multiplied by the number of heads (K * H).
      dropout_p: dropout_p after softmax of attention matrix.
      with_bias: if false (default), the linear projects will not have a bias.
      re_im_separate_projection: if true real and imaginary components are
        projected without weight sharing.
      name: Optional name for this module.
    """
    super().__init__(name=name)
    self.num_heads = num_heads
    self.key_size = key_size
    self.value_size = value_size or key_size
    self.model_size = model_size or key_size * num_heads
    self.dropout_p = dropout_p
    self.with_bias = with_bias
    self.re_im_separate_projection = re_im_separate_projection

    self.w_init = w_init

  def __call__(
      self,
      query: jnp.ndarray,
      key: jnp.ndarray,
      value: jnp.ndarray,
      is_training: bool,
      logit_offset: Optional[jnp.ndarray] = None,
      mask: Optional[jnp.ndarray] = None,
  ):
    """Computes (optionally masked) MHA with queries, keys & values.
    This module broadcasts over zero or more 'batch-like' leading dimensions.
    Args:
      query: Embeddings sequence used to compute queries; shape [..., T', D_q].
      key: Embeddings sequence used to compute keys; shape [..., T, D_k].
      value: Embeddings sequence used to compute values; shape [..., T, D_v].
      is_training: if True (not the default), dropout will not be applied. # #
      logit_offset: Optional offset/bias that is applied right before applying
        the softmax and before the mask for the attention scores (broadcast to
        [..., T', T, D_o]). A head specific linear transformation is applied.
      mask: Optional mask applied to attention weights; shape [..., H=1, T', T]
        or [..., T', T].
    Returns:
      A new sequence of embeddings, consisting of a projection of the
        attention-weighted value projections; shape [..., T', D'].
    """
    # In shape hints below, we suppress the leading dims [...] for brevity.
    # Hence e.g. [A, B] should be read in every case as [..., A, B].
    *leading_dims, sequence_length, _ = query.shape
    projection = self._linear_projection

    # Compute key/query/values (overload K/Q/V to denote the respective sizes).
    query_heads = projection(query, self.key_size, 'query')  # [T', H, Q=K]
    key_heads = projection(key, self.key_size, 'key')  # [T, H, K]
    value_heads = projection(value, self.value_size, 'value')  # [T, H, V]

    # Compute attention weights.
    attn_logits = jnp.einsum('...thd,...Thd->...htT', query_heads, key_heads)
    attn_logits = jnp.real(attn_logits)  # In case the logits are complex
    attn_logits = attn_logits / jnp.sqrt(self.key_size).astype(value.dtype)

    # E.g. to apply relative positional encodings or add edge bias
    if logit_offset is not None:
      logit_offset = hk.Linear(self.num_heads)(logit_offset)
      new_order = list(range(logit_offset.ndim - 3)) + [
          logit_offset.ndim - 1, logit_offset.ndim - 3, logit_offset.ndim - 2
      ]
      logit_offset = logit_offset.transpose(*new_order)
      attn_logits = attn_logits + logit_offset

    if mask is not None:
      if mask.ndim == attn_logits.ndim - 1:
        mask = mask[..., None, :, :]
      elif mask.ndim != attn_logits.ndim:
        raise ValueError(
            f'Mask dimensionality {mask.ndim} must match logits dimensionality '
            f'{attn_logits.ndim}.')
      attn_logits = jnp.where(mask, attn_logits, -1e30)
    attn_weights = jax.nn.softmax(attn_logits)  # [H, T', T]

    if is_training and self.dropout_p > 0:
      attn_weights = hk.dropout(hk.next_rng_key(), self.dropout_p, attn_weights)

    # Weight the values by the attention and flatten the head vectors.
    attn = jnp.einsum('...htT,...Thd->...thd', attn_weights, value_heads)
    attn = jnp.reshape(attn, (*leading_dims, sequence_length, -1))  # [T', H*V]

    # Apply another projection to get the final embeddings.
    final_projection = hk.Linear(self.model_size, w_init=self.w_init)
    return final_projection(attn)  # [T', D']

  Tensor = jnp.ndarray

  @hk.transparent
  def _linear_projection(
      self,
      x: Tensor,
      head_size: int,
      name: Optional[str] = None,
  ) -> Tensor:
    lin = hk.Linear(
        self.num_heads * head_size,
        w_init=self.w_init,
        name=name,
        with_bias=self.with_bias)
    if jnp.iscomplexobj(x):
      if self.re_im_separate_projection:
        y_re = lin(jnp.real(x))
        lin_im = hk.Linear(
            self.num_heads * head_size,
            w_init=self.w_init,
            name=name,
            with_bias=self.with_bias)
        y_im = lin_im(jnp.imag(x))
      else:
        y_re = lin(jnp.real(x))
        y_im = lin(jnp.imag(x))
      y = y_re + 1j * y_im
    else:
      y = lin(x)
    *leading_dims, _ = x.shape
    return y.reshape((*leading_dims, self.num_heads, head_size))