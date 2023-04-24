import haiku as hk
import jax
import jax.numpy as jnp
from typing import Callable, Any, Optional
from haiku_geometric.nn import GCNConv
from haiku_geometric.nn import Transformer
from haiku_geometric.models import MLP
from haiku_geometric.utils import to_undirected, eigv_laplacian


class LaplacianEncoder(hk.Module):
    r"""
    Graph Laplacian Positional Encoder described in the 
    `"Rethinking Graph Transformers with Spectral Attention" <https://arxiv.org/abs/1907.08610>`_ paper.

    Usage::

        from haiku_geometric.utils import eigv_laplacian
        from haiku_geometric.posenc import LaplacianEncoder

        # Compute the eigenvectors of the Laplacian matrix
        eigenvalues, eigenvectors = eigv_laplacian(
            senders=senders,
            receivers=receivers,
            k=...)

        # The function that you will transform with Haiku
        def your_forward_function(...):

            # Create the encoder model
            model = LaplacianEncoder(...)

            # Encode the eignevalues and eigenvectors
            h = model(eigenvalues, eigenvectors, is_training)

    """
    def __init__(self,
                 dim: int,
                 model: str,
                 model_dropout: float = 0.0,
                 layers: int = 1,
                 heads: int = 1,
                 post_layers: int = 1,
                 norm: str = None,
                 norm_decay=0.9
                 ):
        r"""
        Args:
            dim (int): Dimension of the output features.
            model (str): Model to use for the encoder. Can be either :obj:`"Transformer"` or :obj:`"DeepSet"`.
            model_dropout (float, optional): Dropout rate for the model.
                (default: :obj:`0.0`).
            layers (int, optional): Number of layers for the model.
                (default: :obj:`1`).
            heads (int, optional): Number of heads for the model. Only used if :obj:`model="Transformer"`.
                (default: :obj:`1`).
            post_layers (int, optional): Number of post layers after the model.
                (default: :obj:`1`).
            norm (str, optional): Normalization layer to use. Can be either :obj:`"batchnorm"` or :obj:`None`.
                (default: :obj:`None`).
            norm_decay (float, optional): Decay rate for the normalization layer.
                (default: :obj:`0.9`).
        """
        super().__init__()
        self.dim = dim
        if model not in ['Transformer', 'DeepSet']:
            raise ValueError(f"Unexpected model {model}")
        self.model = model
        self.model_dropout = model_dropout
        self.n_layers = layers
        self.heads = heads
        self.post_layers = post_layers
        self.norm = norm

        self.linear1 = hk.Linear(dim)
        if norm == "batchnorm":
            self.norm_layer = hk.BatchNorm(True, True, norm_decay)
        else:
            self.norm_layer = None

        activation = jax.nn.relu

        if model == 'Transformer':
            self.encoder_layer = Transformer(
                heads,
                layers,
                dim,
                model_dropout)
        elif model == 'DeepSet':
            layers = []
            if self.n_layers == 1:
                layers.append(activation)
            else:
                self.linear1 = hk.Linear(2 * dim)
                layers.append(activation)
                for _ in range(self.n_layers - 2):
                    layers.append(hk.Linear(2 * dim))
                    layers.append(activation)
                layers.append(hk.Linear(dim))
                layers.append(activation)
            self.encoder_layer = hk.Sequential(layers)

        self.post_mlp = None
        if self.post_layers > 0:
            layers = []
            if self.post_layers == 1:
                layers.append(hk.Linear(dim))
                layers.append(activation)
            else:
                layers.append(hk.Linear(2 * dim))
                layers.append(activation)
                for _ in range(self.post_layers - 2):
                    layers.append(hk.Linear(2 * dim))
                    layers.append(activation)
                layers.append(hk.Linear(dim))
                layers.append(activation)
            self.post_mlp = hk.Sequential(layers)


    def __call__(self, eigenvalues: jnp.ndarray, eigenvectors: jnp.ndarray,
                 is_training: bool, call_args=None):
        r"""
        Args:
            eigenvalues (torch.Tensor): Eigenvalues of the Laplacian matrix with shape :obj:`[K,]`.
            eigenvectors (torch.Tensor): Eigenvectors of the Laplacian matrix with shape :obj:`[N, K]`.
            is_training (bool): Whether the model is in training mode.
        Returns:
            torch.Tensor: Encoded features with shape :obj:`[N, dim]`.
        """

        if is_training:
            sign_flip = jax.random.uniform(hk.next_rng_key(), shape=(eigenvectors.shape[1],))
            sign_flip.at[sign_flip >= 0.5].set(1.0)
            sign_flip.at[sign_flip < 0.5].set(-1.0)
            eigenvectors = eigenvectors * jnp.expand_dims(sign_flip, axis=0)

        def _expand_repeat_eigenvalues(evals, N):
            """Reshapes eigenvalues from (k,) to (N, K, 1)"""
            evals = jnp.expand_dims(evals, axis=0)
            evals = jnp.repeat(evals, N, axis=0)
            evals = jnp.expand_dims(evals, axis=2)
            return evals

        expanded_vec = jnp.expand_dims(eigenvectors, axis=2)
        expanded_val = _expand_repeat_eigenvalues(eigenvalues, eigenvectors.shape[0])
        pos_enc = jnp.concatenate((expanded_vec, expanded_val), axis=2)
        empty_mask = jnp.isnan(pos_enc)
        pos_enc.at[empty_mask].set(0)

        if self.norm:
            pos_enc = self.norm_layer(pos_enc, is_training=is_training)
        pos_enc = self.linear1(pos_enc)

        padding_mask = ~jnp.isnan(pos_enc)  # [N, K, D]
        padding_mask = padding_mask[..., 0]  # [N, K] # TODO: is it correct to select only the value at zero?
        aux1 = padding_mask[..., None]  # [N, K, 1]
        aux2 = padding_mask[..., None, :] # [N, 1, K]
        padding_mask = aux1 & aux2  # [N, K, K]
        padding_mask = padding_mask[..., None, :, :]  # [N, 1, K, K]

        if self.model == 'Transformer':
            pos_enc = self.encoder_layer(
                pos_enc,
                mask=padding_mask,
                is_training=is_training
            )
        else:
            pos_enc = self.encoder_layer(pos_enc)
            if is_training:
                pos_enc = hk.dropout(hk.next_rng_key(), self.model_dropout, pos_enc)

        pos_enc.at[empty_mask[:, :, 0]].set(0.)
        pos_enc = jnp.sum(pos_enc, axis=1, keepdims=False)

        if self.post_mlp is not None:
            pos_enc = self.post_mlp(pos_enc)

        return pos_enc