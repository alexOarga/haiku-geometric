# This is just a transformer layer.
# Adapted from Haiku examples: github.com/deepmind/dm-haiku/blob/main/examples/transformer
import dataclasses
from typing import Optional
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np


def layer_norm(x: jnp.ndarray) -> jnp.ndarray:
    """Applies a unique LayerNorm to x with default settings."""
    ln = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)
    return ln(x)


class Transformer(hk.Module):
    r"""Transformer layer from the `"Attention is all you need" <https://arxiv.org/abs/1801.07829>`_ paper.
    Where each layer computes the following function:
    
    .. math::
        h &= \mathrm{LayerNorm}(x) \\
        h_a &= x + \mathrm{Dropout}(\mathrm{MultiHeadAttention}(h, h, h)) \\
        \mathrm{Transformer}(h_a) &= h_a + \mathrm{Dropout}(\mathrm{DenseBlock}(\mathrm{LayerNorm}(h_a)))

    """
    def __init__(
            self,
            num_heads: int,
            num_layers: int,
            key_size: int,
            dropout_rate: float,
            widening_factor: int = 4):
        r"""
        Args:
            num_heads (int): Number of attention heads.
            num_layers (int): Number of layers.
            key_size (int): Size of the key and query vector.
            dropout_rate (float): Dropout rate.
            widening_factor (int, optional): Widening factor for the DenseBlock.
                (default: :obj:`4`)
        """
        super().__init__()
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.key_size = key_size
        self.dropout_rate = dropout_rate
        self.widening_factor = widening_factor


    def __call__(
            self,
            embeddings: jnp.ndarray,  # [B, T, D]
            *,
            mask: jnp.ndarray = None,  # [B, T]
            is_training: bool = True,
    ) -> jnp.ndarray:  # [B, T, D]
        r"""Transforms input embedding sequences to output embedding sequences.
        
        Args:
            embeddings (jnp.ndarray): Input embedding sequences of shape :obj:`[B, T, D]`, where :obj:`B` is the batch size, :obj:`T` is the sequence length, and :obj:`D` is the embedding dimension.
            mask (jnp.ndarray, optional): Mask for the input embedding sequences of shape :obj:`[B, T]`
                (default: :obj:`None`).
            is_training (bool, optional): Whether the model is in training mode or not 
                (default: :obj:`True`).
        """
        initializer = hk.initializers.VarianceScaling(2 / self.num_layers)
        dropout_rate = self.dropout_rate if is_training else 0.
        _, seq_len, model_size = embeddings.shape

        h = embeddings
        for _ in range(self.num_layers):
            # First the attention block.
            attn_block = hk.MultiHeadAttention(
                num_heads=self.num_heads,
                key_size=self.key_size,
                model_size=model_size,
                w_init=initializer,
            )
            h_norm = layer_norm(h)
            h_attn = attn_block(h_norm, h_norm, h_norm, mask=mask)
            h_attn = hk.dropout(hk.next_rng_key(), dropout_rate, h_attn)
            h = h + h_attn

            # Then the dense block.
            dense_block = hk.Sequential([
                hk.Linear(self.widening_factor * model_size, w_init=initializer),
                jax.nn.gelu,
                hk.Linear(model_size, w_init=initializer),
            ])
            h_norm = layer_norm(h)
            h_dense = dense_block(h_norm)
            h_dense = hk.dropout(hk.next_rng_key(), dropout_rate, h_dense)
            h = h + h_dense

        return layer_norm(h)
