import haiku as hk
import jax
import warnings
import jax.numpy as jnp
from typing import Optional


class SelfAttention(hk.MultiHeadAttention):
    """Self attention with a causal mask applied."""

    def __call__(
            self,
            query: jnp.ndarray,
            key: Optional[jnp.ndarray] = None,
            value: Optional[jnp.ndarray] = None,
            mask: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        """"""
        warnings.warn(
            "SelfAttention will be removed"
            "Please use haiku.MultiHeadAttention.",
            DeprecationWarning,
        )

        key = key if key is not None else query
        value = value if value is not None else query

        seq_len = query.shape[1]
        causal_mask = jnp.tril(jnp.ones((seq_len, seq_len)))
        mask = mask * causal_mask if mask is not None else causal_mask
        mask_shape = mask.shape
        mask = mask.reshape(1, mask_shape[0], mask_shape[1])

        return super().__call__(query, key, value, mask)