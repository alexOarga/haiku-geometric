import haiku as hk
import jax
import jax.numpy as jnp


def normalize_features(x: jnp.ndarray) -> jnp.ndarray:
    """Normalizes node features to sum to 1 on the last axis.
    
    Args:
        x (jnp.ndarray): Array of node features.
    """
    x = x / (jnp.clip(x.sum(axis=-1, keepdims=True), a_min=1.0, a_max=None))
    return x
