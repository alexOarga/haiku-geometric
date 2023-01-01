import jax.numpy as jnp
from jraph._src.utils import segment_sum


def degree(index: jnp.ndarray, total_num_nodes):
    d = jnp.ones_like(index)
    return segment_sum(
        d, index, total_num_nodes)